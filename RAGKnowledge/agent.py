"""
Agentic RAG 主流程模块。

实现特点（相对纯检索直答）：

- **早退路径**：先判断是否需要检索；不需要时直接走 LLM/VLM。
- **主循环**（最多约 3 轮有效扩展）：扩展/生成子问题 → 多路检索 →
  并行摘要各子问题召回 → `generate_next_instruction` 判断是否可答；
  若不可答则使用 `rewrite_query` 进入下一轮。
- **收尾**：合并去重多类型 chunk → 文本 cross-encoder 重排 → 按是否有页面图块
  选择纯文本 LLM、多模态 VLM 或兜底 `fast_answer`。

调试辅助：`beautify_messages`（截断展示多模态消息）、`display_chunks`（打印 chunk 预览）。
"""
import concurrent.futures
from typing import List, Dict, Tuple

from .query.query_processer import QueryProcessor
from .prompt.prompt import get_user_query_prompt, get_answer_prompt
from .utils.llm import get_llm
from .utils.vlm import get_vlm
from .utils.image2caption import get_image2caption_model
from .utils.rerank import get_rerank
from .retrieval.retriever import Retriever
from .utils.logger_utils import logger
from .utils.time_utils import time_it


class AgenticRAG:
    """
    面向指定知识库 ``kb_id`` 的 Agentic RAG 入口。

    Args:
        kb_id: 向量库检索时使用的知识库标识。
        n_round: 预留的最大轮次参数。当前逻辑主要由 ``loop > 3`` 控制，
            后续可与 ``_n_round`` 做统一。
    """

    def __init__(self, kb_id: str, n_round: int = 5):
        self._n_round = n_round
        self._retriever = Retriever()
        self._kb_id = kb_id

    @staticmethod
    def llm_answer(question: str):
        """无检索文本问答，直接调用大模型生成答案。"""
        prompt = get_user_query_prompt().get_default_prompt(question=question)
        response = get_llm().llm_big(prompt)
        return response

    @staticmethod
    def vlm_answer(question: str, image_path: str):
        """
        无检索视觉问答，根据图片直接回答问题。
        """
        prompt = f"根据图片回答问题：{question}"
        response = get_vlm().vlm("Qwen/Qwen3.5-397B-A17B", prompt, image_path)
        return response

    def fast_answer(self, question: str, image_path: str = None):
        """兜底快速回答：无图走 LLM，有图走 VLM。"""
        if not image_path:
            return self.llm_answer(question)
        else:
            return self.vlm_answer(question, image_path)

    @staticmethod
    def merge_retrieval_results(resp: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        将多轮、多路检索的 chunk 列表按类型去重合并，并按 score 降序排序。

        - **text**：以 ``payload['file_sorted']`` 为键，同键保留最后一次写入（通常为更高分覆盖）。
        - **image / ocr_text / caption**：优先 ``image_id``，否则 ``page_id`` 归入 page 侧 map。
        - **page**：以 ``page_path`` 为键。

        Returns:
            三元组 ``(text_chunks, image_chunks, page_chunks)``。

        Notes:
            调试时会打印图片块与页面块的路径与分数（``build_text_context`` 当前注释掉）。
        """
        # 按 chunk_type 分桶，用业务键去重
        text_chunk_map = {}
        image_chunk_map = {}
        page_chunk_map = {}
        for ret in resp:
            chunk_type = ret['payload']['chunk_type']
            if chunk_type == "text":
                # print(ret)
                key = ret['payload']['file_sorted']
                text_chunk_map[key] = ret
            elif chunk_type == "image" or chunk_type == "ocr_text" or chunk_type == "caption":
                # print("image: ", ret)
                if "image_id" in ret['payload']:
                    key = ret['payload']['image_id']
                    image_chunk_map[key] = ret
                elif "page_id" in ret['payload']:
                    key = ret['payload']['page_id']
                    page_chunk_map[key] = ret
            elif chunk_type == "page":
                key = ret['payload']['page_path']
                page_chunk_map[key] = ret

        text_chunks = list(sorted(text_chunk_map.values(), key=lambda k: k['score'], reverse=True))
        image_chunks = list(sorted(image_chunk_map.values(), key=lambda k: k['score'], reverse=True))
        page_chunks = list(sorted(page_chunk_map.values(), key=lambda k: k['score'], reverse=True))

        def build_text_context():
            context = "文本检索内容：\n"
            for doc in text_chunks:
                context += doc["payload"]['text'][:100] + "\n"

            print(context)

        def build_image_context():
            context = ""
            for doc in image_chunks:
                context += f'{doc["payload"]["image_path"]} {doc["score"]}' + "\n"
            print(context)

        def build_page_context():
            context = ""
            for doc in page_chunks:
                context += f'{doc["payload"]["page_path"]} {doc["score"]}' + "\n"
            print(context)

        # build_text_context()
        build_image_context()
        build_page_context()

        return text_chunks, image_chunks, page_chunks

    @staticmethod
    def build_ref_context(docs: List[Dict]):
        """
        将文本类检索结果拼成带引用边界的上下文，供 ``TEXT_PROMPT`` / ``IMAGE_PROMPT`` 使用。

        格式：``[ref i start]`` ... ``[ref i end]``，仅包含 ``payload`` 中带 ``text`` 的项。
        """
        context = ""
        for i, doc in enumerate(docs):
            if doc['payload'].get("text"):
                context += f"\n[ref {i + 1} start]\n{doc['payload']['text']}\n[ref {i + 1} end]\n"
        return context

    @time_it
    def run(self, question: str, image_path: str = None):
        """
        Agentic RAG 主入口。

        流程概要：
        1) 可选图像描述；
        2) 简单问句/图问答早退；
        3) 多轮子问题扩展 + 检索 + 摘要 + 可答判断；
        4) 合并去重与重排；
        5) 选择 LLM / VLM 完成最终作答。

        Args:
            question: 用户问题。
            image_path: 可选图片路径。

        Returns:
            最终回答文本。
        """
        logger.info(f"AIAgent正在处理问题: {question}, 图片: {image_path}")
        if image_path:
            image_desc = get_image2caption_model().image2caption(image_path)
        else:
            image_desc = ""

        # ------------------------------ 无需 RAG 检索的早退路径 ------------------------------

        # 无需检索的简单文本对话：直接 LLM
        check_rag = QueryProcessor.rag_check(question)
        if not check_rag:
            return self.llm_answer(question)

        # 图文且无需检索的对话：直接 VLM
        check_image_rag = QueryProcessor.image_rag_check(question, image_desc)
        if not check_image_rag:
            return self.vlm_answer(question, image_path)

        # ------------------------------ RAG 检索阶段 ------------------------------

        loop = 1
        # Agentic RAG 主循环（最多 3 轮）
        answer_question = question

        total_sub_questions = []
        total_sub_summaries = []

        total_chunks = []

        while True:
            logger.info(f"第{loop}轮RAG检索查询")
            # ------------------------------ 生成子查询 ------------------------------
            if loop == 1 and image_path:
                sub_questions = QueryProcessor.expand_question_with_images(answer_question, image_desc)
            else:
                sub_questions = QueryProcessor.extend_questions(answer_question)

            if loop == 1:
                sub_questions.insert(0, question)

            total_sub_questions.extend(sub_questions)

            # ------------------------------ 子问题四路并行检索 ------------------------------
            logger.info("开始多路检索阶段")
            current_chunks = self._retriever.retrieval_by_texts(self._kb_id, sub_questions)

            for query_chunks in current_chunks:
                for chunk in query_chunks:
                    # logger.info(f"检索结果: {chunk['payload']}")
                    total_chunks.append(chunk)

            loop += 1
            # 控制展开深度：loop 自增后大于 3 则停止迭代。
            if loop > 3:
                break

            # ------------------------------ 对每个 (子问题, 对应召回) 并行摘要 ------------------------------
            tasks = {}
            summarized_infos = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                for sub_question, query_chunks in zip(sub_questions, current_chunks):
                    task = executor.submit(QueryProcessor.summarize_subquery, sub_question, query_chunks)
                    tasks[task] = sub_question

                for future in concurrent.futures.as_completed(tasks):
                    sub_question = tasks[future]
                    try:
                        result = future.result()
                        logger.info(f"总结结果: {result}")
                        summarized_infos[sub_question] = result
                    except Exception as e:
                        logger.error(f"总结生成失败: {sub_question}: {e}")

            # 按 sub_questions 原顺序写入，保持与后续指令生成的一致性。
            for sub_question in sub_questions:
                total_sub_summaries.append(summarized_infos[sub_question])

            # ------------------------------ 判断当前检索+摘要是否足够作答 ------------------------------
            next_instruction = QueryProcessor.generate_next_instruction(question, total_sub_questions,
                                                                        total_sub_summaries)

            if next_instruction['is_answer']:
                break
            else:
                answer_question = next_instruction['rewrite_query']

        # ------------------------------ 合并去重、分类型排序 ------------------------------
        text_chunks, image_chunks, page_chunks = self.merge_retrieval_results(total_chunks)

        logger.info(
            f"RAG检索结果: 文本: {len(text_chunks)}, 图片: {len(image_chunks)}, 页面: {len(page_chunks)}")

        # 页面级证据只保留 Top-1，控制 VLM 输入规模与成本
        page_chunks = page_chunks[:1]

        # ------------------------------ 文本重排 ------------------------------
        logger.info("开始重排阶段")
        texts = [text_chunk['payload']['text'] for text_chunk in text_chunks]
        scores = get_rerank().rerank(question, texts)

        for text_chunk, score in zip(text_chunks, scores):
            text_chunk['score'] = score

        text_chunks = sorted(text_chunks, key=lambda k: k['score'], reverse=True)

        # 打印检索 chunk 的序号、score 与正文前 100 字，用于调试检索/重排效果。
        for i, chunk in enumerate(text_chunks):
            logger.info(f"=======================Chunk {i}: ")
            logger.info(f"score: {chunk['score']}")
            logger.info(f"chunk: {chunk['payload']['text'][:100]}")

        # 去除低相关文本，避免低质量上下文干扰最终回答。
        text_chunks = [text_chunk for text_chunk in text_chunks if text_chunk['score'] > 0.3]

        # ------------------------------ 回答阶段 ------------------------------

        if not text_chunks:
            logger.info("没有找到文本检索结果")

            if page_chunks:
                logger.info("使用图片问答")
                image_path = [page_chunk['payload']['image_path'] for page_chunk in page_chunks]
                return self.vlm_answer(question, image_path[0])

            logger.info("使用LLM回答")
            return self.fast_answer(question, image_path)

        context = self.build_ref_context(text_chunks)

        if not page_chunks:
            logger.info("没有找到图片, 使用LLM回答")
            prompt = get_answer_prompt().get_text_answer_prompt(context=context, question=question)
            return get_llm().llm_big(prompt)

        logger.info("使用多模态模型回答")
        prompt = get_answer_prompt().get_image_answer_prompt(context=context, question=question)
        image_paths = [page_chunk['payload']['image_url'] for page_chunk in page_chunks]

        image_path = image_paths[0]

        response = get_vlm().vlm("Qwen/Qwen3.5-397B-A17B" ,prompt, image_path)

        return response
