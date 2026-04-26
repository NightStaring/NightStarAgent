"""
RAG 侧查询与路由辅助（agent 层）

本文件提供能力：

**QueryProcessor**：基于 LLM 的查询处理流水线中的若干步骤，例如：
   - 查询扩展、结合前置思考与图片描述生成子问题；
   - 将检索到的文档块拼成上下文并做子问题摘要；
   - 根据原问题、子问题与摘要生成下一步 JSON 指令；
   - 轻量级布尔/工具选择校验（短输出、低温度）。

与「完整对话 Agent」不同，这里侧重 **检索前/检索后的文本变换与路由信号**，
具体 Prompt 定义在 `generation.PromptManager` 中。
"""
import json
from typing import List, Dict

from ..prompt.prompt import get_user_query_prompt, get_answer_prompt
from ..utils.llm import get_llm
from ..utils.logger_utils import logger
from ..utils.time_utils import time_it


class QueryProcessor:
    """
    查询处理器，封装 RAG 查询阶段的多步 LLM 调用。

    主要职责包括：
    1. 判断是否需要触发检索；
    2. 生成扩展子问题；
    3. 将检索结果拼接为摘要上下文；
    4. 生成下一步 JSON 指令。

    Notes:
        大多数方法设计为 ``@staticmethod``，便于按流水线方式组合调用，
        并通过 ``@time_it`` 记录关键步骤耗时。
    """

    @staticmethod
    @time_it
    def rag_check(question: str) -> bool:
        """
        仅基于用户问题判断是否需要触发 RAG 检索。

        Args:
            question: 用户输入问题。

        Returns:
            当模型返回 ``"1"`` 时为 ``True``，否则为 ``False``。
        """
        prompt = get_user_query_prompt().get_rag_query_check_prompt(question=question)
        response = get_llm().llm_small(prompt).strip() == "1"

        logger.info(f"是否需要进行RAG检索: {response}")
        return response

    @staticmethod
    @time_it
    def image_rag_check(question: str, image_desc: str) -> bool:
        """
        基于用户问题与图片描述判断是否需要触发检索。

        Args:
            question: 用户输入问题。
            image_desc: 图片描述列表，会被按换行拼接为单段文本。

        Returns:
            当模型返回 ``"2"`` 时为 ``True``，否则为 ``False``。
        """
        prompt = get_user_query_prompt().get_image_rag_query_check_prompt(
            question=question,
            image_desc="\n".join(image_desc)
        )
        response = get_llm().llm_small(prompt).strip() == "2"

        logger.info(f"是否需要进行RAG检索: {response}")
        return response


    @staticmethod
    @time_it
    def get_pre_think_results(question: str) -> str:
        """
        执行前置思考，产出后续扩展查询的上游输入。

        Args:
            question: 用户输入问题。

        Returns:
            前置思考结果，格式为自然语言文本（非 JSON）。
        """
        prompt = get_user_query_prompt().get_pre_think_prompt(question)
        response = get_llm().llm_small(prompt)
        return response

    @staticmethod
    def expand_question_with_images(question: str, image_desc: str) -> List[str]:
        """
        结合前置思考与可选图片描述，生成扩展子问题列表。

        Args:
            question: 用户输入问题。
            image_desc: 可选图片描述，为空时仅使用前置思考结果。

        Returns:
            扩展后的子问题列表。模型返回按行拆分并做去空白处理。

        Notes:
            流程为：前置思考 -> 拼接输入 -> 调用扩展提示词 -> 按行清洗。
        """
        logger.info("开始前置的思考")
        pre_think_results = QueryProcessor.get_pre_think_results(question)
        logger.info(f"前置思考结果: {pre_think_results}")

        logger.info("开始基于图片生成子查询")
        if image_desc:
            full_image_desc = "\n".join(image_desc)
            full_input = f"输入:\n{pre_think_results}\n<图片描述>{full_image_desc}</图片描述>"
        else:
            full_input = f"输入:\n{pre_think_results}"

        prompt = get_user_query_prompt().get_extend_question_prompt(question=full_input)
        response = get_llm().llm_big(prompt)

        logger.info(f"expand_question_with_images: {response}")
        questions = response.split("\n")
        questions = [question.strip() for question in questions if question.strip()]
        return questions

    @staticmethod
    @time_it
    def extend_questions(question: str) -> List[str]:
        """
        将单个问题扩展成多条候选子问题。

        Args:
            question: 用户输入问题。

        Returns:
            子问题列表。模型输出约定为“每行一个子问题”，方法内部会去空白与空行。
        """
        prompt = get_user_query_prompt().get_extend_words_prompt(question)
        response = get_llm().llm_small(prompt)

        questions = response.split("\n")
        return [question.strip() for question in questions if question.strip()]

    @staticmethod
    def build_context(docs: List[Dict]) -> str:
        """
        将检索结果拼接为摘要模型输入上下文。

        仅使用 ``payload.text`` 字段；每段文本包裹片段边界标记，
        便于后续提示词区分不同来源片段。

        Args:
            docs: 检索结果列表，元素中需包含 ``payload``。

        Returns:
            带片段边界标记的上下文字符串。
        """
        context = ""
        for doc in docs:
            if doc['payload'].get("text"):
                context += "[文本片段开始]\n" + doc["payload"]['text'] + "\n[文本片段结束]\n"

        return context

    @staticmethod
    @time_it
    def summarize_subquery(question: str, chunks: List[Dict]) -> str:
        """
        针对单个子问题，对命中块做相关信息摘要。

        Args:
            question: 当前子问题。
            chunks: 当前子问题命中的检索块列表。

        Returns:
            摘要文本结果。
        """
        context = QueryProcessor.build_context(chunks)
        prompt = get_answer_prompt().get_chunk_answer_prompt(context=context, question=question)
        response = get_llm().llm_small(prompt)
        return response

    @staticmethod
    @time_it
    def generate_next_instruction(
        question: str,
        sub_questions: List[str],
        summarized_contexts: List[str]
    ) -> Dict:
        """
        综合原问题、子问题和摘要，生成下一步 JSON 指令。

        Args:
            question: 原始用户问题。
            sub_questions: 子问题列表。
            summarized_contexts: 每个子问题对应的摘要列表。

        Returns:
            解析后的下一步指令字典。

        Raises:
            json.JSONDecodeError: 当模型返回的 JSON 代码块无法反序列化时抛出。

        Notes:
            约定模型输出包含 `````json`` 代码块，本方法会提取代码块内容并反序列化。
        """
        prompt = get_answer_prompt().get_over_check_prompt(
            question=question,
            sub_questions="\n".join(sub_questions),
            context="\n".join(summarized_contexts)
        )
        response = get_llm().llm_big(prompt)

        l_pos = response.find("```json")
        r_pos = response.rfind("```", l_pos + 1)
        response = response[l_pos + 7:r_pos]
        try:
            next_instruction = json.loads(response)
        except Exception as e:
            logger.error(f"生成下一步指令失败: {e}")
            logger.error(f"response: {response}")
            return {}
        logger.info("next_instruction: ", next_instruction)
        return next_instruction
