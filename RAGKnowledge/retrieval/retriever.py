"""
统一检索路由模块。

该模块将多路检索器（文本稠密、文本稀疏、文本到图像、文本到页面）并发执行，
并把结果按查询维度合并，作为后续重排与摘要阶段的输入。
"""
import concurrent.futures
import os

from .image_retriever import ImageRetriever
from .text_retriever import TextRetriever


class Retriever:
    """统一检索器。

    负责协调多路召回并聚合结果，不直接实现具体的向量检索算法。
    """

    def __init__(self):
        self._image_retriever = ImageRetriever()
        self._text_retriever = TextRetriever()

    def retrieval_by_texts(self, kb_id: str, queries: list[str]) -> list[list[dict]]:
        """按文本查询列表执行多路并发召回并合并结果。

        Args:
            kb_id: 知识库 ID，用于限定召回范围。
            queries: 查询文本列表。每个元素会在四路检索中并行执行。

        Returns:
            二维结果列表。外层与 ``queries`` 一一对应，内层为该查询合并后的命中结果。

        Notes:
            - 通过 ``ThreadPoolExecutor(max_workers=4)`` 并发执行四路检索；
            - 使用 ``as_completed`` 按完成先后收集结果，因此不同路的拼接顺序非固定；
            - 最终会等待四路任务全部结束后再返回。
        """

        tasks = []
        # 每个 query 对应一个结果槽位，后续把多路召回结果 append 进同一槽位。
        res = [[] for _ in range(len(queries))]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 1) 文本稠密向量召回（语义相似）
            task = executor.submit(self._text_retriever.vector_search, kb_id, queries,
                                   score_threshold=float(os.getenv("RETRIEVAL_TEXT_THRESHOLD")))
            tasks.append(task)

            # 2) 文本稀疏向量召回（关键词/BM25）
            task = executor.submit(self._text_retriever.sparse_search, kb_id, queries)
            tasks.append(task)

            # 3) 文本到图像召回
            task = executor.submit(self._image_retriever.text2image_search, kb_id, queries,
                                   score_threshold=float(os.getenv("RETRIEVAL_IMAGE_THRESHOLD")))
            tasks.append(task)
            
            # 4) 文本到页面召回
            task = executor.submit(self._image_retriever.text2page_search, kb_id, queries,
                                   score_threshold=float(os.getenv("RETRIEVAL_PAGE_THRESHOLD")))
            tasks.append(task)

            # 按任务完成顺序收集结果；每路返回结构都应与 queries 对齐。
            for task in concurrent.futures.as_completed(tasks):
                current_res = task.result()
                if current_res:
                    for i, query in enumerate(queries):
                        res[i].extend(current_res[i])

        return res