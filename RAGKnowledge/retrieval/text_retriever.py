"""
文本检索模块。

当前实现聚焦两种召回能力：
1. 稠密向量召回（语义相似度检索）；
2. 稀疏向量召回（BM25 关键词检索）。

两种召回均支持通过 ``kb_id`` 与额外过滤条件限制检索范围，
并统一委托给向量数据库层执行搜索。
"""

from typing import Dict, Optional

from ..embedding.text_bm25 import get_bm25_embedding_model
from ..embedding.text_embedding import get_text_embedding_model
from ..vector_db.db import VectorDB
from ..utils.logger_utils import logger


class TextRetriever:
    """文本检索器。

    该类负责把自然语言查询编码为检索向量，并调用底层向量库执行搜索。
    """

    def __init__(self):
        self.embedding_model = get_text_embedding_model()
        self.vector_store = VectorDB()
        self.bm25_embedding_model = get_bm25_embedding_model()

    def vector_search(self, kb_id: str, queries: list[str], limit: int = 10, score_threshold: float = 0.0,
                      filter_conditions: Optional[Dict] = None
                      ) -> list[list[dict]]:
        """执行稠密向量（语义）召回。

        Args:
            kb_id: 知识库 ID。会写入过滤条件，确保只在指定知识库中检索。
            queries: 查询文本列表，每条查询会编码为一条稠密向量。
            limit: 每条查询返回的最大结果数。
            score_threshold: 相似度分数下限，低于该阈值的结果会被过滤。
            filter_conditions: 额外过滤条件（会与 ``kb_id`` 合并）。

        Returns:
            二维结果列表。外层与 ``queries`` 一一对应，内层为单条查询的命中文档列表。
        """
        logger.info(f"正在对: {queries} 进行向量相似度召回, 过滤条件: {filter_conditions}")
        if not filter_conditions:
            filter_conditions = {}
        # 强制限定知识库范围，避免跨库命中。
        filter_conditions.update({"kb_id": kb_id})
        logger.info(f"过滤条件: {filter_conditions}")
        query_vectors = self.embedding_model.encode_text_batch(queries)
        return self.vector_store.search_text_vector(
            query_vectors=query_vectors,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions
        )

    def sparse_search(self, kb_id: str, queries: list[str], limit: int = 10, score_threshold: float = 0.0,
                      filter_conditions: Optional[Dict] = None):
        """执行稀疏向量（BM25）召回。

        Args:
            kb_id: 知识库 ID。会写入过滤条件，确保只在指定知识库中检索。
            queries: 查询文本列表，每条查询会编码为 BM25 稀疏向量。
            limit: 每条查询返回的最大结果数。
            score_threshold: 召回分数下限，低于该阈值的结果会被过滤。
            filter_conditions: 额外过滤条件（会与 ``kb_id`` 合并）。

        Returns:
            关键词检索结果，结构由 ``vector_store.keyword_search`` 定义。
        """
        logger.info(f"正在对进行稀疏向量召回...")
        if not filter_conditions:
            filter_conditions = {}
        # 强制限定知识库范围，避免跨库命中。
        filter_conditions.update({"kb_id": kb_id})
        query_vectors = self.bm25_embedding_model.encode_text_batch(queries)
        print(query_vectors)
        return self.vector_store.keyword_search(
            queries=queries,
            sparse_vectors=query_vectors,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions
        )
