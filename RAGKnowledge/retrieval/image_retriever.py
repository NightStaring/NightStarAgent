"""
图像检索模块。

当前实现支持四类检索：
1. 文本到图像（Text -> Image）；
2. 图像到图像（Image -> Image）；
3. 文本到页面（Text -> Page）；
4. 图像到页面（Image -> Page）。

所有检索都会将输入编码为向量，并调用向量数据库执行相似度搜索。
"""
from typing import Dict, Optional

from PIL import Image

from ..embedding.image_embedding import get_image_embedding_model
from ..vector_db.db import VectorDB
from ..utils.logger_utils import logger


class ImageRetriever:
    """图像检索器。

    负责将文本或图像编码为向量，并在图像集合或页面集合中执行召回。
    """

    def __init__(self):
        self.embedding_model = get_image_embedding_model()
        self.vector_store = VectorDB()

    def text2image_search(self, kb_id: str, queries: list[str], limit: int = 10, score_threshold: float = 0.0,
                          filter_conditions: Optional[Dict] = None):
        """执行文本到图像检索。

        Args:
            kb_id: 知识库 ID。会写入过滤条件，确保只在指定知识库中检索。
            queries: 文本查询列表，每条查询会编码为图像语义空间向量。
            limit: 每条查询返回的最大结果数。
            score_threshold: 相似度分数下限，低于阈值的结果会被过滤。
            filter_conditions: 额外过滤条件（会与 ``kb_id`` 合并）。

        Returns:
            二维结果列表。外层与 ``queries`` 一一对应，内层为单条查询的命中结果。
        """
        logger.info(f"正在对文本: {queries} 进行图像内容召回, 过滤条件: {filter_conditions}")
        if not filter_conditions:
            filter_conditions = {}
        # 强制限定知识库范围，避免跨库命中。
        filter_conditions.update({"kb_id": kb_id})
        text_embeddings = self.embedding_model.encode_text_batch(queries)
        return self.vector_store.search_image_vector(
            query_vectors=text_embeddings,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions
        )

    def image2image_search(self,
                           kb_id: str, image: Image.Image, limit: int = 10, score_threshold: float = 0.0,
                           filter_conditions: Optional[Dict] = None):
        """执行图像到图像检索。

        Args:
            kb_id: 知识库 ID。会写入过滤条件，确保只在指定知识库中检索。
            image: 待检索图像对象。
            limit: 返回的最大结果数。
            score_threshold: 相似度分数下限，低于阈值的结果会被过滤。
            filter_conditions: 额外过滤条件（会与 ``kb_id`` 合并）。

        Returns:
            单条查询对应的命中结果列表。
        """
        logger.info(f"正在对图像: {image} 进行图像相似度召回, 过滤条件: {filter_conditions}")
        if not filter_conditions:
            filter_conditions = {}
        # 强制限定知识库范围，避免跨库命中。
        filter_conditions.update({"kb_id": kb_id})
        image_embeddings = self.embedding_model.encode_image_batch([image])
        return self.vector_store.search_image_vector(
            query_vectors=image_embeddings,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions
        )[0]

    def text2page_search(self, kb_id: str, queries: list[str], limit: int = 10, score_threshold: float = 0.0,
                         filter_conditions: Optional[Dict] = None):
        """执行文本到页面检索。

        Args:
            kb_id: 知识库 ID。会写入过滤条件，确保只在指定知识库中检索。
            queries: 文本查询列表，每条查询会编码为页面语义空间向量。
            limit: 每条查询返回的最大结果数。
            score_threshold: 相似度分数下限，低于阈值的结果会被过滤。
            filter_conditions: 额外过滤条件（会与 ``kb_id`` 合并）。

        Returns:
            二维结果列表。外层与 ``queries`` 一一对应，内层为单条查询的命中结果。
        """
        logger.info(f"正在对文本: {queries} 进行页面内容召回, 过滤条件: {filter_conditions}")
        if not filter_conditions:
            filter_conditions = {}
        # 强制限定知识库范围，避免跨库命中。
        filter_conditions.update({"kb_id": kb_id})
        text_embeddings = self.embedding_model.encode_text_batch(queries)
        return self.vector_store.search_page_vector(
            query_vectors=text_embeddings,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions
        )

    def image2page_search(self,
                          kb_id: str, image: Image.Image, limit: int = 10, score_threshold: float = 0.0,
                          filter_conditions: Optional[Dict] = None):
        """执行图像到页面检索。

        Args:
            kb_id: 知识库 ID。会写入过滤条件，确保只在指定知识库中检索。
            image: 待检索图像对象。
            limit: 返回的最大结果数。
            score_threshold: 相似度分数下限，低于阈值的结果会被过滤。
            filter_conditions: 额外过滤条件（会与 ``kb_id`` 合并）。

        Returns:
            命中结果列表（对应单条图像查询）。
        """
        logger.info(f"正在对图像: {image} 进行页面内容召回, 过滤条件: {filter_conditions}")
        if not filter_conditions:
            filter_conditions = {}
        # 强制限定知识库范围，避免跨库命中。
        filter_conditions.update({"kb_id": kb_id})
        image_embeddings = self.embedding_model.encode_image_batch([image])
        return self.vector_store.search_page_vector(
            query_vectors=image_embeddings,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions
        )
