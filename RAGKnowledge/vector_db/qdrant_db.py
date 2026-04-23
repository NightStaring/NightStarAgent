"""
Qdrant 向量数据库实现模块

架构设计：
1. QdrantVectorStore - 核心类，包含所有 Qdrant 操作的实现
2. QdrantTextVectorStore - 文本向量包装器，委托操作给核心类
3. QdrantGenericVectorStore - 通用向量包装器，用于 Image/Page 等非文本类型

核心功能：
- 向量的增删改查
- 批量向量操作
- 相似度检索
- 索引管理
- 集合(Collection)管理

设计理念：
- 核心逻辑集中在 QdrantVectorStore
- 子类只是轻量级包装器，指定不同的集合和字段配置
- 所有实际操作都委托给核心类
- 避免代码重复，易于维护和扩展
"""

import uuid
from typing import List, Dict, Tuple, Optional, Any

import dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import SparseVector

from .db_base import BaseVectorDB, BaseCollectionVectorDB
from .models.image_chunk_model import ImageChunkModel
from .models.text_chunk_model import TextChunkModel
from ..utils.logger_utils import logger
from ..config.config import get_config

dotenv.load_dotenv()


class QdrantVectorDB(BaseVectorDB):
    """Qdrant 向量数据库实现"""

    def __init__(self, config):
        """
        初始化 Qdrant 客户端

        Args:
            config: Qdrant 配置
        """
        super().__init__(config)
        self.client = self._create_client()
        self.config = get_config().vector_db

    def _create_client(self) -> QdrantClient:
        """创建 Qdrant 客户端"""
        url = self.config.url
        port = self.config.port
        api_key = self.config.api_key
        if isinstance(api_key, str) and not api_key.strip():
            api_key = None

        logger.info(f"Qdrant向量数据库连接 url: {url}, port: {port}")

        client = QdrantClient(
            url=url,
            port=port,
            timeout=30,
            # api_key=api_key,
            https=False,
            prefer_grpc=True,
            check_compatibility=False,
        )
        return client

    def create_collection(self, collection_name: str, **kwargs) -> bool:
        """
        创建 Qdrant 集合

        Args:
            collection_name: 集合名称
            **kwargs: 其他参数
                - vector_size: 向量维度
                - distance: 距离度量 (默认 COSINE)
                - enable_sparse: 是否启用稀疏向量

        Returns:
            bool: 是否创建成功
        """
        try:
            if self.collection_exists(collection_name):
                logger.info(f"集合 {collection_name} 已存在")
                return True

            vector_size = kwargs.get('vector_size', 768)
            distance = kwargs.get('distance', models.Distance.COSINE)
            enable_sparse = kwargs.get('enable_sparse', False)

            # 构建向量配置
            vectors_config = {
                "vector": models.VectorParams(size=vector_size, distance=distance)
            }

            # 稀疏向量配置
            sparse_vectors_config = None
            if enable_sparse:
                sparse_vectors_config = {
                    "sparse_vector": models.SparseVectorParams(modifier=models.Modifier.IDF)
                }

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config
            )

            logger.info(f"成功创建集合: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"创建集合失败: {collection_name}: {e}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        try:
            return self.client.collection_exists(collection_name)
        except Exception as e:
            logger.error(f"Failed to check collection existence {collection_name}: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        try:
            logger.info(f"Qdrant 删除集合: {collection_name}")
            self.client.delete_collection(collection_name)
            logger.info(f"成功删除集合: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {collection_name}: {e}")
            return False

    def add_vectors(self,
                    collection_name: str,
                    vectors: List[List[float]],
                    payloads: List[Dict],
                    sparse_vectors: Optional[List[Dict]] = None,
                    ids: Optional[List[str]] = None) -> bool:
        """添加向量到集合

        Args:
            collection_name: 集合名称
            vectors: 向量列表
            payloads: 载荷数据列表
            sparse_vectors: 稀疏向量列表
            ids: 向量ID列表，如果为None则自动生成

        Returns:
            bool: 是否添加成功
        """
        try:
            if not vectors or not payloads:
                logger.warning("未提供向量或载荷数据")
                return False

            if len(vectors) != len(payloads):
                logger.error("向量和载荷数据长度不匹配")
                return False

            # 生成 ID
            if ids is None:
                ids = [uuid.uuid4().hex for _ in range(len(vectors))]

            # 构建点数据
            points = []
            if sparse_vectors:
                for i, (vector, sparse_vector, payload) in enumerate(zip(vectors, sparse_vectors, payloads)):
                    point = models.PointStruct(
                        id=ids[i],
                        vector={"vector": vector, "sparse_vector": sparse_vector},
                        payload=payload
                    )
                    points.append(point)
            else:
                for i, (vector, payload) in enumerate(zip(vectors, payloads)):
                    point = models.PointStruct(
                        id=ids[i],
                        vector={"vector": vector},
                        payload=payload
                    )
                    points.append(point)

            # 批量插入
            batch_size = 1000
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch_points,
                    wait=True
                )

            logger.info(f"成功添加 {len(vectors)} 个向量到集合: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"添加向量失败: {collection_name}: {e}")
            return False

    def search_vectors(self,
                       collection_name: str,
                       query_vectors: List[List[float]],
                       limit: int = 10,
                       score_threshold: float = 0.0,
                       filter_conditions: Optional[Dict] = None) -> List[List[Dict]]:
        """搜索相似向量

        Args:
            collection_name: 集合名称
            query_vectors: 查询向量
            limit: 返回结果数量限制
            score_threshold: 相似度阈值
            filter_conditions: 过滤条件
        """
        logger.info(f"搜索集合: {collection_name}")
        logger.info(f"过滤条件: {filter_conditions}")
        logger.info(f"相似度阈值: {score_threshold}")
        logger.info(f"返回结果数量限制: {limit}")
        try:
            if not query_vectors:
                logger.info("未提供查询向量")
                return []

            # 构建过滤器
            query_filter = self._build_filter(filter_conditions)

            # 执行搜索
            batch_requests = []
            for query_vector in query_vectors:
                batch_requests.append(
                    models.QueryRequest(
                        query=query_vector,
                        limit=limit,
                        filter=query_filter,
                        using="vector",
                        score_threshold=score_threshold,
                        with_payload=True,
                        with_vector=False
                    )
                )
            search_results = self.client.query_batch_points(
                collection_name=collection_name,
                requests=batch_requests,
            )
            logger.info(f"搜索集合: {collection_name}")
            logger.info(f"搜索结果: {search_results}")

            # 格式化结果
            results = []
            for i, search_result in enumerate(search_results):
                result = []

                for point in search_result.points:
                    result.append({
                        'id': point.id,
                        'score': point.score,
                        'payload': point.payload
                    })
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"搜索向量失败: {collection_name}: {e}")
            return []

    def keyword_search(self,
                       collection_name: str,
                       queries: List[str],
                       sparse_vectors: Optional[List[Dict]] = None,
                       limit: int = 10,
                       score_threshold: float = 0.0,
                       filter_conditions: Optional[Dict] = None) -> List[List[Dict]]:
        """关键词搜索

        Args:
            collection_name: 集合名称
            queries: 查询关键词
            sparse_vectors: 稀疏向量查询
            limit: 返回结果数量限制
            score_threshold: 相似度阈值
            filter_conditions: 过滤条件

        Returns:
            List[List[Dict]]: 搜索结果列表
        """
        try:
            if not queries:
                logger.info("未提供查询关键词")
                return []

            # 构建过滤器
            query_filter = self._build_filter(filter_conditions)

            logger.info(f"Qdrant 关键词搜索: {sparse_vectors}")

            query_requests = []
            for sparse_vector in sparse_vectors:
                query_requests.append(
                    models.QueryRequest(
                        query=SparseVector(**sparse_vector),
                        limit=limit,
                        filter=query_filter,
                        using="sparse_vector",
                        score_threshold=score_threshold,
                        with_payload=True,
                        with_vector=False
                    )
                )

            search_results = self.client.query_batch_points(
                collection_name=collection_name,
                requests=query_requests,
            )
            results = []
            for search_result in search_results:
                result = []
                for point in search_result.points:
                    result.append({
                        'id': point.id,
                        'score': point.score,
                        'payload': point.payload
                    })
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"关键词搜索失败: {collection_name}: {e}")
            return []

    def delete_vectors(self,
                       collection_name: str,
                       filter_conditions: Dict) -> bool:
        """根据条件删除向量

        Args:
            collection_name: 集合名称
            filter_conditions: 删除条件

        Returns:
            bool: 是否删除成功
        """
        logger.info(f"删除集合: {collection_name}, 删除条件: {filter_conditions}")
        try:
            query_filter = self._build_filter(filter_conditions)
            if query_filter is None:
                logger.error("无效的删除条件")
                return False

            self.client.delete(
                collection_name=collection_name,
                points_selector=query_filter,
                wait=True
            )

            logger.info(f"成功删除向量: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"删除向量失败: {collection_name}: {e}")
            return False

    def count_vectors(self,
                      collection_name: str,
                      filter_conditions: Optional[Dict] = None) -> int:
        """统计向量数量

        Args:
            collection_name: 集合名称
            filter_conditions: 过滤条件

        Returns:
            int: 向量数量
        """
        try:
            query_filter = self._build_filter(filter_conditions)

            count_result = self.client.count(
                collection_name=collection_name,
                count_filter=query_filter,
                exact=True
            )

            return count_result.count

        except Exception as e:
            logger.error(f"统计向量数量失败: {collection_name}: {e}")
            return 0

    def get_collection_info(self, collection_name: str) -> Dict:
        """获取集合信息

        Args:
            collection_name: 集合名称

        Returns:
            Dict: 集合信息
        """
        try:
            collection_info = self.client.get_collection(collection_name)
            return {
                'name': collection_name,
                'points_count': collection_info.points_count,
                'vectors_count': collection_info.indexed_vectors_count,
                'status': collection_info.status,
                'config': collection_info.config
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {collection_name}: {e}")
            return {'error': str(e)}

    def scroll_vectors(self,
                       collection_name: str,
                       limit: int = 100,
                       offset: Optional[str] = None,
                       filter_conditions: Optional[Dict] = None) -> Tuple[List[Dict], Optional[str]]:
        """滚动获取向量数据

        Args:
            collection_name: 集合名称
            limit: 返回结果数量限制
            offset: 偏移量
            filter_conditions: 过滤条件

        Returns:
            Tuple[List[Dict], Optional[str]]: 向量数据列表和下一个偏移量
        """
        try:
            query_filter = self._build_filter(filter_conditions)

            scroll_result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            points, next_offset = scroll_result

            # 格式化结果
            formatted_points = []
            for point in points:
                formatted_points.append({
                    'id': point.id,
                    'payload': point.payload
                })

            return formatted_points, next_offset

        except Exception as e:
            logger.error(f"滚动获取向量数据失败: {collection_name}: {e}")
            return [], None

    def create_index(self,
                     collection_name: str,
                     field_name: str,
                     field_type: str) -> bool:
        """创建索引

        Args:
            collection_name: 集合名称
            field_name: 字段名称
            field_type: 字段类型

        Returns:
            bool: 是否创建成功
        """
        try:
            # 映射字段类型
            type_mapping = {
                'integer': models.PayloadSchemaType.INTEGER,
                'keyword': models.PayloadSchemaType.KEYWORD,
                'text': models.PayloadSchemaType.TEXT,
                'float': models.PayloadSchemaType.FLOAT,
                'bool': models.PayloadSchemaType.BOOL,
            }

            qdrant_type = type_mapping.get(field_type.lower(), models.PayloadSchemaType.KEYWORD)

            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=qdrant_type
            )

            logger.info(f"成功创建索引: {field_name} 在 {collection_name}")
            return True

        except Exception as e:
            logger.error(f"创建索引失败: {field_name} 在 {collection_name}: {e}")
            return False

    def _build_filter(self, filter_conditions: Optional[Dict]) -> Optional[models.Filter]:
        """构建 Qdrant 过滤器

        Args:
            filter_conditions: 过滤条件

        Returns:
            Optional[models.Filter]: 过滤器
        """
        if not filter_conditions:
            return None

        must_conditions = []
        for key, value in filter_conditions.items():
            # Warning: int 和 str 的匹配方式不同
            if isinstance(value, int):
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            elif isinstance(value, str):
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchText(text=value)
                    )
                )
            elif isinstance(value, list):
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    )
                )

        if must_conditions:
            return models.Filter(must=must_conditions)

        return None


class QdrantTextVectorDB(BaseCollectionVectorDB):
    """
    Qdrant 文本向量存储包装器

    这是一个轻量级的包装器，所有实际操作都委托给 QdrantVectorDB。
    只负责指定集合名称和文本特定的配置。

    Args:
        vector_db: Qdrant 向量数据库实例
        collection_name: 集合名称
        vector_size: 向量维度

    """

    def __init__(self, vector_db: QdrantVectorDB, collection_name: str, vector_size: int = 768):
        """
        初始化 Qdrant 文本向量存储

        Args:
            vector_db: Qdrant 向量数据库实例
            collection_name: 集合名称
            vector_size: 向量维度
        """
        super().__init__(
            vector_db=vector_db,
            collection_name=collection_name,
            embedding_size=vector_size,
            index_fields=["kb_id", "doc_id", "chunk_id"],
        )
        self.vector_size = vector_size

    def create_collection(self) -> bool:
        """创建文本向量集合"""
        success = self.vector_db.create_collection(
            collection_name=self.collection_name,
            vector_size=self.vector_size,
            enable_sparse=True
        )

        if success:
            # 创建索引
            for field in self.index_fields:
                self.vector_db.create_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_type="integer"
                )

        return success

    def drop_collection(self) -> bool:
        """删除集合"""
        return self.vector_db.delete_collection(self.collection_name)

    def add_docs(self, text_chunks: List[Dict]) -> bool:
        """
        添加文档到向量数据库

        Args:
            text_chunks: 文档列表

        Returns:
            bool: 是否添加成功
        """
        try:
            text_chunk_models = [TextChunkModel.model_validate(chunk) for chunk in text_chunks]

            vectors = []
            sparse_vectors = []
            payloads = []

            for chunk_model in text_chunk_models:
                vectors.append(chunk_model.vector)
                sparse_vectors.append(chunk_model.sparse_vector)
                payloads.append(chunk_model.model_dump(exclude={'vector', 'sparse_vector'}))
            return self.vector_db.add_vectors(
                collection_name=self.collection_name,
                vectors=vectors,
                sparse_vectors=sparse_vectors,
                payloads=payloads
            )

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            logger.error(f"添加文本文档失败: {self.collection_name}: {e}")
            return False

    def search_vector(self,
                      query_vectors: List[List[float]],
                      limit: int = 10,
                      score_threshold: float = 0.0,
                      filter_conditions: Optional[Dict] = None) -> List[List[Dict]]:
        """
        搜索相似向量

        Args:
            query_vectors: 查询向量
            limit: 返回结果数量限制
            score_threshold: 相似度阈值
            filter_conditions: 过滤条件

        Returns:
            List[List[Dict]]: 搜索结果列表
        """
        if not query_vectors:
            logger.info(f"未提供查询向量: {self.collection_name}, 返回空列表")
            return []

        return self.vector_db.search_vectors(
            collection_name=self.collection_name,
            query_vectors=query_vectors,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions
        )

    def keyword_search(self,
                       queries: List[str],
                       sparse_vectors: Optional[List[Dict]] = None,
                       limit: int = 10,
                       score_threshold: float = 0.0,
                       filter_conditions: Optional[Dict] = None) -> List[List[Dict]]:
        """关键词搜索

        Args:
            queries: 查询关键词
            sparse_vectors: 稀疏向量查询
            limit: 返回结果数量限制
            score_threshold: 相似度阈值
            filter_conditions: 过滤条件
        
        Returns:
            List[List[Dict]]: 搜索结果列表
        """
        return self.vector_db.keyword_search(
            collection_name=self.collection_name,
            queries=queries,
            sparse_vectors=sparse_vectors,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions
        )

    def delete_by_kb_id(self, kb_id: str) -> bool:
        """根据知识库ID删除向量"""
        return self.vector_db.delete_vectors(
            collection_name=self.collection_name,
            filter_conditions={"kb_id": kb_id}
        )

    def delete_by_doc_ids(self, doc_ids: List[int]) -> bool:
        """根据文档ID列表删除向量"""
        return self.vector_db.delete_vectors(
            collection_name=self.collection_name,
            filter_conditions={"doc_id": doc_ids}
        )

    def delete_by_chunk_ids(self, chunk_ids: List[int]) -> bool:
        """根据文档块ID列表删除向量"""
        return self.vector_db.delete_vectors(
            collection_name=self.collection_name,
            filter_conditions={"chunk_id": chunk_ids}
        )

    def delete_by_key(self, key: str, values: List[str]) -> bool:
        """
        根据键值删除向量

        Args:
            key: 字段名称
            values: 字段值列表

        Returns:
            bool: 是否删除成功
        """
        return self.vector_db.delete_vectors(
            collection_name=self.collection_name,
            filter_conditions={key: values}
        )

    def delete_by_file_ids(self, kb_id: str, file_ids: List[str]) -> bool:
        return self.vector_db.delete_vectors(
            collection_name=self.collection_name,
            filter_conditions={"kb_id": kb_id, "file_id": file_ids}
        )


class QdrantGenericVectorDB(BaseCollectionVectorDB):
    """
    Qdrant 通用向量存储包装器

    这是一个轻量级的包装器，所有实际操作都委托给 QdrantVectorDB。
    用于存储和检索所有类型的向量数据，包括：
    - 文本向量
    - 图像向量
    - 页面向量
    - 其他多模态向量

    该类提供了统一的接口，避免为每种向量类型创建重复的实现。
    """

    def create_collection(self) -> bool:
        """创建向量集合"""
        success = self.vector_db.create_collection(
            collection_name=self.collection_name,
            vector_size=self.embedding_size
        )

        if success:
            # 创建索引
            for field in self.index_fields:
                self.vector_db.create_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_type="keyword"
                )

        return success

    def drop_collection(self) -> bool:
        return self.vector_db.delete_collection(self.collection_name)

    def add_images(self, image_chunks: List[Dict]) -> bool:
        """
        添加向量数据到数据库

        注意：虽然方法名为 add_images（为了兼容基类接口），
        但实际上可以添加任何类型的向量数据（图像、页面等）

        Args:
            image_chunks: 向量数据块列表

        Returns:
            bool: 是否添加成功
        """
        try:
            image_chunk_models = [ImageChunkModel.model_validate(chunk) for chunk in image_chunks]

            vectors = []
            payloads = []

            for chunk_model in image_chunk_models:
                vectors.append(chunk_model.vector)
                payloads.append(chunk_model.model_dump(exclude={'vector'}))

            return self.vector_db.add_vectors(
                collection_name=self.collection_name,
                vectors=vectors,
                payloads=payloads
            )

        except Exception as e:
            logger.error(f"添加向量数据失败: {self.collection_name}: {e}")
            return False

    def search_vector(self,
                      query_vectors: List[list[float]],
                      limit: int = 10,
                      score_threshold: float = 0.0,
                      filter_conditions: Optional[Dict] = None) -> List[List[Dict]]:
        """搜索相似向量"""
        if not query_vectors:
            logger.info(f"未提供查询向量: {self.collection_name}, 返回空列表")
            return []

        if len(query_vectors[0]) != self.embedding_size:
            raise ValueError(
                f"Query vector size {len(query_vectors[0])} does not match "
                f"embedding size {self.embedding_size} for collection {self.collection_name}"
            )

        return self.vector_db.search_vectors(
            collection_name=self.collection_name,
            query_vectors=query_vectors,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions
        )

    def delete_by_kb_id(self, kb_id: str) -> bool:
        """根据知识库ID删除向量"""
        return self.vector_db.delete_vectors(
            collection_name=self.collection_name,
            filter_conditions={"kb_id": kb_id}
        )

    def delete_by_doc_ids(self, doc_ids: List[int]) -> bool:
        """根据文档ID列表删除向量"""
        return self.vector_db.delete_vectors(
            collection_name=self.collection_name,
            filter_conditions={"doc_id": doc_ids}
        )

    def delete_by_key(self, key: str, values: List[str]) -> bool:
        """根据键值删除向量"""
        return self.vector_db.delete_vectors(
            collection_name=self.collection_name,
            filter_conditions={key: values}
        )


QdrantImageVectorDB = QdrantGenericVectorDB
QdrantPageVectorDB = QdrantGenericVectorDB
