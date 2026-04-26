import os
from typing import List

from ..utils.logger_utils import logger
from .embedding import TextEmbedding
from ..utils.models_client import get_client


class OpenAITextEmbedding(TextEmbedding):
    def __init__(self):
        super().__init__()
        self.client = get_client()
        self.model_name = "BAAI/bge-m3"

    def _encode_text_batch(self, texts: List[str]) -> list[list[float]]:
        """
        批量编码文本为向量

        Args:
            texts: 文本列表

        Returns:
            向量数组，形状为 (len(texts), embedding_dim)
        """
        if not texts:
            return []

        max_text_length = int(os.getenv("TEXT_EMBEDDING_MAX_TEXT_LENGTH", 8000))
        texts = [text[:max_text_length] for text in texts]

        try:
            batch_size = 10
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                logger.info(f"正在编码文本: {batch_texts}")
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts,
                    timeout=int(os.getenv("API_TIMEOUT", 300)),
                )
                logger.info(f"文本编码完成.")
                response_data = response.data
                if not response_data:
                    raise ValueError("文本编码失败: 未返回任何数据")

                for embedding in response_data:
                    embeddings.append(embedding.embedding)
            return embeddings

        except Exception as e:
            import traceback
            logger.error(f"文本编码失败: {e}\n{traceback.format_exc()}")
            raise Exception(f"文本编码失败: {e}") from e


def get_text_embedding_model() -> TextEmbedding:
    """获取文本embedding模型"""
    return OpenAITextEmbedding()
