import os
from http import HTTPStatus
from typing import List

import requests
from PIL import Image

from .embedding import ImageEmbedding
from ..utils import image_utils
from ..utils.logger_utils import logger


class QwenVLEmbedding(ImageEmbedding):
    def __init__(self):
        super().__init__()
        self.timeout = int(os.getenv("API_TIMEOUT", 300))
        self.api_key = os.getenv("SILICONFLOW_API")
        self.model_name = "Qwen/Qwen3-VL-Embedding-8B"
        self.url = "https://api.siliconflow.cn/v1/embeddings"

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """
        将PIL图像转换为base64编码字符串

        Args:
            image: PIL图像对象

        Returns:
            base64编码的图像字符串
        """
        return "data:image/png;base64," + image_utils.image_to_base64(image)

    def _encode_image(self, image: Image.Image) -> list[float]:
        
        client = OpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key=os.getenv("SILICONFLOW_API"),
        )
        response = client.embeddings.create(
            model=self.model_name,
            input=[{"image": self._image_to_base64(image)}],
            timeout=self.timeout
        )
        return response.data[0].embedding

    def _encode_image_batch(self, images: List[Image.Image]) -> list[list[float]]:
        """
        批量编码图片为向量

        Args:
            images: 图片列表

        Returns:
            向量数组，形状为 (len(images), embedding_dim)
        """
        if not images:
            return []

        embeddings = []
        for image in images:
            embedding = self._encode_image(image)
            embeddings.append(embedding)
        return embeddings


def get_image_embedding_model() -> ImageEmbedding:
    """获取图像embedding模型"""
    return QwenVLEmbedding()