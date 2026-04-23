import os

from .models_client import get_client
from .logger_utils import logger
from .vlm import get_vlm


class Image2Caption:

    def __init__(self):
        self.model = "Qwen/Qwen3.6-35B-A3B"

    def image2caption(self, image_path: str) -> str:
        try:
            logger.info("正在调用VLM描述图片...")
            result = get_vlm().vlm(self.model, "请根据图片内容生成一段描述性文字，不超过100个字", image_path)
            logger.info("VLM描述图片调用完成")
            return result
        except Exception as e:
            logger.error(f"VLM描述图片 error: {e}")
            return ""


def get_image2caption_model() -> Image2Caption:
    return Image2Caption()
