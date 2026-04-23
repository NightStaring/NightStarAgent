"""
获取图片OCR的文字结果
"""
from abc import abstractmethod, ABC

from .logger_utils import logger
from .vlm import get_vlm


class OCRBase(ABC):

    @abstractmethod
    def ocr(self, image: str) -> str:
        pass


class OCR(OCRBase):

    def __init__(self):
        self.model = "deepseek-ai/DeepSeek-OCR"

    def ocr(self, image_path: str) -> str:
        try:
            logger.info("正在调用OCR...")
            result = get_vlm().vlm(self.model, "只需提取图片中的文字，不要包含任何其他内容", image_path)
            logger.info("OCR调用完成")
            return result
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""


def get_ocr_model() -> OCRBase:
    return OCR()
