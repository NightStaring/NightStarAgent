import os

from .models_client import get_client
from .image_utils import image_path_to_base64
from .logger_utils import logger


class VLM:

    def __init__(self):
        self.client = get_client()

    def vlm(self, model, prompt, image_path: str) -> str:
        try:
            mime = self._mime_from_path(image_path)
            base64_image = image_path_to_base64(image_path)

            logger.info("正在调用VLM...")
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": [
                               {"type": "text", "text": prompt},
                               {"type": "image_url", "image_url": f"data:{mime};base64,{base64_image}"}]
                           }],
            )
            logger.info("VLM调用完成")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Image2Caption error: {e}")
            return ""

    def _mime_from_path(self, image_path: str) -> str:
        ext = os.path.splitext(image_path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            return "image/jpeg"
        if ext == ".webp":
            return "image/webp"
        return "image/png"


def get_vlm() -> VLM:
    return VLM()
