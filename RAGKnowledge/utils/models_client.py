import os
from openai import OpenAI

from .logger_utils import logger


def get_client():
    logger.info("正在调用外部模型...")
    return OpenAI(
        base_url="https://api.siliconflow.cn/v1",
        api_key=os.getenv("SILICONFLOW_API"),
    )