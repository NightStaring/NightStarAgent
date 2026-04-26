"""
Rerank API 工具模块。

提供对 SiliconFlow ``/v1/rerank`` 接口的轻量封装，
用于在检索阶段之后对候选文档进行相关性重排序。
"""

import os
from typing import Any
import requests


from .logger_utils import logger


DEFAULT_RERANK_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class Rerank:
    """Rerank 客户端封装。

    该类负责：
    - 读取 API Key 与默认模型配置；
    - 组装重排序请求参数；
    - 发送 HTTP 请求并返回原始 JSON 响应。
    """

    def __init__(self):
        self.url = "https://api.siliconflow.cn/v1"
        self.api_key = os.getenv("SILICONFLOW_API")
        self.model = "Qwen/Qwen3-Reranker-8B"
        self.timeout = 60

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        return_documents: bool = False,
    ) -> dict[str, Any]:
        """调用重排序接口并返回原始响应。

        Args:
            query: 用户查询文本。
            documents: 待重排文档列表。
            top_n: 返回前 N 条。为空时由服务端按默认策略处理。
            return_documents: 是否在响应中返回文档内容。

        Returns:
            接口返回的原始 JSON 字典，通常包含 ``results`` 字段。
        """
        # 按 SiliconFlow /v1/rerank 协议组织请求体。
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": return_documents,
            "top_n": top_n,
        }
        # 使用 Bearer Token 进行鉴权。
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(f"调用Rerank API: model={self.model}, top_n={top_n}, docs={len(documents)}")
        response = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        return response.json()

def get_rerank() -> Rerank:
    """获取 Rerank 客户端实例。"""
    return Rerank()
