"""
文本重排序模块。

该模块基于外部 Rerank API 对检索候选文本做二次排序，
输出与原候选列表一一对齐的相关性分数。
"""
import os
from typing import Dict, Any
import requests

from ..utils.rerank import get_rerank


class TextReranker:
    """文本重排序器。

    内部组合 ``utils.rerank`` 中的客户端，负责把 API 响应转换为
    与输入文本列表对齐的分数列表，便于后续统一排序。
    """

    def __init__(self):
        self.reranker = get_rerank()

    def rerank(self, question: str, texts: list[str]) -> list[float]:
        """对候选文本进行重排序评分。

        Args:
            question: 用户问题（重排查询）。
            texts: 待重排文本列表。

        Returns:
            与 ``texts`` 顺序对齐的相关性分数列表。
        """

        if not texts:
            return []

        response_data = self.reranker.rerank(question, texts)
        text_scores = response_data['results']

        # 先按输入长度初始化，确保返回分数与原文本索引严格对齐。
        score_list = [0.0] * len(texts)
        for i, text_score in enumerate(text_scores):
            # API 返回的是局部结果列表，通过 index 回填到原始位置。
            score_list[text_score.get("index")] = text_score.get('relevance_score')

        return score_list

def get_text_reranker() -> TextReranker:
    return TextReranker()
