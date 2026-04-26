from .models_client import get_client
from .logger_utils import logger


class LLM:

    def __init__(self):
        self.client = get_client()

    def llm(self, model, prompt) -> str:
        try:
            logger.info("正在调用LLM...")
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            logger.info("LLM调用完成")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""

    def llm_small(self, prompt) -> str:
        try:
            logger.info("正在调用LLM...")
            response = self.client.chat.completions.create(
                model="Qwen/Qwen3.6-35B-A3B",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.01,
            )
            logger.info("LLM调用完成")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""

    def llm_big(self, prompt) -> str:
        try:
            logger.info("正在调用LLM...")
            response = self.client.chat.completions.create(
                model="Qwen/Qwen3.5-397B-A17B",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.01,
            )
            logger.info("LLM调用完成")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""

    # def llm_json(self, prompt) -> str:
    #     try:
    #         logger.info("正在调用LLM...")
    #         response = self.client.chat.completions.create(
    #             model="Qwen/Qwen3.6-7B-A3B",
    #             messages=[{"role": "user", "content": prompt}],
    #         )
    #         logger.info("LLM调用完成")
    #         return response.choices[0].message.content
    #     except Exception as e:
    #         logger.error(f"LLM error: {e}")
    #         return ""

def get_llm() -> LLM:
    return LLM()
