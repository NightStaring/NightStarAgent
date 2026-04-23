from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import yaml
import dotenv

ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}$")


@dataclass
class VectorDBConfig:
    """向量数据库相关配置。"""

    # 向量数据库
    db_type: Optional[str] = None
    text_collection: Optional[str] = None
    image_collection: Optional[str] = None
    page_collection: Optional[str] = None
    text_dimension: int = 1024
    image_dimension: int = 4096
    page_dimension: int = 4096

    # Qdrant 配置
    url: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None


@dataclass
class Config:
    """项目配置聚合类。"""

    vector_db: VectorDBConfig

    @staticmethod
    def _resolve_env_placeholders(value: Any) -> Any:
        """
        递归解析 YAML 中的环境变量占位符。

        支持两种写法：
        - ${VAR}
        - ${VAR:default_value}
        """
        if isinstance(value, dict):
            return {k: Config._resolve_env_placeholders(v) for k, v in value.items()}
        if isinstance(value, list):
            return [Config._resolve_env_placeholders(v) for v in value]
        if not isinstance(value, str):
            return value

        match = ENV_PATTERN.match(value.strip())
        if not match:
            return value

        env_key, default_value = match.groups()
        env_value = os.getenv(env_key, default_value)
        return env_value

    @classmethod
    def from_yaml(cls, file_path: str | Path | None = None) -> "Config":
        """
        从 YAML 文件加载配置。

        默认读取当前目录下的 `config.yaml`。
        """
        if file_path is None:
            file_path = Path(__file__).resolve().parent / "config.yaml"
        else:
            file_path = Path(file_path).resolve()

        # 自动加载项目根目录的 .env 文件（如果存在）
        project_root = Path(__file__).resolve().parents[2]
        dotenv.load_dotenv(project_root / ".env")

        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw = cls._resolve_env_placeholders(raw)

        # 向量数据库 配置
        vector_db_raw: dict[str, Any] = raw.get("vector_db", {}) or {}
        qdrant_raw: dict[str, Any] = vector_db_raw.get("Qdrant", {}) or {}
        vector_db = VectorDBConfig(
            db_type=vector_db_raw.get("db_type"),
            text_collection=vector_db_raw.get("text_collection"),
            image_collection=vector_db_raw.get("image_collection"),
            page_collection=vector_db_raw.get("page_collection"),
            text_dimension=int(vector_db_raw.get("text_dimension", 1024)),
            image_dimension=int(vector_db_raw.get("image_dimension", 4096)),
            page_dimension=int(vector_db_raw.get("page_dimension", 4096)),

            # Qdrant 配置
            url=qdrant_raw.get("url"),
            port=qdrant_raw.get("port"),
            api_key=qdrant_raw.get("api_key"),
        )
        return cls(vector_db=vector_db)


_CONFIG: Config | None = None


def get_config(file_path: str | Path | None = None, force_reload: bool = False) -> Config:
    """读取并缓存配置，避免重复加载文件。"""
    global _CONFIG
    if force_reload or _CONFIG is None:
        _CONFIG = Config.from_yaml(file_path=file_path)
    return _CONFIG
