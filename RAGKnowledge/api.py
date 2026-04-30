"""
FastAPI 包装层，用于暴露 Agentic RAG 的查询功能

提供 RESTful API 接口，支持前端调用：
- POST /api/query: 处理用户查询，返回回答
"""
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent import AgenticRAG
from .utils.logger_utils import logger


# 定义请求和响应模型
class QueryRequest(BaseModel):
    question: str
    kb_id: str = "chart-mrag"
    image_path: Optional[str] = None


class QueryResponse(BaseModel):
    success: bool
    answer: str
    error: Optional[str] = None


# 全局 AgenticRAG 实例
rag_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理
    启动时初始化 AgenticRAG 实例
    """
    global rag_instance
    try:
        logger.info("正在初始化 Agentic RAG 实例...")
        # 默认使用 "chart-mrag" 知识库，也可以通过环境变量配置
        default_kb_id = os.getenv("DEFAULT_KB_ID", "chart-mrag")
        rag_instance = AgenticRAG(kb_id=default_kb_id, n_round=5)
        logger.info(f"Agentic RAG 实例初始化成功，默认知识库: {default_kb_id}")
    except Exception as e:
        logger.error(f"Agentic RAG 实例初始化失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
    yield
    # 关闭时的清理工作
    logger.info("正在关闭 API 服务...")


# 创建 FastAPI 应用
app = FastAPI(
    title="NightStarAgent RAG API",
    description="Agentic RAG 查询服务 API",
    version="1.0.0",
    lifespan=lifespan
)

# 配置 CORS 中间件，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境可以配置为具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)


@app.get("/")
async def root():
    """
    根路径，返回服务基本信息
    """
    return {
        "message": "NightStarAgent RAG API Service",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {
        "status": "healthy",
        "rag_initialized": rag_instance is not None
    }


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    处理用户查询的核心接口

    Args:
        request: 包含查询信息的请求体
            - question: 用户的问题
            - kb_id: 知识库ID，默认为 "chart-mrag"
            - image_path: 可选的图片路径

    Returns:
        QueryResponse: 包含回答的响应
    """
    global rag_instance

    if not rag_instance:
        raise HTTPException(
            status_code=500,
            detail="RAG 服务未初始化，请稍后重试"
        )

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="问题不能为空"
        )

    try:
        logger.info(f"接收到查询请求 - 知识库: {request.kb_id}, 问题: {request.question}")

        # 如果知识库不同，创建新的实例
        if request.kb_id != rag_instance._kb_id:
            logger.info(f"切换知识库: {rag_instance._kb_id} -> {request.kb_id}")
            rag_instance = AgenticRAG(kb_id=request.kb_id, n_round=5)

        # 执行查询
        answer = rag_instance.run(
            question=request.question,
            image_path=request.image_path
        )

        logger.info(f"查询处理完成")

        return QueryResponse(
            success=True,
            answer=answer
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理查询时发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"处理查询时发生错误: {str(e)}"
        )


@app.get("/api/query")
async def process_query_get(
    question: str = Query(..., description="用户的问题"),
    kb_id: str = Query("chart-mrag", description="知识库ID"),
    image_path: Optional[str] = Query(None, description="可选的图片路径")
):
    """
    GET 版本的查询接口，用于简单测试

    Args:
        question: 用户的问题
        kb_id: 知识库ID，默认为 "chart-mrag"
        image_path: 可选的图片路径

    Returns:
        包含回答的响应
    """
    global rag_instance

    if not rag_instance:
        raise HTTPException(
            status_code=500,
            detail="RAG 服务未初始化，请稍后重试"
        )

    if not question or not question.strip():
        raise HTTPException(
            status_code=400,
            detail="问题不能为空"
        )

    try:
        logger.info(f"接收到 GET 查询请求 - 知识库: {kb_id}, 问题: {question}")

        # 如果知识库不同，创建新的实例
        if kb_id != rag_instance._kb_id:
            logger.info(f"切换知识库: {rag_instance._kb_id} -> {kb_id}")
            rag_instance = AgenticRAG(kb_id=kb_id, n_round=5)

        # 执行查询
        answer = rag_instance.run(
            question=question,
            image_path=image_path
        )

        logger.info(f"查询处理完成")

        return {
            "success": True,
            "answer": answer
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理查询时发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"处理查询时发生错误: {str(e)}"
        )