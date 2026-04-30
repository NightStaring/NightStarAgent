"""
启动 FastAPI 服务的脚本

使用方式:
python -m RAGKnowledge.start_api

或使用 uvicorn 直接运行:
uvicorn RAGKnowledge.api:app --host 0.0.0.0 --port 8000 --reload
"""
import uvicorn
import os


def main():
    """
    启动 FastAPI 服务
    """
    # 获取环境变量配置
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"

    print(f"=" * 50)
    print(f"NightStarAgent RAG API Service")
    print(f"=" * 50)
    print(f"服务地址: http://{host}:{port}")
    print(f"API 文档: http://{host}:{port}/docs")
    print(f"健康检查: http://{host}:{port}/health")
    print(f"=" * 50)

    # 启动 uvicorn 服务
    uvicorn.run(
        "RAGKnowledge.api:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    main()