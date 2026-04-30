"""
运行RAGKnowledge
files_path: 需要处理的文件路径

python -m RAGKnowledge.run

终端输出同时写入 .log（PowerShell，仍可在终端看到）:
python -m RAGKnowledge.run 2>&1 | Tee-Object -FilePath logs/run_0.log
"""


import os
from tqdm import tqdm
import uuid

from .file_processer import DocumentProcessor
from .config import get_config
from .utils.logger_utils import logger
from .vector_db.db import VectorDB
from .agent import AgenticRAG


def show_db_info():
    db = VectorDB()
    resp = db.base_store.client.get_collections()
    names = [c.name for c in resp.collections]
    print(f"Qdrant 集合（共 {len(names)} 个）: {names}")
    for name in names:
        info = db.base_store.get_collection_info(name)
        if info.get("error"):
            print(f"  - {name}: 读取失败 — {info['error']}")
        else:
            points = info.get("points_count")
            indexed = info.get("vectors_count")
            status = info.get("status")
            print(f"  - {name}: points_count={points}, indexed_vectors_count={indexed}, status={status}")

def create_db():
    db = VectorDB()
    db.create_all_collections()
    show_db_info()

def clear_db():
    show_db_info()
    db = VectorDB()
    cfg = get_config().vector_db
    collections = [cfg.text_collection, cfg.image_collection, cfg.page_collection]
    collections = list(dict.fromkeys(collections))

    for name in collections:
        db.base_store.delete_collection(name)
        logger.info(f"已删除集合: {name}")
    show_db_info()

"""
处理文档并写入向量库
"""
def process_documents():
    # files_path = os.path.join(os.path.dirname(__file__), "dataset", "Chart-MRAG", "files")
    files_path = os.path.join(os.path.dirname(__file__), "dataset", "test", "files")
    files = os.listdir(files_path)
    for file in tqdm(files):
        file_path = os.path.join(files_path, file)
        try:
            file_processor = DocumentProcessor(
                kb_id="chart-mrag",
                file_id=uuid.uuid4().hex,
                work_dir=os.path.join(os.path.dirname(__file__), "dataset", "Chart-MRAG", "knowledge", os.path.splitext(file)[0]),
                file_path=file_path
            )
            file_processor.process()
        except Exception as e:
            logger.error(f"处理文件 {file} 失败: {e}")

"""
agent
"""
def agent():
    question = "公司去年业绩如何？"
    agentic_rag = AgenticRAG(
        kb_id="chart-mrag",
        n_round=5
    )
    logger.info(agentic_rag.run(question))

if __name__ == "__main__":
    show_db_info()