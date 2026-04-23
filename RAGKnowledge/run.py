"""
运行RAGKnowledge
files_path: 需要处理的文件路径
"""


import os
from tqdm import tqdm
import uuid

from .file_processer import DocumentProcessor
from .config import get_config
from .utils.logger_utils import logger
from .vector_db.db import VectorDB


def show_db_info():
    db = VectorDB()
    print(db.base_store.client.get_collections())

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

def main():
    files_path = os.path.join(os.path.dirname(__file__), "dataset", "Chart-MRAG", "files")
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

if __name__ == "__main__":
    main()