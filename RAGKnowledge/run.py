"""
运行RAGKnowledge
files_path: 需要处理的文件路径
"""


import os
from tqdm import tqdm
import uuid

from .file_processer import DocumentProcessor
from .utils.logger_utils import logger


def main():
    # files_path = "RAGKnowledge/dataset/Chart-MRAG/files"
    # files_path = os.path.join(os.path.dirname(__file__), files_path)
    # files = os.listdir(files_path)
    # for file in tqdm(files):
    #     file_path = os.path.join(files_path, file)
    #     try:
    #         file_processor = DocumentProcessor(
    #             kb_id="chart-mrag",
    #             file_id=uuid.uuid4().hex,
    #             work_dir=os.path.join(os.path.dirname(__file__), "dataset", "Chart-MRAG", "knowledge"),
    #             file_path=file_path
    #         )
    #         file_processor.process()
    #     except Exception as e:
    #         logger.error(f"Error processing file {file}: {e}")
    from .embedding.text_bm25 import get_bm25_embedding_model
    text_list = [
        "work_dir=os.path.join(os.path.dirname(__file__)",
        "file_processor.process()"
    ]
    print(get_bm25_embedding_model().encode_text_batch(text_list))

if __name__ == "__main__":
    main()