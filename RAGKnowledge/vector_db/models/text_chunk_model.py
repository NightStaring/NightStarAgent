from typing import Optional, List
from pydantic import BaseModel, Field


class TextChunkModel(BaseModel):
    """文本切块模型

    Args:
        kb_id: 知识库ID
        text: 原始文本
        vector: 文本向量
        sparse_vector: 文本稀疏向量
        ref_id: 引用ID
        file_sorted: 文件内排序
        chunk_type: 切块类型
        file_path: 文件路径
        filename: 文件名
        created: 创建时间
        split_type: 切分类型
    Returns:
        TextChunkModel: 文本切块模型
    """

    kb_id: str

    text: Optional[str] = Field(None, description="原始文本")
    vector: Optional[List[float]] = Field(None, description="文本向量")
    sparse_vector: Optional[dict] = Field(None, description="文本稀疏向量")
    ref_id: Optional[str] = Field(None, description="引用ID")
    file_sorted: Optional[str] = Field(None, description="文件内排序")
    chunk_type: Optional[str] = Field(None, description="切块类型")
    file_path: Optional[str] = Field(None, description="文件路径")
    filename: Optional[str] = Field(None, description="文件名")
    created: Optional[float] = Field(None, description="创建时间")
    split_type: Optional[str] = Field(None, description="切分类型")

    class Config:
        extra = "allow"
