from typing import Optional, List

from pydantic import BaseModel, Field


class ImageChunkModel(BaseModel):
    """图片切块模型
        Args:
        kb_id: 知识库ID
        page_id: 来源文档的ID
        doc_id: 所属页的chunk_id
        chunk_id: 文档的块ID
        page_chunk_id: page_chunk_id
        page_no: 文档的页码
        tags: 文档的标签
        caption: 图片的标题
        vector: 图片的向量表示
        doc: 文档的内容
        image_url: 图片的URL
        md5sum: 图片的MD5值
        image_type: 图片的类型
        Returns:
        ImageChunkModel: 图片切块模型
    """

    
    kb_id: str = Field(..., description="知识库的ID")
    page_id: Optional[str] = Field(None, description="来源文档的ID")
    doc_id: Optional[str] = Field(None, description="所属页的chunk_id")
    chunk_id: Optional[str] = Field(None, description="文档的块ID")
    page_chunk_id: Optional[str] = Field(None, description="page_chunk_id")
    page_no: Optional[str] = Field(None, description="文档的页码")
    tags: Optional[List[str]] = Field(default_factory=list, description="文档的标签")
    caption: Optional[str] = Field(None, description="图片的标题")
    vector: Optional[List[float]] = Field(None, description="图片的向量表示")
    doc: Optional[str] = Field(None, description="文档的内容")
    image_url: Optional[str] = Field(None, description="图片的URL")
    md5sum: Optional[str] = Field(None, description="图片的MD5值")
    image_type: Optional[str] = Field(None, description="图片的类型")

    class Config:
        extra = "allow"
