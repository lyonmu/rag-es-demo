"""Request body models for API endpoints."""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="搜索查询文本")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, description="用户提问")
    top_k: int = Field(default=5, ge=1, le=20, description="检索参考文档数")


class CreateKbRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="知识库名称")
    description: str = Field(default="", max_length=512, description="知识库描述")
