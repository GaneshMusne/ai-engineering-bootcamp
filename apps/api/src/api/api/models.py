from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in RAG pip line")

class RAGResponse(BaseModel):
    request_id : str = Field(..., description="The request id")
    message: str = Field(..., description="Answer to the query")