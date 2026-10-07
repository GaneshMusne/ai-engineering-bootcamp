from qdrant_client import QdrantClient

from api.api.models import RAGRequest, RAGResponse
from fastapi import Request, APIRouter
from api.agents.retrieval_generation import rag_pipeline

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()

qdrant_client = QdrantClient(url="http://qdrant:6333")

@rag_router.post("/")
def rag(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:

    response = rag_pipeline(payload.query, qdrant_client)

    return RAGResponse(
        request_id=request.state.request_id,
        message=response
    )

api_router = APIRouter();
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
