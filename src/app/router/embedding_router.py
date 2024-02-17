from fastapi import APIRouter
from service.embedding_service import EmbeddingService
from dto.embedding_info_dto import EmbeddingInfo


embedding_router = APIRouter()
embedding_service = EmbeddingService()


@embedding_router.post("/collection")
def embedding_o9platform_wiki(embeddingInfo: EmbeddingInfo):
    embedding_service.embedding_o9platform_wiki_qdrant(embeddingInfo)
    return {"result": "success"}


@embedding_router.post("/collection/dir")
def embedding_o9platform_wiki_by_dir(embeddingInfo: EmbeddingInfo):
    embedding_service.embedding_o9_platform_wiki_chroma(embeddingInfo)
    return {"result": "success"}
