from fastapi import APIRouter
from service.retrieval_service import RetrievalService
from dto.qa_dto import QA
from dto.chat_dto import ChatDTO

retrieval_qa_router = APIRouter()
retrieval_service = RetrievalService()


@retrieval_qa_router.post("/retrieval/qa")
def get_retrieval_qa(qa: QA):
    return retrieval_service.retrieval_qa(qa)

@retrieval_qa_router.post("/retrieval/conversation")
def get_retrieval_qa(chat_dto: ChatDTO):
    return retrieval_service.run_chat_retrieval_qa(chat_dto.query, chat_dto.collection, chat_dto.chat_history)
