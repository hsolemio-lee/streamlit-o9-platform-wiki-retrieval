import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from router.embedding_router import embedding_router
from router.retrieval_qa_router import retrieval_qa_router

load_dotenv()
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

def create_app():
    app = FastAPI()
    app.include_router(embedding_router, prefix="/api/v1/embedding", tags=["v1"])
    app.include_router(retrieval_qa_router, prefix="/api/v1/qa", tags=["v1"])
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True, port=9999)
