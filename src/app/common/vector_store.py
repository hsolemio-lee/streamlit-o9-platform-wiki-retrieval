from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Qdrant, Chroma
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")


def get_qdrant_db(collection: str) -> Qdrant:
    print('qdrant url', qdrant_url)

    qdrant_client = QdrantClient(url=qdrant_url)
    return Qdrant(
        client=qdrant_client, collection_name=collection, embeddings=OpenAIEmbeddings()
    )


def get_chroma_db(collection: str) -> Chroma:
    return Chroma(
        persist_directory="data",
        embedding_function=OpenAIEmbeddings(),
        collection_name=collection,
    )
