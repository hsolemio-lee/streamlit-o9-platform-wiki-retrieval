from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import BSHTMLLoader, TextLoader
from langchain_community.vectorstores import Qdrant, Chroma
from langchain_openai import OpenAIEmbeddings
from dto.embedding_info_dto import EmbeddingInfo

load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")


class EmbeddingService:

    def embedding_o9platform_wiki_qdrant(self, embeddingInfo: EmbeddingInfo):

        loader = BSHTMLLoader(embeddingInfo.file_url)
        # loader = TextLoader(embeddingInfo.file_url)
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

        docs = text_splitter.split_documents(document)

        Qdrant.from_documents(
            documents=docs,
            embedding=OpenAIEmbeddings(),
            url=qdrant_url,
            collection_name=embeddingInfo.collection,
        )

    def embedding_o9platform_wiki_qdrant_by_dir(self, embeddingInfo: EmbeddingInfo):
        file_names = os.listdir(embeddingInfo.dir_url)
        for file_name in file_names:
            loader = BSHTMLLoader(embeddingInfo.dir_url + "/" + file_name)
            document = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=10,
            )

            docs = text_splitter.split_documents(document)

            Qdrant.from_documents(
                documents=docs,
                embedding=OpenAIEmbeddings(),
                url=qdrant_url,
                collection_name=embeddingInfo.collection,
            )

    def embedding_o9_platform_wiki_chroma(self, embeddingInfo: EmbeddingInfo):
        file_names = os.listdir(embeddingInfo.dir_url)
        for file_name in file_names:
            loader = BSHTMLLoader(embeddingInfo.dir_url + "/" + file_name)
            document = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=10,
            )

            docs = text_splitter.split_documents(document)

            chroma_db = Chroma.from_documents(
                documents=docs,
                embedding=OpenAIEmbeddings(),
                persist_directory="data",
                collection_name=embeddingInfo.collection,
            )

        return {"result": "success"}
