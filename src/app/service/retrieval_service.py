from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from common.vector_store import get_qdrant_db, get_chroma_db
from dto.qa_dto import QA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# from dto.chat_dto import ChatDTO
from typing import Any, List, Dict
from agent.chains import get_translate_chain, get_text_language, source_url_retieval_chain
from parser.ouput_parser import translate_parser, language_parser, source_url_parser

load_dotenv()


class RetrievalService:
    
    def retrieval_qa(self, qa_dto: QA):
        chat = ChatOpenAI(model_name="gpt-3.5-turbo", verbose=True, temperature=0)
        
        vectore_store = get_qdrant_db(qa_dto.collection)
        qa = RetrievalQA.from_chain_type(
            llm=chat,
            retriever=vectore_store.as_retriever(),
            chain_type="stuff",
            verbose=True,
            return_source_documents=True,
        )
        query = f"""I want to know about o9 platform. {qa_dto.query}"""
        result = qa({"query": query})
        return result

    def run_chat_retrieval_qa(
        self, query: str, collection: str, chat_history: List[Dict[str, Any]] = []
    ):
        query_language = language_parser.parse(get_text_language().run(text=query))

        chat = ChatOpenAI(model_name="gpt-3.5-turbo", verbose=True, temperature=0)

        vectore_store = get_qdrant_db(collection)
        # vectore_store = get_chroma_db(collection)

        translated_query = query
        if query_language != "English":
            translated_query = translate_parser.parse(
                get_translate_chain().run(
                    text=query, input_language=query_language, output_language="English"
                )
            )

        qa = ConversationalRetrievalChain.from_llm(
            llm=chat,
            retriever=vectore_store.as_retriever(),
            return_source_documents=True,
        )

        prompt = f"""{translated_query.translated}"""

        result = qa({"question": prompt, "chat_history": chat_history})

        url_store = get_qdrant_db("o9_platform_all_pages")
        rag_chain = (
            {"context": url_store.as_retriever(), "query": RunnablePassthrough()}
            | source_url_retieval_chain()
        )
        try:
            source_urls = source_url_parser.parse(rag_chain.invoke(prompt)['text'])
            result["source_urls"] = source_urls.urls
            print('===============result===============', result["source_urls"])
        except Exception as e:
                print(e)

        if query_language == "English":
            return result
        else:
            try:
                translated_result = translate_parser.parse(
                    get_translate_chain().run(
                        text=result["answer"],
                        input_language="English",
                        output_language=query_language.language,
                    )
                )
                result["answer"] = translated_result.translated
            except Exception as e:
                print(e)
            finally:
                return result
            
    

