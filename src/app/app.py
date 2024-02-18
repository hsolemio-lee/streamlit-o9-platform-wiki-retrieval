import streamlit as st
from streamlit_chat import message
from service.retrieval_service import RetrievalService
from typing import List
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

retrieval_service = RetrievalService()

st.header("Cloud SCM Dev - o9 Platform Helper bot")
openai_api_key = st.text_input("OpenAI API Key", placeholder="Please enter your API Key")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

if st.button("Reset", type="primary"):
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answers_history"] = []
    st.session_state["chat_history"] = []

prompt = st.chat_input(placeholder="Enter your prompt here..")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def create_sources_string(source_urls: List[str]) -> str:
    if not source_urls:
        return ""
    source_urls.sort()
    sources_string = "Related Sources:\n"
    for i, source in enumerate(source_urls):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

if prompt:
    with st.spinner("Generating response.."):
        generated_response = retrieval_service.run_chat_retrieval_qa(
            query=prompt,
            collection="o9_platform_wiki",
            chat_history=st.session_state["chat_history"],
        )
        
        source_urls = generated_response["source_urls"]
        formatted_response = (
            f"{generated_response['answer']}\n\n\n\n{create_sources_string(source_urls)}"
        )
        
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(
            (generated_response["question"], generated_response["answer"])
        )

    if st.session_state["chat_answers_history"]:
        for generated_response, user_query in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
        ):
            message(user_query, is_user=True)
            message(generated_response)
