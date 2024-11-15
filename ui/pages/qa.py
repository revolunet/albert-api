import logging
import traceback

from openai import OpenAI
import requests
import streamlit as st
from streamlit_extras.stylable_container import stylable_container

from config import BASE_URL
from utils import get_collections, get_models, header, set_config

# Config
set_config()
API_KEY = header()

# Data
try:
    language_models, embeddings_models, _ = get_models(api_key=API_KEY)
    collections = get_collections(api_key=API_KEY)
except Exception:
    st.error("Error to fetch user data.")
    logging.error(traceback.format_exc())
    st.stop()

openai_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


with stylable_container(
    key="Chat",
    css_styles="""
    button {
        float: right;
    }
    """,
):
    col1, col2 = st.columns(2)
    with col2:
        new_chat = st.button("Nouvelle question")
    with col1:
        st.title("Albert RH QA")

if prompt := st.chat_input("Message to Albert"):
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    # https://resana.numerique.gouv.fr/public/document/afficherOnlyOffice?slug=891981&id_information=22003369

    with st.chat_message("assistant"):
        try:
            data = {
                "collections": "aaa",
                "model": "AgentPublic/llama3-instruct-8b",
                "k": 2,
                "prompt": prompt,
            }
            response = requests.post(f"{BASE_URL}/search", json=data, headers={"Authorization": f"Bearer {API_KEY}"})
            assert response.status_code == 200
            prompt_template = "Réponds à la question suivante en te basant sur les documents ci-dessous : {prompt}\n\nDocuments :\n{chunks}"
            chunks = "\n".join([result["chunk"]["content"] for result in response.json()["data"]])

            sources = list(set(result["chunk"]["metadata"]["document_name"] for result in response.json()["data"]))

            prompt = prompt_template.format(prompt=prompt, chunks=chunks)
            messages = st.session_state.messages[:-1] + [{"role": "user", "content": prompt}]
        except Exception:
            logging.error(traceback.format_exc())
            st.error("Error to fetch user data.")
            st.stop()
