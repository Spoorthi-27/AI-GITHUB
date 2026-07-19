import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings


@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings