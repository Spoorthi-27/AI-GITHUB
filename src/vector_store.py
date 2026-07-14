import os
import shutil
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_db"


def create_vector_store(chunks: list, embeddings: HuggingFaceEmbeddings) -> Chroma:
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return db


def load_vector_store(embeddings: HuggingFaceEmbeddings):
    if not os.path.exists(CHROMA_DIR):
        return None
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return db
