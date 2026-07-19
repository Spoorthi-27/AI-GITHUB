import os
import shutil
import stat
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def _remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def create_vector_store(chunks: list, embeddings: HuggingFaceEmbeddings, session_id: str) -> Chroma:
    """
    Each session gets its own Chroma persist directory, so two users
    processing different repos at the same time don't overwrite or
    read each other's vector data.
    """
    chroma_dir = os.path.join("chroma_sessions", f"session_{session_id}")

    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir, onerror=_remove_readonly)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    return db


def load_vector_store(embeddings: HuggingFaceEmbeddings, session_id: str):
    chroma_dir = os.path.join("chroma_sessions", f"session_{session_id}")
    if not os.path.exists(chroma_dir):
        return None
    db = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
    )
    return db


def cleanup_vector_store(session_id: str):
    """Call when a session ends or a new repo is processed, to free disk space."""
    chroma_dir = os.path.join("chroma_sessions", f"session_{session_id}")
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir, onerror=_remove_readonly)