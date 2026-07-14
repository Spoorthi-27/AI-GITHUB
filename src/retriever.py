from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


def retrieve_docs(db: Chroma, query: str, k: int = 5) -> list:
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    docs = retriever.invoke(query)
    return docs
