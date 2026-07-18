from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

SYSTEM_PROMPT = """You are an expert software engineer and code explainer.
You are analyzing a GitHub repository and answering questions about it.

Rules:
- Always mention which file the information comes from (e.g. "In auth.py...")
- Be concise, technical, and accurate
- If the context does not contain enough info, say so clearly
- Use bullet points for lists
- Format code with proper markdown code blocks
- Never make up code that is not in the context
"""


def get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")
    return ChatGroq(
        groq_api_key=api_key,
        model_name="openai/gpt-oss-20b",
        temperature=0.2,
    )


def generate_response(query: str, docs: list) -> str:
    llm = get_llm()

    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown file")
        context_parts.append(f"--- From: {source} ---\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Repository context:

{context}

Question: {query}

Answer based only on the repository content above.""")
    ]

    response = llm.invoke(messages)
    return response.content


def generate_summary(docs: list) -> str:
    llm = get_llm()

    sample_docs = docs[:20]
    context_parts = []
    for doc in sample_docs:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"File: {source}\n{doc.page_content[:500]}")

    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Analyze this repository and provide a comprehensive summary.

Repository files:
{context}

Provide:
1. **What this project does** (2-3 sentences)
2. **Tech stack detected** (list all languages, frameworks, libraries)
3. **Project structure** (main folders and what they contain)
4. **Key components** (most important files and their purpose)
5. **How to get started** (based on what you see in the code)
6. **Architecture overview** (how the main parts connect)

Be specific and technical. Reference actual file names.""")
    ]

    response = llm.invoke(messages)
    return response.content
