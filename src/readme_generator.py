from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()


def generate_readme(docs: list, tech_stack: list, repo_url: str) -> str:
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-20b",
        temperature=0.3,
        reasoning_effort="low",
        max_tokens=4000,
    )

    sample = docs[:25]
    context_parts = []
    for doc in sample:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"File: {source}\n{doc.page_content[:400]}")

    context = "\n\n".join(context_parts)
    tech_str = ", ".join(tech_stack) if tech_stack else "Not detected"

    messages = [
        SystemMessage(content="""You are a technical writer and senior developer.
Generate professional, complete README.md files.
Use proper markdown. Be specific — use actual file names and commands from the code."""),

        HumanMessage(content=f"""Generate a complete README.md for this repository.

Repository URL: {repo_url}
Tech Stack: {tech_str}

Files:
{context}

Include these sections:
# Project Name
## Overview
## Features
## Tech Stack
## Project Structure
## Prerequisites
## Installation
## Configuration
## Usage
## How It Works
## Contributing
## License""")
    ]

    response = llm.invoke(messages)

    if not response.content or not response.content.strip():
        raise ValueError(
            "README generation returned empty content — the model may have "
            "used its full token budget on reasoning. Try increasing max_tokens "
            "further or reducing the number of sample files."
        )

    return response.content