from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()


def generate_mermaid_diagram(docs: list) -> str:
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0.1,
    )

    sample = docs[:15]
    file_list = []
    for doc in sample:
        filename = doc.metadata.get("filename", "")
        if filename:
            file_list.append(f"{filename}: {doc.page_content[:300]}")

    context = "\n\n".join(file_list)

    messages = [
        SystemMessage(content="""You are a software architect.
Generate ONLY a valid Mermaid.js flowchart diagram.
Output ONLY the mermaid code, nothing else.
No explanation, no markdown fences, just the diagram."""),

        HumanMessage(content=f"""Based on these repository files, generate a Mermaid flowchart
showing how the main components connect:

{context}

Use this format:
graph TD
    A[Component A] --> B[Component B]
    B --> C[Component C]

Max 10 nodes. Show how data flows between main modules.""")
    ]

    response = llm.invoke(messages)
    diagram = response.content.strip()

    if "```mermaid" in diagram:
        diagram = diagram.split("```mermaid")[1].split("```")[0].strip()
    elif "```" in diagram:
        diagram = diagram.split("```")[1].split("```")[0].strip()

    return diagram
