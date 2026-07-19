# GitMind — AI GitHub Repository Explainer

GitMind is a full RAG (Retrieval-Augmented Generation) application that takes any public GitHub repository URL and turns it into an interactive, explorable knowledge base — chat with the codebase, generate an architecture diagram, get an AI-written summary, and auto-generate a professional README, all from the actual source code.

## Overview

Understanding an unfamiliar codebase usually means hours of manually reading through files, tracing imports, and piecing together how components connect. GitMind automates that first pass: paste a repo URL, and within a couple of minutes it clones the repository, indexes the source code into a vector store, and gives you four ways to explore it — ask it questions directly, get a structured summary, view a generated architecture diagram, or download a written README.

## Features

- **Chat with the repo** — ask natural-language questions about the codebase and get answers grounded in the actual source files, with citations to which file each answer came from
- **Auto summary** — a structured breakdown covering what the project does, its tech stack, folder structure, key components, and how the pieces connect
- **Architecture diagrams** — an AI-generated Mermaid.js flowchart showing how the main modules and components interact
- **README generator** — writes a complete, professional `README.md` for the analyzed repository, downloadable as a file
- **Multi-user safe** — each user session gets its own isolated cloned repo and vector store, so multiple people can analyze different repositories concurrently without collisions

## Tech Stack

| Layer | Technology |
|---|---|
| UI / App framework | Streamlit |
| LLM orchestration | LangChain |
| LLM inference | Groq (`openai/gpt-oss-20b`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| Repo cloning | GitPython |
| Diagram rendering | Mermaid.js |

## Project Structure


## Project Structure

.
|-- app.py                     Streamlit UI, page routing, and orchestration
|-- requirements.txt
|-- .env                       GROQ_API_KEY (not committed)
`-- src/
    |-- clone_repo.py          Session-isolated repo cloning (temp dirs, no collisions)
    |-- load_files.py          Reads repo files, filters noise, caps size/count for speed
    |-- chunking.py            Splits documents into embeddable chunks
    |-- embeddings.py          Loads and caches the MiniLM embedding model
    |-- vector_store.py        Session-isolated ChromaDB vector store
    |-- retriever.py           Similarity search over the vector store
    |-- llm_response.py        Chat + summary generation via Groq
    |-- diagram_generator.py   Mermaid diagram generation with syntax sanitization
    |-- readme_generator.py    README generation for the analyzed repo
    `-- tech_detector.py       Detects languages/frameworks from repo contents
## Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com/keys) (free tier available)
- Git installed and available on your PATH

## Installation


git clone https://github.com/Spoorthi-27/AI-GITHUB.git
cd AI-GITHUB
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt


## Configuration

Create a `.env` file in the project root:


GROQ_API_KEY=your_groq_api_key_here



## Usage


python -m streamlit run app.py

Open the local URL Streamlit prints (typically `http://localhost:8501`), paste a public GitHub repository URL, and click **Analyze Repository**.

## How It Works

1. **Clone** — the target repository is cloned into an isolated, per-session temp directory
2. **Load & filter** — source files are read, with generated/vendor directories (`node_modules`, `.git`, `dist`, etc.) and oversized files skipped
3. **Chunk** — documents are split into overlapping chunks sized for retrieval
4. **Embed** — chunks are embedded using a cached MiniLM sentence-transformer model
5. **Index** — embeddings are stored in a session-scoped ChromaDB vector store
6. **Query** — chat questions, summaries, diagrams, and READMEs all retrieve relevant chunks and generate grounded responses via Groq's `openai/gpt-oss-20b`

## Contributing

Issues and pull requests are welcome. If you spot a bug or have a feature idea, please open an issue describing the repository you tested against and what you expected to happen.


