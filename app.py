import warnings
warnings.filterwarnings("ignore")
import uuid
import streamlit as st
from src.clone_repo import clone_repo, cleanup_repo
from src.load_files import load_files
from src.chunking import chunk_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store, cleanup_vector_store
from src.retriever import retrieve_docs
from src.llm_response import generate_response, generate_summary
from src.tech_detector import detect_tech_stack
from src.diagram_generator import generate_mermaid_diagram
from src.readme_generator import generate_readme
import streamlit.components.v1 as components
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GitMind — AI Repo Explainer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── All CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #080b0f !important;
    color: #e8e6e0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stSidebar"] { display: none; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
footer { display: none; }
#MainMenu { display: none; }

/* hide default streamlit padding */
[data-testid="stMain"] > div { padding-top: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Hero section ── */
.gm-hero {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 24px;
    position: relative;
    overflow: hidden;
    background: #080b0f;
}

.gm-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 40%, rgba(0,255,136,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 60%, rgba(0,180,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}

.gm-grid {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
}

.gm-logo {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3em;
    color: #00ff88;
    text-transform: uppercase;
    margin-bottom: 32px;
    opacity: 0.9;
}

.gm-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(42px, 8vw, 96px);
    font-weight: 800;
    line-height: 0.95;
    letter-spacing: -0.03em;
    text-align: center;
    color: #f0ede6;
    margin-bottom: 24px;
}

.gm-title span {
    color: #00ff88;
    position: relative;
}

.gm-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 300;
    color: #6b7280;
    text-align: center;
    max-width: 520px;
    line-height: 1.7;
    margin-bottom: 56px;
}

/* ── Input area ── */
.gm-input-wrap {
    width: 100%;
    max-width: 680px;
    position: relative;
    margin-bottom: 20px;
}

.stTextInput > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 4px !important;
    color: #e8e6e0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    padding: 16px 20px !important;
    transition: border-color 0.2s !important;
    box-shadow: none !important;
}

.stTextInput > div > div:focus-within {
    border-color: #00ff88 !important;
    background: rgba(0,255,136,0.03) !important;
    box-shadow: 0 0 0 1px rgba(0,255,136,0.2) !important;
}

.stTextInput input {
    color: #e8e6e0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
}

.stTextInput input::placeholder { color: #4b5563 !important; }
.stTextInput label { display: none !important; }

/* ── Buttons ── */
.stButton > button {
    background: #00ff88 !important;
    color: #080b0f !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 14px 32px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: #00e87a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,255,136,0.25) !important;
}

.stButton > button:active { transform: translateY(0) !important; }

/* ── Feature pills ── */
.gm-pills {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 48px;
}

.gm-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 400;
    color: #4b5563;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 2px;
    padding: 6px 14px;
    letter-spacing: 0.05em;
}

/* ── Dashboard wrapper ── */
.gm-dashboard {
    min-height: 100vh;
    background: #080b0f;
    padding: 0;
}

/* ── Top bar ── */
.gm-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 40px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: rgba(8,11,15,0.95);
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
}

.gm-topbar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 16px;
    font-weight: 800;
    color: #f0ede6;
    letter-spacing: -0.02em;
}

.gm-topbar-logo span { color: #00ff88; }

.gm-topbar-repo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #4b5563;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 2px;
    padding: 6px 14px;
    max-width: 400px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.gm-topbar-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #00ff88;
}

.gm-status-dot {
    width: 6px;
    height: 6px;
    background: #00ff88;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Stats row ── */
.gm-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.gm-stat {
    padding: 28px 40px;
    border-right: 1px solid rgba(255,255,255,0.06);
}

.gm-stat:last-child { border-right: none; }

.gm-stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: #f0ede6;
    line-height: 1;
    margin-bottom: 6px;
}

.gm-stat-num.green { color: #00ff88; }

.gm-stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #4b5563;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Tech stack badges ── */
.gm-tech-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    padding: 20px 40px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    align-items: center;
}

.gm-tech-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #4b5563;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-right: 4px;
}

.gm-tech-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #00ff88;
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 2px;
    padding: 4px 10px;
}

/* ── Tab nav ── */
.gm-tabs {
    display: flex;
    gap: 0;
    padding: 0 40px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.gm-tab {
    font-family: 'Syne', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #4b5563;
    padding: 16px 24px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.15s;
    white-space: nowrap;
}

.gm-tab.active {
    color: #00ff88;
    border-bottom-color: #00ff88;
}

/* ── Content area ── */
.gm-content {
    padding: 40px;
    min-height: 60vh;
}

/* ── Section title ── */
.gm-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.gm-section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* ── Answer card ── */
.gm-answer {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 3px solid #00ff88;
    border-radius: 4px;
    padding: 28px 32px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.8;
    color: #c9c5bc;
    white-space: pre-wrap;
    margin-top: 20px;
}

/* ── Source files ── */
.gm-sources {
    margin-top: 16px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.gm-source-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #6b7280;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 2px;
    padding: 4px 10px;
}

/* ── Chat input area ── */
.gm-chat-area {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 4px;
    padding: 24px;
    margin-bottom: 16px;
}

.gm-chat-hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #374151;
    margin-bottom: 16px;
}

.gm-chat-examples {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}

.gm-example {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #4b5563;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 2px;
    padding: 6px 12px;
    cursor: pointer;
    transition: all 0.15s;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 2px !important;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #00ff88, #00e0ff) !important;
    border-radius: 2px !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #00ff88 !important;
}

/* ── Alerts ── */
.stSuccess {
    background: rgba(0,255,136,0.08) !important;
    border: 1px solid rgba(0,255,136,0.2) !important;
    color: #00ff88 !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

.stError {
    background: rgba(255,80,80,0.08) !important;
    border: 1px solid rgba(255,80,80,0.2) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

/* ── Markdown output ── */
[data-testid="stMarkdownContainer"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.8 !important;
    color: #c9c5bc !important;
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: 'Syne', sans-serif !important;
    color: #f0ede6 !important;
    font-weight: 700 !important;
    margin: 24px 0 12px !important;
}

[data-testid="stMarkdownContainer"] code {
    background: rgba(0,255,136,0.08) !important;
    color: #00ff88 !important;
    border-radius: 2px !important;
    padding: 2px 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}

[data-testid="stMarkdownContainer"] pre {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 4px !important;
    padding: 20px !important;
}

/* ── Select box ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 4px !important;
    color: #e8e6e0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: transparent !important;
    color: #00ff88 !important;
    border: 1px solid rgba(0,255,136,0.3) !important;
    border-radius: 4px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 10px 24px !important;
    transition: all 0.15s !important;
}

.stDownloadButton > button:hover {
    background: rgba(0,255,136,0.08) !important;
    border-color: #00ff88 !important;
}

/* ── Diagram box ── */
.gm-diagram-box {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 4px;
    padding: 32px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #6b7280;
    line-height: 1.8;
    overflow-x: auto;
}

/* ── Columns ── */
[data-testid="column"] { padding: 0 8px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child { padding-right: 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key in ["db", "docs", "tech_stack", "repo_url", "processed",
            "file_count", "chunk_count", "active_tab", "repo_path"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if not st.session_state.processed:
    st.session_state.processed = False
if not st.session_state.active_tab:
    st.session_state.active_tab = "chat"

# ═════════════════════════════════════════════════════════════════════════════
# HERO PAGE — shown before any repo is processed
# ═════════════════════════════════════════════════════════════════════════════
if not st.session_state.processed:

    st.markdown("""
    <div class="gm-hero">
        <div class="gm-grid"></div>
        <div class="gm-logo">AI Github Repo Explainer</div>
        <h1 class="gm-title">Understand any<br><span>codebase</span><br>instantly.</h1>
        <p class="gm-subtitle">
            Paste a GitHub URL. Get an AI-powered explanation,
            architecture diagram, and auto-generated README — in seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Input centered
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        repo_url = st.text_input(
            "url",
            placeholder="https://github.com/username/repository",
            key="repo_input"
        )
        process_btn = st.button("→  Analyze Repository", use_container_width=True)

    st.markdown("""
    <div style="display:flex; justify-content:center;">
        <div class="gm-pills">
            <span class="gm-pill">RAG Pipeline</span>
            <span class="gm-pill">ChromaDB Vector Store</span>
            <span class="gm-pill">LLaMA 3 via Groq</span>
            <span class="gm-pill">MiniLM Embeddings</span>
            <span class="gm-pill">Architecture Diagrams</span>
            <span class="gm-pill">README Generator</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Process
    if process_btn and repo_url:
        col_l2, col_c2, col_r2 = st.columns([1, 2, 1])
        with col_c2:
            progress = st.progress(0, text="")
            try:
                progress.progress(10, text="↓  cloning repository...")
                repo_path = clone_repo(repo_url, session_id=st.session_state.session_id)

                progress.progress(28, text="⬡  detecting tech stack...")
                tech_stack = detect_tech_stack(repo_path)
                st.session_state.tech_stack = tech_stack

                progress.progress(45, text="◈  loading files...")
                docs = load_files(repo_path)
                if not docs:
                    st.error("No readable files found.")
                    st.stop()
                st.session_state.docs = docs

                progress.progress(60, text="⟁  chunking documents...")
                chunks = chunk_documents(docs)

                progress.progress(75, text="◉  creating embeddings — first run downloads model...")
                embeddings = get_embeddings()

                progress.progress(90, text="▣  building vector store...")
                db = create_vector_store(chunks, embeddings, session_id=st.session_state.session_id)

                st.session_state.db = db
                st.session_state.repo_url = repo_url
                st.session_state.repo_path = repo_path
                st.session_state.file_count = len(docs)
                st.session_state.chunk_count = len(chunks)
                st.session_state.processed = True

                progress.progress(100, text="✓  ready")
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif process_btn and not repo_url:
        col_l2, col_c2, col_r2 = st.columns([1, 2, 1])
        with col_c2:
            st.warning("Enter a GitHub URL first.")

# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD — shown after repo is processed
# ═════════════════════════════════════════════════════════════════════════════
else:

    repo_short = st.session_state.repo_url.replace("https://github.com/", "") if st.session_state.repo_url else ""
    file_count = st.session_state.file_count or 0
    chunk_count = st.session_state.chunk_count or 0
    tech_count = len(st.session_state.tech_stack or [])

    # ── Top bar ──
    st.markdown(f"""
    <div class="gm-topbar">
        <div class="gm-topbar-logo">Git<span>Mind</span></div>
        <div class="gm-topbar-repo">github.com/{repo_short}</div>
        <div class="gm-topbar-status">
            <div class="gm-status-dot"></div>
            indexed &amp; ready
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats row ──
    st.markdown(f"""
    <div class="gm-stats">
        <div class="gm-stat">
            <div class="gm-stat-num green">{file_count}</div>
            <div class="gm-stat-label">Files indexed</div>
        </div>
        <div class="gm-stat">
            <div class="gm-stat-num">{chunk_count}</div>
            <div class="gm-stat-label">Vector chunks</div>
        </div>
        <div class="gm-stat">
            <div class="gm-stat-num">{tech_count}</div>
            <div class="gm-stat-label">Technologies</div>
        </div>
        <div class="gm-stat">
            <div class="gm-stat-num green">RAG</div>
            <div class="gm-stat-label">Pipeline active</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tech badges ──
    if st.session_state.tech_stack:
        badges = "".join([
            f'<span class="gm-tech-badge">{t}</span>'
            for t in st.session_state.tech_stack
        ])
        st.markdown(f"""
        <div class="gm-tech-row">
            <span class="gm-tech-label">Stack →</span>
            {badges}
        </div>
        """, unsafe_allow_html=True)

    # ── Tab selector using selectbox ──
    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.06);margin:0 0 24px;"></div>', unsafe_allow_html=True)

    col_tab, col_new = st.columns([3, 1])
    with col_tab:
        tab_choice = st.selectbox(
            "mode",
            ["💬  Chat with repo", "📋  Auto summary", "🗺️  Architecture diagram", "📄  Generate README"],
            label_visibility="collapsed",
            key="tab_select"
        )
    with col_new:
        if st.button("← New repo", use_container_width=True):
            cleanup_repo(st.session_state.get("repo_path"))
            cleanup_vector_store(st.session_state.session_id)
            for key in ["db", "docs", "tech_stack", "repo_url", "processed",
                        "file_count", "chunk_count", "active_tab", "repo_path"]:
                st.session_state[key] = None
            st.session_state.processed = False
            st.rerun()

    st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════
    # TAB: CHAT
    # ═══════════════════════════════
    if "Chat" in tab_choice:

        st.markdown('<div class="gm-section-title">Ask the codebase</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="gm-chat-hint">// try asking —</div>
        <div class="gm-chat-examples">
            <span class="gm-example">How does authentication work?</span>
            <span class="gm-example">What does main.py do?</span>
            <span class="gm-example">How is the database connected?</span>
            <span class="gm-example">What APIs are exposed?</span>
            <span class="gm-example">Explain the folder structure</span>
        </div>
        """, unsafe_allow_html=True)

        question = st.text_input(
            "question",
            placeholder="// ask anything about this repository...",
            key="chat_question",
            label_visibility="collapsed"
        )

        ask_col, _ = st.columns([1, 3])
        with ask_col:
            ask_btn = st.button("→  Ask", use_container_width=True, key="ask_btn")

        if ask_btn and question:
            with st.spinner("searching vector store and generating answer..."):
                retrieved = retrieve_docs(st.session_state.db, question)
                answer = generate_response(question, retrieved)

            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="gm-section-title">Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="gm-answer">{answer}</div>', unsafe_allow_html=True)

            sources = list(set(doc.metadata.get("filename", "unknown") for doc in retrieved))
            chips = "".join([f'<span class="gm-source-chip">{s}</span>' for s in sources])
            st.markdown(f"""
            <div style="margin-top:16px;">
                <div class="gm-section-title">Sources used</div>
                <div class="gm-sources">{chips}</div>
            </div>
            """, unsafe_allow_html=True)

        elif ask_btn and not question:
            st.warning("Type a question first.")

    # ═══════════════════════════════
    # TAB: SUMMARY
    # ═══════════════════════════════
    elif "summary" in tab_choice.lower():

        st.markdown('<div class="gm-section-title">Repository summary</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#4b5563;margin-bottom:24px;">
        // AI analyzes the full codebase and generates a comprehensive breakdown
        </p>
        """, unsafe_allow_html=True)

        sum_col, _ = st.columns([1, 3])
        with sum_col:
            gen_btn = st.button("→  Generate Summary", use_container_width=True, key="sum_btn")

        if gen_btn:
            with st.spinner("analyzing repository structure and content..."):
                summary = generate_summary(st.session_state.docs)
            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            st.markdown(summary)

    # ═══════════════════════════════
    # TAB: DIAGRAM
    # ═══════════════════════════════
    elif "diagram" in tab_choice.lower():

        st.markdown('<div class="gm-section-title">Architecture diagram</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#4b5563;margin-bottom:24px;">
        // generates a Mermaid.js component flow diagram from your actual code
        </p>
        """, unsafe_allow_html=True)

        diag_col, _ = st.columns([1, 3])
        with diag_col:
            diag_btn = st.button("→  Generate Diagram", use_container_width=True, key="diag_btn")

        if diag_btn:
            with st.spinner("tracing component relationships..."):
                diagram = generate_mermaid_diagram(st.session_state.docs)

            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="gm-section-title">Mermaid source</div>', unsafe_allow_html=True)
            st.code(diagram, language="text")

            st.markdown('<div class="gm-section-title" style="margin-top:32px;">Rendered diagram</div>', unsafe_allow_html=True)

            mermaid_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
            <style>
                body {{ background: transparent; margin: 0; padding: 20px; }}
                .mermaid {{ background: transparent; }}
                .mermaid svg {{ max-width: 100%; }}
            </style>
            </head>
            <body>
            <div class="mermaid">{diagram}</div>
            <script>
                mermaid.initialize({{
                    startOnLoad: true,
                    theme: 'dark',
                    themeVariables: {{
                        primaryColor: '#1a1f2e',
                        primaryTextColor: '#e8e6e0',
                        primaryBorderColor: '#00ff88',
                        lineColor: '#00ff88',
                        secondaryColor: '#0d1117',
                        tertiaryColor: '#161b22',
                        background: 'transparent',
                        mainBkg: '#0d1117',
                        nodeBorder: '#00ff88',
                        clusterBkg: '#161b22',
                        titleColor: '#e8e6e0',
                        edgeLabelBackground: '#0d1117',
                        fontSize: '14px'
                    }}
                }});
            </script>
            </body>
            </html>
            """
            components.html(mermaid_html, height=500, scrolling=True)

    # ═══════════════════════════════
    # TAB: README
    # ═══════════════════════════════
    elif "readme" in tab_choice.lower():

        st.markdown('<div class="gm-section-title">README generator</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#4b5563;margin-bottom:24px;">
        // AI writes a complete professional README based on your actual code
        </p>
        """, unsafe_allow_html=True)

        readme_col, _ = st.columns([1, 3])
        with readme_col:
            readme_btn = st.button("→  Generate README", use_container_width=True, key="readme_btn")

        if readme_btn:
            with st.spinner("writing README from codebase analysis..."):
                readme = generate_readme(
                    st.session_state.docs,
                    st.session_state.tech_stack or [],
                    st.session_state.repo_url or "",
                )

            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

            dl_col, _ = st.columns([1, 3])
            with dl_col:
                st.download_button(
                    "↓  Download README.md",
                    data=readme,
                    file_name="README.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
            st.markdown(readme)