import os
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = (
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".kt", ".go", ".rs",
    ".c", ".cpp", ".h", ".cs", ".rb", ".php", ".swift",
    ".md", ".txt", ".rst",
    ".yaml", ".yml", ".json", ".toml",
    ".sh", ".bash",
)

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv",
    "venv", "env", "dist", "build", ".next",
    ".nuxt", "coverage", ".pytest_cache", ".mypy_cache",
}


def load_files(repo_path: str) -> list:
    documents = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for file in files:
            file_path = os.path.join(root, file)

            if not _is_supported(file):
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()

                if not content:
                    continue

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "filename": file,
                        "extension": os.path.splitext(file)[1],
                    }
                )
                documents.append(doc)

            except Exception:
                pass

    return documents


def _is_supported(filename: str) -> bool:
    full_name_matches = {"Dockerfile", "Makefile", "Procfile"}
    if filename in full_name_matches:
        return True
    return any(filename.endswith(ext) for ext in SUPPORTED_EXTENSIONS)
