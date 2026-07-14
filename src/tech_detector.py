import os

TECH_INDICATORS = {
    "requirements.txt":     "Python",
    "setup.py":             "Python",
    "pyproject.toml":       "Python",
    "manage.py":            "Django",
    "fastapi":              "FastAPI",
    "flask":                "Flask",
    "package.json":         "Node.js",
    "next.config.js":       "Next.js",
    "next.config.ts":       "Next.js",
    "vite.config.js":       "Vite",
    "vite.config.ts":       "Vite",
    "angular.json":         "Angular",
    "vue.config.js":        "Vue.js",
    "pom.xml":              "Java (Maven)",
    "build.gradle":         "Java/Kotlin (Gradle)",
    "Cargo.toml":           "Rust",
    "go.mod":               "Go",
    "Gemfile":              "Ruby",
    "composer.json":        "PHP",
    "Dockerfile":           "Docker",
    "docker-compose.yml":   "Docker Compose",
    "docker-compose.yaml":  "Docker Compose",
    ".github":              "GitHub Actions",
    "prisma":               "Prisma ORM",
    "alembic.ini":          "Alembic (DB Migrations)",
    "langchain":            "LangChain",
    "chromadb":             "ChromaDB",
    "openai":               "OpenAI API",
    "transformers":         "HuggingFace",
    "torch":                "PyTorch",
    "tensorflow":           "TensorFlow",
    "sklearn":              "Scikit-learn",
    "pytest.ini":           "Pytest",
}


def detect_tech_stack(repo_path: str) -> list:
    detected = set()

    all_names = set()
    for root, dirs, files in os.walk(repo_path):
        depth = root.replace(repo_path, "").count(os.sep)
        if depth > 3:
            continue
        for name in files + dirs:
            all_names.add(name.lower())
            all_names.add(name)

    content_to_scan = _read_dependency_files(repo_path)

    for indicator, tech in TECH_INDICATORS.items():
        if indicator.lower() in all_names or indicator in all_names:
            detected.add(tech)
        elif content_to_scan and indicator.lower() in content_to_scan:
            detected.add(tech)

    return sorted(detected)


def _read_dependency_files(repo_path: str) -> str:
    content = ""
    for filename in ["requirements.txt", "package.json", "pyproject.toml"]:
        filepath = os.path.join(repo_path, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content += f.read().lower()
            except Exception:
                pass
    return content
