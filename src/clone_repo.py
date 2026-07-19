from git import Repo
import os
import stat
import shutil
import tempfile
import uuid


def _remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clone_repo(repo_url: str, session_id: str = None) -> str:
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]

    base_dir = os.path.join(tempfile.gettempdir(), "gitmind_sessions")
    os.makedirs(base_dir, exist_ok=True)

    repo_path = os.path.join(base_dir, f"repo_{session_id}")

    if os.path.exists(repo_path):
        shutil.rmtree(repo_path, onerror=_remove_readonly)

    try:
        Repo.clone_from(repo_url, repo_path)
        return repo_path
    except Exception as e:
        raise Exception(f"Failed to clone repository: {str(e)}")


def cleanup_repo(repo_path: str):
    if repo_path and os.path.exists(repo_path):
        shutil.rmtree(repo_path, onerror=_remove_readonly)
