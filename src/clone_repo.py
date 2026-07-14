from git import Repo
import os
import shutil


def clone_repo(repo_url: str) -> str:
    repo_path = "cloned_repo"

    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    try:
        Repo.clone_from(repo_url, repo_path)
        return repo_path
    except Exception as e:
        raise Exception(f"Failed to clone repository: {str(e)}")
