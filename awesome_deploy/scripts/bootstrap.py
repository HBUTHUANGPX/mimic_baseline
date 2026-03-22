import sys
from pathlib import Path


def ensure_repo_root_on_path(script_file) -> str:
    script_path = Path(script_file).resolve()
    repo_root = script_path.parents[2]
    repo_root_str = str(repo_root)
    if sys.path[0] != repo_root_str:
        sys.path.insert(0, repo_root_str)
    return repo_root_str
