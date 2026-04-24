from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_sys_path(module_file: str | Path) -> Path:
    module_path = Path(module_file).resolve()
    repo_root = module_path.parents[2]
    resolved = str(repo_root)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return repo_root
