import sys
from pathlib import Path

from awesome_deploy.scripts.bootstrap import ensure_repo_root_on_path


def test_script_bootstrap_adds_repo_root_to_sys_path():
    script_dir = Path("/tmp/repo/awesome_deploy/scripts")
    repo_root = Path("/tmp/repo")

    sys.path[:] = [str(script_dir)]

    ensure_repo_root_on_path(script_dir / "deploy_g1_mujoco.py")

    assert sys.path[0] == str(repo_root)
