import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_MUJOCO = REPO_ROOT / "mujoco_deploy/deploy/deploy_mujoco/deploy_mujoco.py"


def _is_video_recorder_stop_call(node):
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "stop"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "video_recorder"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "self"
    )


def _find_simulator_run(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "simulator":
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "run":
                return item
    raise AssertionError("simulator.run was not found")


def main():
    tree = ast.parse(DEPLOY_MUJOCO.read_text())
    run_fn = _find_simulator_run(tree)

    for node in ast.walk(run_fn):
        if isinstance(node, ast.Try):
            if any(_is_video_recorder_stop_call(child) for child in ast.walk(ast.Module(body=node.finalbody, type_ignores=[]))):
                return

    raise AssertionError("self.video_recorder.stop() must be called from a finally block in simulator.run")


if __name__ == "__main__":
    main()
