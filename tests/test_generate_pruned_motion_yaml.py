import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rsl_rl.load_motion_file import collect_npz_paths


def test_motion_action_key_strips_take_id_and_optional_m_suffix():
    from scripts.rsl_rl.generate_pruned_motion_yaml import motion_action_key

    assert motion_action_key(Path("body_check_001__A548.npz")) == "body_check_001"
    assert motion_action_key(Path("body_check_001__A548_M.npz")) == "body_check_001"
    assert motion_action_key(Path("Neutral_walk_forward_002.npz")) == "Neutral_walk_forward_002"


def test_select_representative_motion_prefers_sorted_non_m_variant():
    from scripts.rsl_rl.generate_pruned_motion_yaml import select_representative_motion

    candidates = [
        Path("body_check_001__A551_M.npz"),
        Path("body_check_001__A550.npz"),
        Path("body_check_001__A548_M.npz"),
        Path("body_check_001__A549.npz"),
    ]

    assert select_representative_motion(candidates) == Path("body_check_001__A549.npz")


def test_select_representative_motion_falls_back_to_m_when_needed():
    from scripts.rsl_rl.generate_pruned_motion_yaml import select_representative_motion

    candidates = [
        Path("body_check_001__A551_M.npz"),
        Path("body_check_001__A548_M.npz"),
    ]

    assert select_representative_motion(candidates) == Path("body_check_001__A548_M.npz")


def test_generate_yaml_writes_loader_compatible_motion_group(tmp_path: Path):
    from scripts.rsl_rl.generate_pruned_motion_yaml import generate_motion_yaml

    source_dir = tmp_path / "motions"
    source_dir.mkdir()
    (source_dir / "body_check_001__A549.npz").touch()
    (source_dir / "body_check_001__A548_M.npz").touch()
    (source_dir / "body_check_001__A551_M.npz").touch()
    (source_dir / "take_a_sip_180_R_001__A552_M.npz").touch()
    (source_dir / "Neutral_walk_forward_002.npz").touch()

    output_yaml = tmp_path / "motion_file_pruned.yaml"

    generate_motion_yaml(
        source_dir=source_dir,
        output_path=output_yaml,
        motion_group_name="pruned_set",
    )

    payload = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))
    assert sorted(payload["motion_group"].keys()) == ["pruned_set"]
    group = payload["motion_group"]["pruned_set"]
    assert group["folder_name"] == []
    assert group["wo_file_name"] == []
    assert group["wo_folder_name"] == []
    assert group["file_name"] == [
        str(source_dir / "Neutral_walk_forward_002.npz"),
        str(source_dir / "body_check_001__A549.npz"),
        str(source_dir / "take_a_sip_180_R_001__A552_M.npz"),
    ]

    collected = collect_npz_paths(str(output_yaml))
    assert collected == {
        "pruned_set": [
            str(source_dir / "Neutral_walk_forward_002.npz"),
            str(source_dir / "body_check_001__A549.npz"),
            str(source_dir / "take_a_sip_180_R_001__A552_M.npz"),
        ]
    }
