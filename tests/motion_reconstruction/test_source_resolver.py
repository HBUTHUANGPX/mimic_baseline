from pathlib import Path

import yaml

from motion_reconstruction.data.source_resolver import MotionSourceResolver


def test_resolver_collects_all_legacy_motion_groups_with_excludes(tmp_path: Path):
    root = tmp_path
    explicit = root / "explicit.npz"
    duplicate = root / "duplicate.npz"
    explicit.touch()
    duplicate.touch()

    folder = root / "folder"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    folder_motion = folder / "folder_motion.npz"
    folder_duplicate = folder / "duplicate.npz"
    nested_motion = nested / "nested_motion.npz"
    excluded = nested / "excluded.npz"
    for path in (folder_motion, folder_duplicate, nested_motion, excluded):
        path.touch()

    second = root / "second.npz"
    second.touch()

    yaml_path = root / "motion_file.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "motion_group": {
                    "group_a": {
                        "file_name": [str(explicit), str(duplicate)],
                        "folder_name": [str(folder)],
                        "wo_file_name": [str(excluded)],
                        "wo_folder_name": [],
                    },
                    "group_b": {
                        "file_name": [str(second)],
                        "folder_name": [],
                        "wo_file_name": [],
                        "wo_folder_name": [],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    resolved = MotionSourceResolver.from_legacy_yaml(yaml_path).resolve()

    assert set(resolved.by_group) == {"group_a", "group_b"}
    assert resolved.by_group["group_b"] == [second]
    assert explicit in resolved.by_group["group_a"]
    assert duplicate in resolved.by_group["group_a"]
    assert folder_motion in resolved.by_group["group_a"]
    assert nested_motion in resolved.by_group["group_a"]
    assert excluded not in resolved.by_group["group_a"]
    assert folder_duplicate not in resolved.by_group["group_a"]
    assert resolved.files == sorted(
        resolved.by_group["group_a"] + resolved.by_group["group_b"]
    )
