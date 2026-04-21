import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import yaml


_MOTION_TAKE_PATTERN = re.compile(r"^(?P<action>.+?)__A\d{3}(?P<mirrored>_M)?$")


class MotionYamlDumper(yaml.SafeDumper):
    pass


def _represent_motion_yaml_string(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = '"' if any(character.isspace() for character in value) else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


MotionYamlDumper.add_representer(str, _represent_motion_yaml_string)


def motion_action_key(path: str | Path) -> str:
    stem = Path(path).stem
    match = _MOTION_TAKE_PATTERN.match(stem)
    if match is None:
        return stem
    return match.group("action")


def is_mirrored_motion(path: str | Path) -> bool:
    stem = Path(path).stem
    match = _MOTION_TAKE_PATTERN.match(stem)
    return bool(match and match.group("mirrored"))


def select_representative_motion(paths: Iterable[str | Path]) -> Path:
    candidates = sorted((Path(path) for path in paths), key=lambda path: path.as_posix())
    if not candidates:
        raise ValueError("Expected at least one motion path to select from.")

    non_mirrored = [path for path in candidates if not is_mirrored_motion(path)]
    return non_mirrored[0] if non_mirrored else candidates[0]


def collect_pruned_motion_files(source_dir: str | Path) -> list[Path]:
    source_path = Path(source_dir)
    grouped_paths: dict[str, list[Path]] = defaultdict(list)

    for npz_path in sorted(source_path.rglob("*.npz"), key=lambda path: path.as_posix()):
        grouped_paths[motion_action_key(npz_path)].append(npz_path)

    selected = [
        select_representative_motion(grouped_paths[action_key])
        for action_key in sorted(grouped_paths)
    ]
    return sorted(selected, key=lambda path: path.as_posix())


def _format_yaml_path(path: Path, relative_to: Path | None) -> str:
    if relative_to is None:
        return str(path)

    try:
        return path.resolve().relative_to(relative_to.resolve()).as_posix()
    except ValueError:
        return str(path)


def generate_motion_yaml(
    source_dir: str | Path,
    output_path: str | Path,
    motion_group_name: str = "soma_uniform_bvh_export_pruned",
    relative_to: str | Path | None = None,
) -> dict:
    source_path = Path(source_dir)
    output_file = Path(output_path)
    relative_root = Path(relative_to) if relative_to is not None else Path.cwd()

    selected_paths = collect_pruned_motion_files(source_path)
    yaml_paths = [_format_yaml_path(path, relative_root) for path in selected_paths]

    payload = {
        "motion_group": {
            motion_group_name: {
                "file_name": yaml_paths,
                "folder_name": [],
                "wo_file_name": [],
                "wo_folder_name": [],
            }
        }
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        yaml.dump(
            payload,
            Dumper=MotionYamlDumper,
            sort_keys=False,
            allow_unicode=True,
            width=10_000,
        ),
        encoding="utf-8",
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a pruned motion YAML with one representative npz per action.",
    )
    parser.add_argument(
        "--source-dir",
        default="soma-retargeter/assets/motions/soma_uniform_bvh_export",
        help="Directory containing motion .npz files.",
    )
    parser.add_argument(
        "--output",
        default="scripts/rsl_rl/motion_file_pruned.yaml",
        help="Path to the generated YAML file.",
    )
    parser.add_argument(
        "--motion-group-name",
        default="soma_uniform_bvh_export_pruned",
        help="Motion group name written under motion_group in the generated YAML.",
    )
    parser.add_argument(
        "--relative-to",
        default=".",
        help="Write file_name entries relative to this directory when possible.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    payload = generate_motion_yaml(
        source_dir=args.source_dir,
        output_path=args.output,
        motion_group_name=args.motion_group_name,
        relative_to=args.relative_to,
    )

    group = payload["motion_group"][args.motion_group_name]
    print(
        f"Generated {args.output} with {len(group['file_name'])} pruned motions "
        f"from {args.source_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
