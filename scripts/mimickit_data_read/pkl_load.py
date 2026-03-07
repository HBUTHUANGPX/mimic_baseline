#!/usr/bin/env python3
"""Simple utilities to load and inspect .pkl motion files."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Iterable


def load_pkl(file_path: str | Path) -> Any:
    """Load one pickle file and return the deserialized object."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PKL file not found: {path}")
    if path.suffix.lower() != ".pkl":
        raise ValueError(f"Expected a .pkl file, got: {path}")

    with path.open("rb") as f:
        return pickle.load(f)


def _safe_len(obj: Any) -> str:
    try:
        return str(len(obj))
    except Exception:
        return "N/A"


def summarize_object(obj: Any, max_keys: int = 12) -> str:
    """Return a concise summary string for a loaded object."""
    obj_type = type(obj).__name__
    lines = [f"type={obj_type}", f"len={_safe_len(obj)}"]

    if isinstance(obj, dict):
        keys = list(obj.keys())
        preview = keys[:max_keys]
        lines.append(f"keys({len(keys)}): {preview}")

        shape_like = []
        for key in preview:
            value = obj[key]
            shape = getattr(value, "shape", None)
            if shape is not None:
                shape_like.append(f"{key}: shape={tuple(shape)}")
            else:
                shape_like.append(f"{key}: type={type(value).__name__}")
        if shape_like:
            lines.append("values: " + "; ".join(shape_like))

    elif isinstance(obj, (list, tuple)):
        if obj:
            first = obj[0]
            shape = getattr(first, "shape", None)
            if shape is not None:
                lines.append(f"first_item_shape={tuple(shape)}")
            lines.append(f"first_item_type={type(first).__name__}")

    return " | ".join(lines)


def iter_pkl_files(directory: str | Path, recursive: bool = True) -> Iterable[Path]:
    """Yield .pkl files under the given directory."""
    dir_path = Path(directory).expanduser().resolve()
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    pattern = "**/*.pkl" if recursive else "*.pkl"
    yield from sorted(dir_path.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and inspect .pkl files.")
    parser.add_argument("--file", type=str, help="Path to one .pkl file")
    parser.add_argument("--dir", type=str, help="Directory containing .pkl files")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search when using --dir",
    )
    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.error("Please provide --file or --dir")

    if args.file:
        obj = load_pkl(args.file)
        print(f"[OK] {Path(args.file).resolve()}")
        print(summarize_object(obj))

    if args.dir:
        files = list(iter_pkl_files(args.dir, recursive=not args.no_recursive))
        if not files:
            print(f"No .pkl files found in: {Path(args.dir).resolve()}")
            return

        print(f"Found {len(files)} .pkl files.")
        for pkl_file in files:
            try:
                obj = load_pkl(pkl_file)
                print(f"[OK] {pkl_file}")
                print(f"  {summarize_object(obj)}")
            except Exception as exc:
                print(f"[FAIL] {pkl_file}: {exc}")


if __name__ == "__main__":
    main()
