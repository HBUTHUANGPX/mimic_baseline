from __future__ import annotations

from pathlib import Path

from dataset_converter.common.paths import ensure_workspace_on_sys_path


def export_nymeria_to_soma_bvh_bridge(
    sequence_dir: str | Path,
    *,
    output_path: str | Path,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda",
    batch_size: int | None = None,
    soma_x_root: str | Path,
    smpl_model_path: str | Path | None = None,
) -> Path:
    ensure_workspace_on_sys_path()
    from nymeria_parse.motion_export.soma_bvh import export_nymeria_to_soma_bvh

    return export_nymeria_to_soma_bvh(
        sequence_dir,
        output_path=output_path,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        device=device,
        batch_size=batch_size,
        soma_x_root=soma_x_root,
        smpl_model_path=smpl_model_path,
    )
