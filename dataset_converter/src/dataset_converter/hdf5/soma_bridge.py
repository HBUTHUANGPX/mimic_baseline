from __future__ import annotations

from pathlib import Path

from dataset_converter.common.paths import ensure_workspace_on_sys_path


def export_segmented_soma_bvh_bridge(
    hdf5_path: str | Path,
    *,
    soma_bvh_output_dir: str | Path,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda",
    batch_size: int | None = None,
    soma_x_root: str | Path,
    smpl_model_path: str | Path | None = None,
    filename_prefix: str = "annotation",
) -> list[Path]:
    ensure_workspace_on_sys_path()
    from hdf5_parse.motion_export.segmented import export_segmented_soma_bvh

    return export_segmented_soma_bvh(
        hdf5_path=hdf5_path,
        soma_bvh_output_dir=soma_bvh_output_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        device=device,
        batch_size=batch_size,
        soma_x_root=soma_x_root,
        smpl_model_path=smpl_model_path,
        filename_prefix=filename_prefix,
    )
