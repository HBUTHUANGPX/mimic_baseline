# mocap_motion_vae

Data loading skeleton for AMASS SMPL-X npz files.

Structure:
- `src/mocap_motion_vae/data/amass_smplx.py`: SMPL-X parser and bank builder.
- `src/mocap_motion_vae/data/bank.py`: MotionBank / MotionView and feature spec.
- `src/mocap_motion_vae/data/dataset.py`: windowed dataset for training.

Minimal usage:
```python
from mocap_motion_vae.data import (
    SMPLXFieldSpec,
    SMPLXClipParser,
    build_amass_smplx_bank,
    MotionWindowDataset,
    FeatureSpec,
)

files = ["/path/to/amass/SMPLX/file1.npz", "/path/to/file2.npz"]

# 默认包含 joints（需要配置 SMPL-X 模型路径）
bank = build_amass_smplx_bank(
    files,
    SMPLXFieldSpec(),
    smplx_model_path="/path/to/SMPLX/models",
)

spec = FeatureSpec(
    inputs=("pose_body", "root_orient", "trans", "joints"),
    targets=("pose_body",),
)
dataset = MotionWindowDataset(bank, window=120, stride=30, feature_spec=spec)
```

SMPL-X 模型路径配置:
```bash
# 方式1：环境变量
export SMPLX_MODEL_PATH=/path/to/SMPLX/models

# 方式2：代码中显式传入
bank = build_amass_smplx_bank(files, SMPLXFieldSpec(), smplx_model_path="/path/to/SMPLX/models")
```

Tests:
```bash
PYTHONPATH=mocap_motion_vae/src python -m pytest mocap_motion_vae/tests
```
