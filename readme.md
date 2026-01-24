# install conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate

conda init --all
# conda env create
conda create -n mimic_baseline python=3.11

conda activate mimic_baseline

# isaacsim install by pip

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
或
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# isaaclab install by git

git clone https://github.com/HBUTHUANGPX/IsaacLab_v230.git

cd IsaacLab_v230/ && ./isaaclab.sh --install

# rsl-rl install by git

git clone https://github.com/HBUTHUANGPX/rsl_rl_v320.git

cd rsl_rl_v320/ && pip install -e .

# this repo install by git

git clone https://github.com/HBUTHUANGPX/mimic_baseline.git

cd general_motion_tracker_whole_body_teleoperation/ && pip install -e .

# deploy install by pip
pip install mujoco==3.2.7
pip install onnxruntime==1.22.1
conda install pinocchio -c conda-forge

# train scripts

1. python scripts/rsl_rl/train.py  --task=Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Q1_slowly_walk

# eval scripts

1. python scripts/rsl_rl/play.py --task Tracking-Flat-Q1-v0 --num_envs 2 --domain_randomization


python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/rsl_rl/train.py  --task=Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Q1_slowly_walk  --distributed
https://isaac-sim.github.io/IsaacLab/main/source/features/multi_gpu.html#multi-gpu-training

python scripts/rsl_rl/train.py  --task=Diss-Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Q1_slowly_walk --resume --load_run 2025-12-18_21-23-56_Q1_slowly_walk --checkpoint model_90000.pt

python scripts/csvs_to_npzs.py --input_folder lafan_Q1/lafan_bvh/ --output_folder artifacts/lafan_bvh/ --headless


python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 scripts/rsl_rl/train.py  --task=Pure-Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Pure_Q1_slowly_walk

python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 scripts/rsl_rl/train_multi_teacher_student.py  --task=Diss-Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Q1_LAFAN_walk_Diss  --load_run 2025_12_29_15_00_Pure_Q1 --distributed

- #### teacher policy train command
  单卡训练：
  ```
  python scripts/rsl_rl/train_multi_teacher.py  --task=Pure-Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Pure_Q1_slowly_walk
  ```
  多卡训练：
  ```
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 scripts/rsl_rl/train_multi_teacher.py  --task=Pure-Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Pure_Q1_slowly_walk --distributed
  ```
- #### teacher policy eval command
  - `--other_dirs` 表示 `load_run`下的子文件夹，名字与`motion_file.yaml`中描述的 motion_group name一致
  ```
  python scripts/rsl_rl/play.py --task Pure-Tracking-Flat-Q1-v0 --num_envs 2 --load_run 2026_01_05_22_27_Pure_Q1 --other_dirs run
  ```
  smb://shfile.huaqin.com/机器人软件共享盘