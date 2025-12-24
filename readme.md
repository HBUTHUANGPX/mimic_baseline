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

# deploy install by pip
pip install mujoco==3.2.7
pip install onnxruntime==1.22.1
conda install pinocchio -c conda-forge


python scripts/rsl_rl/train.py  --task=Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Q1_slowly_walk

python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/rsl_rl/train.py  --task=Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Q1_slowly_walk  --distributed
https://isaac-sim.github.io/IsaacLab/main/source/features/multi_gpu.html#multi-gpu-training

python scripts/rsl_rl/train.py  --task=Diss-Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Q1_slowly_walk --resume --load_run 2025-12-18_21-23-56_Q1_slowly_walk --checkpoint model_90000.pt