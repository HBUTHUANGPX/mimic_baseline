# conda env create
conda create -n mimic_baseline python=3.11

conda activate mimic_baseline

# isaacsim install by pip

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

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