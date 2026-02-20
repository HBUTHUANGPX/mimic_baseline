### 1. install conda
  ```
mkdir -p ~/miniconda3
  ```
  ```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  ```
  ```
rm ~/miniconda3/miniconda.sh
  ```

  ```
source ~/miniconda3/bin/activate
  ```

  ```
conda init --all
  ```

---

### 2. conda env create
  ```
conda create -n mimic_baseline python=3.11
  ```

  ```
conda activate mimic_baseline
  ```
---

### 3. isaacsim install by pip

  ```
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
  ```
或使用国内pip源头
  ```
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

---

### 4. isaaclab install by git

  ```
git clone https://github.com/HBUTHUANGPX/IsaacLab_v230.git
  ```

  ```
cd IsaacLab_v230/ && ./isaaclab.sh --install
  ```

---

### 5. rsl-rl install by git

  ```
git clone https://github.com/HBUTHUANGPX/rsl_rl_v320.git
  ```

  ```
cd rsl_rl_v320/ && pip install -e .
  ```

---

### 6. this repo install by git

  ```
git clone https://github.com/HBUTHUANGPX/mimic_baseline.git
  ```

  ```
cd general_motion_tracker_whole_body_teleoperation/ && pip install -e .
  ```

---

### 7. deploy install by pip
  ```
pip install mujoco==3.2.7
  ```
  ```
pip install onnxruntime==1.22.1
  ```
  ```
conda install pinocchio -c conda-forge
  ```

---

### base mimic train scripts

  ```
python scripts/rsl_rl/train.py  --task=Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Q1_slowly_walk
  ```

### base mimic eval scripts
  ```
python scripts/rsl_rl/play.py --task Tracking-Flat-Q1-v0 --num_envs 2 --domain_randomization
  ```

---

### CSV to npz 批量转换
第一版，效率较低，逐个文件进行转换
  ```
python scripts/csvs_to_npzs.py --input_folder lafan_Q1/lafan_bvh/ --output_folder artifacts/lafan_bvh/ --headless
  ```
第二版，效率高，并行进行转换，理论上的瓶颈在GPU->CPU数据搬运和保存上
  ```
python scripts/csvs_to_npzs_2.py --input_folder retargeting_data_csv/Q1/100STYLE/ --output_folder artifacts/Q1/100STYLE/ --input_fps 60 --output_fps 50 --headless --preload_csv --async_save_npz
  ```
---
  
### teacher policy train command
  单卡训练：
  ```
  python scripts/rsl_rl/train_multi_teacher.py  --task=Pure-Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Pure_Q1_slowly_walk
  ```
  多卡训练：
  ```
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 scripts/rsl_rl/train_multi_teacher.py  --task=Pure-Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Pure_Q1_slowly_walk --distributed
  ```
  自动化多卡并行单卡训练：
  单卡测试
  ```
  python scripts/rsl_rl/ \
  train_multi_teacher_motion_group_one_by_one_gpu.py \
  --task=Pure-Tracking-Flat-Q1-v0 \
  --headless --logger wandb \
  --log_project_name bydmmc \
  --run_name Q1_lafan \
  --group_name "walk_lafan" \
  --time_stamp "2026_0128_1423" \
  --device=cuda:0
  ```
  自动化脚本：
  ```
  bash train_single_teacher.sh
  ```

### teacher policy eval command
  - `--other_dirs` 表示 `load_run`下的子文件夹，名字与`motion_file.yaml`中描述的 motion_group name一致
  ```  
  python scripts/rsl_rl/play.py --task Pure-Tracking-Flat-Q1-v0 --num_envs 2 --load_run 2026_01_05_22_27_Pure_Q1 --other_dirs run 
  ```

### multi teacher ppo distil
  - 单卡 训练命令
  ```
  python scripts/rsl_rl/train_multi_teacher_student.py  --task=CVAEDissMT-Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Q1_Diss  --load_run 2026_0202_2314_Q1_lafan
  ```

  - 多卡 训练命令
  
  ```
  python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 scripts/rsl_rl/train_multi_teacher_student.py  --task=CVAEDissMT-Tracking-Flat-Q1-v0 --headless --logger wandb --log_project_name bydmmc --run_name Q1_Diss  --load_run 2026_0202_2314_Q1_lafan --distributed 
  ```

  - 测试命令
  ```
  python scripts/rsl_rl/play_multi_teacher_student.py --num_envs 2 --domain_randomization --task=CVAEDissMT-Tracking-Flat-Q1-v0 
  ```







# 报错解决
## 1. 考虑缓存清理：删除Omniverse缓存rm -rf ~/.cache/ov并重试,还不行关掉梯子或者打开梯子

  ```shell
Traceback (most recent call last):
  File "/home/hpx/miniconda3/envs/mimic_baseline/lib/python3.11/site-packages/gymnasium/envs/registration.py", line 734, in make
    env = env_creator(**env_spec_kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hpx/HPX_LOCO_2/mimic_baseline/IsaacLab_v230/source/isaaclab/isaaclab/envs/manager_based_rl_env.py", line 82, in __init__
    super().__init__(cfg=cfg)
  File "/home/hpx/HPX_LOCO_2/mimic_baseline/IsaacLab_v230/source/isaaclab/isaaclab/envs/manager_based_env.py", line 140, in __init__
    self.scene = InteractiveScene(self.cfg.scene)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hpx/HPX_LOCO_2/mimic_baseline/IsaacLab_v230/source/isaaclab/isaaclab/scene/interactive_scene.py", line 180, in __init__
    self._add_entities_from_cfg()
  File "/home/hpx/HPX_LOCO_2/mimic_baseline/IsaacLab_v230/source/isaaclab/isaaclab/scene/interactive_scene.py", line 741, in _add_entities_from_cfg
    self._terrain = asset_cfg.class_type(asset_cfg)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hpx/HPX_LOCO_2/mimic_baseline/IsaacLab_v230/source/isaaclab/isaaclab/terrains/terrain_importer.py", line 109, in __init__
    self.import_ground_plane("terrain")
  File "/home/hpx/HPX_LOCO_2/mimic_baseline/IsaacLab_v230/source/isaaclab/isaaclab/terrains/terrain_importer.py", line 223, in import_ground_plane
    ground_plane_cfg.func(prim_path, ground_plane_cfg)
  File "/home/hpx/HPX_LOCO_2/mimic_baseline/IsaacLab_v230/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py", line 209, in spawn_ground_plane
    bind_physics_material(collision_prim_path, f"{prim_path}/physicsMaterial")
  File "/home/hpx/HPX_LOCO_2/mimic_baseline/IsaacLab_v230/source/isaaclab/isaaclab/sim/utils/prims.py", line 1503, in wrapper
    prim: Usd.Prim = stage.GetPrimAtPath(prim_path)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Boost.Python.ArgumentError: Python argument types in
    Stage.GetPrimAtPath(Stage, NoneType)
did not match C++ signature:
    GetPrimAtPath(pxrInternal_v0_24__pxrReserved__::UsdStage {lvalue}, pxrInternal_v0_24__pxrReserved__::SdfPath path)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/hpx/HPX_LOCO_2/mimic_baseline/IsaacLab_v230/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 101, in hydra_main
    func(env_cfg, agent_cfg, *args, **kwargs)
  File "/home/hpx/HPX_LOCO_2/mimic_baseline/scripts/rsl_rl/train_multi_teacher_motion_group_one_by_one_gpu.py", line 243, in main
    _env = gym.make(
           ^^^^^^^^^
  File "/home/hpx/miniconda3/envs/mimic_baseline/lib/python3.11/site-packages/gymnasium/envs/registration.py", line 746, in make
    raise type(e)(
Boost.Python.ArgumentError: Python argument types in
    Stage.GetPrimAtPath(Stage, NoneType)
did not match C++ signature:
  ```