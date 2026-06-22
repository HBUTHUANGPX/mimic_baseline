#!/bin/bash

# 生成时间戳（格式：YYYY_MMDD_HHMM）
timestamp=$(date +"%Y_%m%d_%H%M")
echo "生成的时间戳: $timestamp"

# YAML 文件路径
yaml_file="scripts/rsl_rl/motion_file.yaml"

# 检查 yq 是否可用
if ! command -v yq >/dev/null 2>&1; then
    echo "错误: 未找到 yq 命令，请先安装 yq :sudo snap install yq" >&2
    exit 1
fi

# 读取 motion_group 下的所有键到数组 motions
mapfile -t motions < <(yq -r '.motion_group | keys | .[]' "$yaml_file")

# 检查是否成功读取到键
if [[ ${#motions[@]} -eq 0 ]]; then
    echo "错误: 未从 $yaml_file 中读取到任何 motion 键，请检查 YAML 文件结构" >&2
    exit 1
fi

total=${#motions[@]}
echo "检测到的 motion 键数量: $total"
echo "键列表: ${motions[*]}"

# GPU 配置
gpu_num=8
if (( total == 0 )); then
    echo "无任务可执行，脚本结束。"
    exit 0
fi

# 计算每个 GPU 平均任务数（向上取整，确保均衡）
per_gpu=$(( (total + gpu_num - 1) / gpu_num ))

# 工作目录（根据您脚本中的 cd 路径设置）
work_dir="/home/jerry_huang/HPX_Loco/mimic_baseline"

active_gpus=0

# 遍历每个 GPU，分配任务并启动终端
for ((gpu=0; gpu<gpu_num; gpu++)); do
    start=$((gpu * per_gpu))
    if (( start >= total )); then
        break
    fi
    end=$((start + per_gpu))
    if (( end > total )); then
        end=$total
    fi
    chunk_size=$((end - start))
    if (( chunk_size == 0 )); then
        continue
    fi

    # 提取该 GPU 的任务子数组
    mapfile -t chunk -O 0 < <(printf '%s\n' "${motions[@]:start:chunk_size}")

    # 安全转义数组元素，用于内联传递
    chunk_escaped=$(printf '%q ' "${chunk[@]}")
    chunk_escaped=${chunk_escaped% }  # 去除末尾空格

    echo "为 GPU $gpu 分配 ${#chunk[@]} 个任务: ${chunk[*]}"

    gnome-terminal \
        --working-directory="$work_dir" \
        --title="GPU $gpu - Training (${chunk[*]})" \
        -- bash -c "
            motions=($chunk_escaped) &&
            for motion in \"\${motions[@]}\"; do
                source /home/hpx/miniconda3/etc/profile.d/conda.sh &&
                conda activate mimic_baseline &&
                echo \"=== 开始训练 motion: \$motion 于 cuda:$gpu ===\" &&
                bash train_obo.sh --motion \"\$motion\" --timestamp \"$timestamp\" --gpu $gpu ;
                echo \"=== motion \$motion 训练完成 ===\"
            done &&
            echo \"GPU $gpu 所有任务已完成，按 Enter 退出终端...\" &&
            read -r ;
            exec bash
        " &
    sleep 15
    ((active_gpus++))
done

echo "已启动 $active_gpus 个训练终端（每个对应一个 GPU，共覆盖 $total 个任务）。"