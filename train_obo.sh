#!/bin/bash

# 脚本名称：train_rsl_rl.sh
# 用法：./train_rsl_rl.sh [--motion <value>] [--timestamp <value>] [--gpu <id>] [--help]
# 示例：
#   ./train_rsl_rl.sh --motion abc --timestamp 20250120 --gpu 0
#   ./train_rsl_rl.sh  # 使用默认值

# 默认值
MOTION="default_motion"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")  # 当前时间戳
GPU="0"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --motion)
            MOTION="$2"
            shift 2
            ;;
        --timestamp)
            TIMESTAMP="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [--motion <value>] [--timestamp <value>] [--gpu <id>]"
            echo "  --motion     : 设置 motion 值（对应 --group_name \"\$motion\" 中的字面值）"
            echo "  --timestamp  : 设置时间戳"
            echo "  --gpu        : 设置 GPU ID"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 注意：原命令中 --group_name "\$motion" 会将字面字符串 "$motion" 传递给参数
# 如果您希望传递 motion 变量的实际值，请将下面一行改为 --group_name "$MOTION"

CUDA_VISIBLE_DEVICES=$GPU python scripts/rsl_rl/train_multi_teacher_motion_group_one_by_one_gpu.py \
    --task=Pure-Tracking-Flat-Q1-v0 \
    --headless \
    --logger wandb \
    --log_project_name bydmmc \
    --run_name Q1_pure \
    --group_name "$MOTION" \
    --time_stamp "$TIMESTAMP" \
    # --device=cuda:$GPU