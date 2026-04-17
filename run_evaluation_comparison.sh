#!/bin/bash

# 评估脚本：用于在指定评估目录下运行 SurgCLIP 对比
# 使用方法：bash run_evaluation_comparison.sh <eval_directory>

set -e

# 获取评估目录，默认为当前目录
EVAL_DIR="${1:-.}"

# 检查评估目录是否存在
if [ ! -d "$EVAL_DIR" ]; then
    echo "错误：评估目录 $EVAL_DIR 不存在"
    exit 1
fi

# 检查是否存在 summary_all.csv 文件
SUMMARY_FILE="$EVAL_DIR/summary_all.csv"
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "错误：在 $EVAL_DIR 中未找到 summary_all.csv 文件"
    exit 1
fi

# 显示评估目录信息
echo "正在评估目录：$EVAL_DIR"
echo "评估文件：$SUMMARY_FILE"
echo

# 运行对比脚本
echo "正在运行 SurgCLIP 对比脚本..."
echo "=================================="
python3 /mnt/mydisk/CLIP/generate_surgclip_comparison.py --input-dir "$EVAL_DIR" --output-dir "$EVAL_DIR"

echo
echo "评估和对比完成！"

