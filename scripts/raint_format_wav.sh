#!/usr/bin/env bash

########## raint 数据集音频预处理脚本 ##########
# 目标：将所有原始 wav 统一为单声道 / 24kHz / s16 格式
# 输入目录：./data/raw/*.wav
# 输出目录：./data/processed/raint/wav/
# bash ./scripts/raint_format_wav.sh

# 失败即退出，未定义变量报错，管道错误冒泡
set -euo pipefail

# 确保输出目录存在
mkdir -p ./data/processed/raint/wav

# 若没有匹配到文件则不进入循环（bash 行为，避免把字面量模式当作文件）
shopt -s nullglob

# 将所有原始 wav 统一为：单声道 / 24kHz / s16
for f in ./data/raw/*.wav; do
    base=$(basename "$f")
    out="./data/processed/raint/wav/$base"
    echo "Processing $base -> $out"
    ffmpeg -y -i "$f" \
        -ac 1 \
        -ar 24000 \
        -sample_fmt s16 \
        "$out"
done

echo "All files processed."