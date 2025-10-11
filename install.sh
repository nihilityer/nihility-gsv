#!/bin/bash

set -e # 遇到错误时立即退出

# 默认值
USE_HF_MIRROR=false
USE_GH_MIRROR=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ======================
# 解析命令行参数
# ======================
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--hf-mirror)
            USE_HF_MIRROR=true
            shift
            ;;
        -g|--gh-mirror)
            USE_GH_MIRROR=true
            shift
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# ======================
# 配置区 - 请在此处修改为您自己的地址和文件名
# ======================
# LibTorch 下载地址
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip"

# Hugging Face 仓库ID
HF_REPO_ID="nihilityer/nihility-gsv"

# 需要下载的6个模型文件
BASE_FILES=("base/bert.pt" "base/g2p-en.pt" "base/g2p-zh.pt" "base/ssl.pt")
MODEL_DEFAULT_FILES=("default/model.pt" "default/ref.txt" "default/ref.wav")

# GitHub 仓库和要下载的应用资产名
GITHUB_APP_REPO="nihilityer/nihility-gsv"
GITHUB_APP_VERSION="v0.0.1"
GITHUB_APP_ASSET_NAME="nihility-gsv-v0.0.1-x86_64-unknown-linux-gnu"
FINAL_APP_NAME="nihiliy-gsv"

# ======================
# 辅助函数
# ======================

# 下载文件的函数 (如果文件已存在则跳过)
download_file() {
    local url="$1"
    local output="$2"

    # 检查文件是否已存在
    if [ -f "$output" ]; then
        echo "文件已存在，跳过下载: $output"
        return 0
    fi

    echo "正在下载: $url"
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$output" "$url"
    elif command -v curl &> /dev/null; then
        curl -L -o "$output" "$url"
    else
        echo "错误: 未找到 wget 或 curl。请先安装其中一个。"
        exit 1
    fi
}

# 为 Hugging Face URL 应用镜像
apply_hf_mirror() {
    local url="$1"
    if [ "$USE_HF_MIRROR" = true ]; then
        echo "${url/huggingface.co/hf-mirror.com}"
    else
        echo "$url"
    fi
}

# 为 GitHub Release URL 应用镜像
apply_gh_mirror() {
    local url="$1"
    if [ "$USE_GH_MIRROR" = true ]; then
        echo "https://ghproxy.com/$url"
    else
        echo "$url"
    fi
}

# ======================
# 主要任务
# ======================

# 1. 处理 LibTorch
echo "=== 步骤 1: 下载并处理 LibTorch ==="
mkdir -p "$SCRIPT_DIR/lib"
LIBTORCH_ARCHIVE="$SCRIPT_DIR/libtorch-archive.zip"
# 检查压缩包是否已存在
if [ ! -f "$LIBTORCH_ARCHIVE" ]; then
    download_file "$LIBTORCH_URL" "$LIBTORCH_ARCHIVE"
else
    echo "LibTorch 压缩包已存在，跳过下载。"
fi

# 创建临时目录并解压
TEMP_DIR=$(mktemp -d)
unzip -q "$LIBTORCH_ARCHIVE" -d "$TEMP_DIR"

# 复制 libtorch/lib 下的所有文件到 ./lib
cp -r "$TEMP_DIR/libtorch/lib/." "$SCRIPT_DIR/lib/"

# 清理
rm -f "$LIBTORCH_ARCHIVE"
rm -rf "$TEMP_DIR"
echo "LibTorch 处理完成。"

# 2. 从 Hugging Face 下载模型
echo "=== 步骤 2: 从 Hugging Face 下载模型 ==="
# 创建目标目录
mkdir -p "$SCRIPT_DIR/base"
mkdir -p "$SCRIPT_DIR/model/default"

# 下载 base 目录下的文件
for file in "${BASE_FILES[@]}"; do
    HF_URL="https://huggingface.co/$HF_REPO_ID/resolve/main/$file?download=true"
    MIRRORED_HF_URL=$(apply_hf_mirror "$HF_URL")
    download_file "$MIRRORED_HF_URL" "$SCRIPT_DIR/$file"
done

# 下载 model/default 目录下的文件
for file in "${MODEL_DEFAULT_FILES[@]}"; do
    HF_URL="https://huggingface.co/$HF_REPO_ID/resolve/main/$file?download=true"
    MIRRORED_HF_URL=$(apply_hf_mirror "$HF_URL")
    download_file "$MIRRORED_HF_URL" "$SCRIPT_DIR/model/$file"
done
echo "Hugging Face 模型下载完成。"

# 3. 从 GitHub Release 下载应用
echo "=== 步骤 3: 从 GitHub 下载应用 ==="
GITHUB_DOWNLOAD_URL="https://github.com/$GITHUB_APP_REPO/releases/download/$GITHUB_APP_VERSION/$GITHUB_APP_ASSET_NAME"
MIRRORED_GITHUB_URL=$(apply_gh_mirror "$GITHUB_DOWNLOAD_URL")
APP_PATH="$SCRIPT_DIR/$FINAL_APP_NAME"

# 下载应用
download_file "$MIRRORED_GITHUB_URL" "$APP_PATH"

# 赋予执行权限
chmod +x "$APP_PATH"
echo "GitHub 应用下载并授权完成。"

echo "所有任务已完成！"
