#!/bin/bash

# ================= 配置区域 =================
# 项目根目录 (默认取当前运行目录)
BASE_DIR=$(pwd)

# [更新] Python 脚本的具体路径
SCRIPT_CLS="${BASE_DIR}/classification/predict_single_image.py"
SCRIPT_SEG="${BASE_DIR}/segmentation/predict_single_image.py"
SCRIPT_LLM="${BASE_DIR}/rag/deepseek_pdf.py"

# 输出目录 (对应 Python 代码中写死的路径)
# 1. 分类结果目录
OUT_CLS_DIR="/commondocument/group3/MuRAG-CAD/classification/single_results"
# 2. 分割结果目录 (注意是 signal_results)
OUT_SEG_DIR="/commondocument/group3/MuRAG-CAD/segmentation/signal_results"

# ===========================================

# 1. 检查输入参数
if [ -z "$1" ]; then
    echo "❌ 错误: 未提供图片路径。"
    echo "用法: ./run_pipeline.sh <path_to_image>"
    echo "示例: ./run_pipeline.sh \"/commondocument/group3/test_images/case1.png\""
    exit 1
fi

IMAGE_PATH="$1"

# 检查图片是否存在
if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ 错误: 图片文件不存在 -> $IMAGE_PATH"
    exit 1
fi

# 检查 Python 脚本是否存在 (防止路径错误)
if [ ! -f "$SCRIPT_CLS" ] || [ ! -f "$SCRIPT_SEG" ] || [ ! -f "$SCRIPT_LLM" ]; then
    echo "❌ 错误: 找不到 Python 脚本文件。"
    echo "请确保你在包含 'MuRAG-CAD' 文件夹的根目录下运行此脚本。"
    echo "检查路径: $SCRIPT_CLS"
    exit 1
fi

echo "========================================================"
echo "🚀 开始全流程分析: $(basename "$IMAGE_PATH")"
echo "========================================================"

# 2. 运行 Step 1: 分类 (Classification)
echo ""
echo "---------- [1/3] 正在运行分类模型 (DenseNet121) ----------"
# 注意：你的分类代码中 argparse 默认参数名为 --image
python "$SCRIPT_CLS" --image "$IMAGE_PATH"

if [ $? -ne 0 ]; then
    echo "❌ 分类步骤失败，流程终止。"
    exit 1
fi

# 3. 运行 Step 2: 分割 (Segmentation)
echo ""
echo "---------- [2/3] 正在运行分割模型 (Unet++) ----------"
python "$SCRIPT_SEG" --image "$IMAGE_PATH"

if [ $? -ne 0 ]; then
    echo "❌ 分割步骤失败，流程终止。"
    exit 1
fi

# 4. 计算生成的报告路径
# Python 逻辑是 os.path.basename(path).split('.')[0]
# Shell 逻辑：提取文件名 -> 截取第一个点之前的部分
FILENAME=$(basename "$IMAGE_PATH")
BASENAME=$(echo "$FILENAME" | cut -d. -f1)

# 构建预期的报告文件路径
CLS_REPORT="${OUT_CLS_DIR}/${BASENAME}_report.txt"
SEG_REPORT="${OUT_SEG_DIR}/${BASENAME}_report.txt"

# 检查报告是否生成成功
if [ ! -f "$CLS_REPORT" ] || [ ! -f "$SEG_REPORT" ]; then
    echo ""
    echo "❌ 错误: 未找到生成的报告文件。"
    echo "预期分类报告: $CLS_REPORT"
    echo "预期分割报告: $SEG_REPORT"
    echo "可能原因：Python代码输出路径与脚本配置不一致，或文件名处理逻辑有差异。"
    exit 1
fi

# 5. 运行 Step 3: LLM 综合分析
echo ""
echo "---------- [3/3] 正在生成 AI 诊断报告 (DeepSeek) ----------"
# 注意：你的 rag 代码中 argparse 参数名为 --cls 和 --det
python "$SCRIPT_LLM" --cls "$CLS_REPORT" --det "$SEG_REPORT"

echo ""
echo "========================================================"
echo "✅ 全流程分析完成！"
echo "========================================================"
