#!/bin/bash
# GSM8K 模型评估脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com

# 默认参数
BASE_MODEL=${1:-"Qwen/Qwen3-30B-A3B-Instruct-2507"}
FINETUNED_MODEL=${2:-"./gsm8k_finetuned"}
MODE=${3:-"evaluate"}  # evaluate 或 compare
NUM_SAMPLES=${4:-100}
OUTPUT_FILE=${5:-"gsm8k_results.json"}

echo "开始GSM8K模型评估..."
echo "基础模型: $BASE_MODEL"
echo "微调模型: $FINETUNED_MODEL"
echo "评估模式: $MODE"
echo "样本数量: $NUM_SAMPLES"
echo "输出文件: $OUTPUT_FILE"

# 创建输出目录
mkdir -p results
mkdir -p logs

# 运行评估
python gsm8k_inference.py \
    --base_model $BASE_MODEL \
    --finetuned_model $FINETUNED_MODEL \
    --mode $MODE \
    --num_samples $NUM_SAMPLES \
    --output_file "results/$OUTPUT_FILE" \
    2>&1 | tee logs/gsm8k_evaluation_$(date +%Y%m%d_%H%M%S).log

echo "评估完成！结果保存在: results/$OUTPUT_FILE"

# 如果结果文件存在，显示摘要
if [ -f "results/$OUTPUT_FILE" ]; then
    echo "评估结果摘要:"
    if [ "$MODE" == "evaluate" ]; then
        python -c "
import json
with open('results/$OUTPUT_FILE', 'r') as f:
    results = json.load(f)
print(f'总样本数: {results[\"total_samples\"]}')
print(f'正确预测数: {results[\"correct_predictions\"]}')
print(f'准确率: {results[\"accuracy\"]:.2%}')
"
    elif [ "$MODE" == "compare" ]; then
        python -c "
import json
with open('results/$OUTPUT_FILE', 'r') as f:
    results = json.load(f)
print(f'总样本数: {results[\"total_samples\"]}')
print(f'基础模型准确率: {results[\"base_model_accuracy\"]:.2%}')
print(f'微调模型准确率: {results[\"finetuned_model_accuracy\"]:.2%}')
print(f'改进样本数: {results[\"improvements\"]}')
"
    fi
fi