#!/bin/bash
# GSM8K 微调训练脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com

# 默认参数
MODEL_NAME=${1:-"Qwen/Qwen3-30B-A3B-Instruct-2507"}
OUTPUT_DIR=${2:-"./gsm8k_finetuned"}
NUM_EPOCHS=${3:-3}
BATCH_SIZE=${4:-4}
LEARNING_RATE=${5:-2e-4}
LORA_R=${6:-64}
MAX_LENGTH=${7:-1024}

echo "开始GSM8K微调训练..."
echo "模型: $MODEL_NAME"
echo "输出目录: $OUTPUT_DIR"
echo "训练轮数: $NUM_EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "LoRA秩: $LORA_R"
echo "最大长度: $MAX_LENGTH"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p logs

# 运行训练
python gsm8k_finetune.py \
    --model $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lora_r $LORA_R \
    --max_length $MAX_LENGTH \
    2>&1 | tee logs/gsm8k_training_$(date +%Y%m%d_%H%M%S).log

echo "训练完成！模型保存在: $OUTPUT_DIR"