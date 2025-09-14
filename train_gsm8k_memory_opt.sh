#!/bin/bash
# GSM8K 内存优化训练脚本 - 适用于有限GPU内存

# 设置环境变量 - 使用两张GPU
export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false  # 禁用并行避免死锁
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true  # 禁用wandb

echo "🧠 开始GSM8K内存优化训练..."
echo "模型: $MODEL_NAME"
echo "输出目录: $OUTPUT_DIR"

# 默认参数 - 双GPU内存优化配置
MODEL_NAME=${1:-"Qwen/Qwen3-30B-A3B-Instruct-2507"}
OUTPUT_DIR=${2:-"./gsm8k_memory_opt"}
NUM_EPOCHS=${3:-1}  # 减少训练轮数
BATCH_SIZE=${4:-1}  # 最小批次大小
LEARNING_RATE=${5:-2e-4}  # 稍微降低学习率
LORA_R=${6:-8}  # 更小的LoRA秩
MAX_LENGTH=${7:-256}  # 更短的序列长度

echo "📋 内存优化配置:"
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
mkdir -p results

# 清理GPU内存
python -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
print('GPU内存已清理')
"

# 运行内存优化训练
echo "🏃 开始内存优化训练..."
WANDB_DISABLED=true python gsm8k_finetune_final.py \
    --model $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lora_r $LORA_R \
    --max_length $MAX_LENGTH \
    2>&1 | tee logs/gsm8k_memory_opt_training_$(date +%Y%m%d_%H%M%S).log

echo "✅ 内存优化训练完成！模型保存在: $OUTPUT_DIR"

# 可选：运行轻量级评估
echo "🔍 开始轻量级评估..."
python gsm8k_inference.py \
    --base_model $MODEL_NAME \
    --finetuned_model $OUTPUT_DIR \
    --mode evaluate \
    --num_samples 50 \
    --output_file "results/gsm8k_memory_opt_results_$(date +%Y%m%d_%H%M%S).json" \
    2>&1 | tee logs/gsm8k_memory_opt_evaluation_$(date +%Y%m%d_%H%M%S).log

echo "🎉 内存优化流程完成！"

echo ""
echo "💡 内存优化建议:"
echo "1. 使用较小的批次大小 (1-2)"
echo "2. 使用较小的LoRA秩 (16-32)"
echo "3. 减少序列长度 (256-512)"
echo "4. 启用梯度检查点 (gradient_checkpointing=True)"
echo "5. 使用FP16训练 (fp16=True)"
echo "6. 设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "7. 定期清理GPU内存"
echo ""
echo "🔧 如果仍然内存不足，请尝试:"
echo "- 减少max_length到256"
echo "- 使用更小的lora_r (8-16)"
echo "- 禁用多线程数据加载"