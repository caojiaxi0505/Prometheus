#!/bin/bash
# GSM8K 完整数据集训练脚本
# 适用于双GPU环境，使用完整7473个训练样本

# 设置环境变量 - 双GPU优化
export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true

echo "🎯 开始GSM8K完整数据集训练..."
echo "模型: Qwen/Qwen3-30B-A3B-Instruct-2507"
echo "输出目录: ./gsm8k_full_finetuned"

# 创建输出目录
mkdir -p ./gsm8k_full_finetuned
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

echo "🏃 开始完整数据集训练..."
WANDB_DISABLED=true python gsm8k_finetune.py \
    --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --output_dir "./gsm8k_full_finetuned" \
    --num_epochs 3 \
    --batch_size 1 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --max_length 512 \
    2>&1 | tee logs/gsm8k_full_training_$(date +%Y%m%d_%H%M%S).log

echo "✅ 完整数据集训练完成！模型保存在: ./gsm8k_full_finetuned"

# 可选：运行完整评估
echo "🔍 开始完整评估..."
python gsm8k_inference.py \
    --base_model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --finetuned_model "./gsm8k_full_finetuned" \
    --mode evaluate \
    --output_file "results/gsm8k_full_results_$(date +%Y%m%d_%H%M%S).json" \
    2>&1 | tee logs/gsm8k_full_evaluation_$(date +%Y%m%d_%H%M%S).log

echo "🎉 完整数据集训练流程完成！"

echo ""
echo "📊 训练统计:"
echo "- 训练样本: 7473个 (完整GSM8K训练集)"
echo "- 验证样本: 748个 (GSM8K测试集)"
echo "- 模型参数: 30.5B (基础) + 6.7M (LoRA可训练)"
echo "- 训练配置: 双GPU, FP16, 梯度检查点"