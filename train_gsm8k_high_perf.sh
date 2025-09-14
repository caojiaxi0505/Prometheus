#!/bin/bash
# GSM8K 高性能训练脚本 - 优化GPU利用率

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=0

# 性能监控函数
monitor_gpu() {
    echo "🔍 GPU利用率监控:"
    nvidia-smi dmon -s pucvmet -d 10 &
    GPU_MONITOR_PID=$!
    trap "kill $GPU_MONITOR_PID 2>/dev/null" EXIT
}

# 默认参数
MODEL_NAME=${1:-"Qwen/Qwen3-30B-A3B-Instruct-2507"}
OUTPUT_DIR=${2:-"./gsm8k_high_perf"}
CONFIG_TYPE=${3:-"high_perf"}  # high_perf, gpu_opt, memory_efficient
NUM_EPOCHS=${4:-3}
BATCH_SIZE=${5:-8}
LEARNING_RATE=${6:-5e-4}
LORA_R=${7:-64}
MAX_LENGTH=${8:-1024}

echo "🚀 开始GSM8K高性能微调训练..."
echo "模型: $MODEL_NAME"
echo "输出目录: $OUTPUT_DIR"
echo "配置类型: $CONFIG_TYPE"
echo "训练轮数: $NUM_EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "LoRA秩: $LORA_R"
echo "最大长度: $MAX_LENGTH"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p logs
mkdir -p results

# 开始GPU监控
monitor_gpu

# 运行高性能训练
echo "🏃 开始训练..."
python gsm8k_finetune.py \
    --model $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lora_r $LORA_R \
    --max_length $MAX_LENGTH \
    2>&1 | tee logs/gsm8k_high_perf_training_$(date +%Y%m%d_%H%M%S).log

# 停止GPU监控
kill $GPU_MONITOR_PID 2>/dev/null

echo "✅ 高性能训练完成！模型保存在: $OUTPUT_DIR"

# 自动运行评估
echo "🔍 开始模型评估..."
python gsm8k_inference.py \
    --base_model $MODEL_NAME \
    --finetuned_model $OUTPUT_DIR \
    --mode evaluate \
    --num_samples 200 \
    --output_file "results/gsm8k_high_perf_results_$(date +%Y%m%d_%H%M%S).json" \
    2>&1 | tee logs/gsm8k_high_perf_evaluation_$(date +%Y%m%d_%H%M%S).log

echo "🎉 完整流程完成！"

# 显示性能优化建议
echo ""
echo "💡 性能优化建议:"
echo "1. 使用 nvidia-smi dmon -s pucvmet 实时监控GPU利用率"
echo "2. 调整批次大小和序列长度以最大化GPU内存使用"
echo "3. 使用多GPU训练: export CUDA_VISIBLE_DEVICES=0,1,2,3"
echo "4. 启用混合精度训练: fp16=True"
echo "5. 优化数据加载: dataloader_num_workers=8-16"