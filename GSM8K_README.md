# GSM8K 数学推理微调指南

本指南介绍如何使用Qwen3模型在GSM8K数据集上进行数学推理任务的微调。

## 功能特性

- 🎯 **专门优化**: 针对数学推理任务特别设计
- 🔧 **LoRA微调**: 使用低秩适应技术，节省计算资源
- 📊 **完整评估**: 提供详细的性能评估和对比功能
- 🚀 **多模型支持**: 支持所有Qwen3系列模型
- ⚙️ **灵活配置**: 提供多种预设配置适应不同硬件环境

## 环境要求

### 硬件要求
- **GPU**: 至少24GB显存（推荐48GB+）
- **内存**: 至少64GB系统内存
- **存储**: 至少100GB可用空间

### 软件依赖
```bash
# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 快速测试
```bash
# 运行快速测试（验证代码正确性）
bash train_gsm8k.sh Qwen/Qwen3-30B-A3B-Instruct-2507 ./gsm8k_quick_test 1 2 3e-4 16 512
```

### 2. 标准训练
```bash
# 使用默认参数进行训练
bash train_gsm8k.sh

# 或自定义参数
bash train_gsm8k.sh Qwen/Qwen3-30B-A3B-Instruct-2507 ./gsm8k_standard 3 4 2e-4 64 1024
```

### 3. 高性能训练（推荐）
```bash
# 高性能训练 - 优化GPU利用率
bash train_gsm8k_high_perf.sh

# 或自定义高性能参数
bash train_gsm8k_high_perf.sh Qwen/Qwen3-30B-A3B-Instruct-2507 ./gsm8k_high_perf high_perf 3 8 5e-4 64 1024
```

### 4. 内存优化训练（适用于GPU内存有限的情况）
```bash
# 内存优化训练 - 适用于44GB GPU内存
bash train_gsm8k_memory_opt.sh

# 或自定义内存优化参数
bash train_gsm8k_memory_opt.sh Qwen/Qwen3-30B-A3B-Instruct-2507 ./gsm8k_memory_opt 3 1 3e-4 16 512
```

### 5. 模型评估
```bash
# 评估微调模型
bash evaluate_gsm8k.sh Qwen/Qwen3-30B-A3B-Instruct-2507 ./gsm8k_standard evaluate 100

# 比较基础模型和微调模型
bash evaluate_gsm8k.sh Qwen/Qwen3-30B-A3B-Instruct-2507 ./gsm8k_standard compare 100
```

## 详细使用指南

### 数据格式
GSM8K数据集包含小学数学应用题，格式如下：
```json
{
  "question": "小明有15个苹果，他给了小红6个，又买了8个，现在小明有多少个苹果？",
  "answer": "小明原来有15个苹果。给了小红6个后，剩下15 - 6 = 9个。又买了8个，现在有9 + 8 = 17个苹果。#### 17"
}
```

### 训练流程

#### 步骤1: 数据预处理
```python
from gsm8k_finetune import GSM8KDataProcessor, GSM8KConfig
from transformers import AutoTokenizer

config = GSM8KConfig()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
processor = GSM8KDataProcessor(tokenizer, config)

# 加载和处理数据
train_dataset, eval_dataset = processor.load_and_process_dataset()
```

#### 步骤2: 模型微调
```python
from gsm8k_finetune import GSM8KTrainer

trainer = GSM8KTrainer(config)
trainer.setup_model_and_tokenizer()
trainer.setup_lora()
trainer.train()
```

#### 步骤3: 模型推理
```python
from gsm8k_inference import GSM8KEvaluator

evaluator = GSM8KEvaluator(
    base_model_path="Qwen/Qwen3-30B-A3B-Instruct-2507",
    finetuned_model_path="./gsm8k_finetuned"
)

# 单个问题求解
question = "一个长方形的长是12厘米，宽是8厘米，它的面积是多少平方厘米？"
response = evaluator.generate_response(question)
print(response)
```

### 配置选项

#### 预设配置
```python
from gsm8k_config import get_config, list_experiments

# 查看可用配置
list_experiments()

# 获取配置
config = get_config("Qwen/Qwen3-30B-A3B-Instruct-2507", "standard")
```

#### 配置类型
- **quick**: 快速测试配置
- **standard**: 标准训练配置（推荐）
- **high_perf**: 高性能配置
- **low_resource**: 低资源配置

### 高级用法

#### 自定义训练参数
```python
from gsm8k_finetune import GSM8KConfig

config = GSM8KConfig(
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
    num_epochs=5,
    learning_rate=1e-4,
    lora_r=128,
    lora_alpha=256,
    max_length=2048,
    output_dir="./custom_finetuned"
)
```

#### 批量推理
```python
from gsm8k_examples import batch_solve_problems

questions = [
    "问题1...",
    "问题2...",
    "问题3..."
]

results = batch_solve_problems(model, tokenizer, questions)
```

#### 交互模式
```python
# 运行交互模式
python gsm8k_examples.py interactive
```

## 性能优化建议

### 1. 内存优化
- 使用4-bit量化：`load_in_4bit=True`
- 减少批次大小：`per_device_train_batch_size=1`
- 增加梯度累积：`gradient_accumulation_steps=8`

### 2. 训练优化
- 使用适当的学习率：2e-4 到 5e-4
- 调整LoRA秩：64-128 通常效果良好
- 适当的数据预处理：确保数据质量
- **GPU利用率优化**：
  - 增大批次大小（8-16）
  - 增加数据加载线程（8-16）
  - 启用内存锁定（dataloader_pin_memory=True）
  - 使用高性能训练脚本：`bash train_gsm8k_high_perf.sh`

### 3. 推理优化
- 使用适当的温度参数：0.7-0.9
- 限制生成长度：避免过长的生成
- 批量推理：提高推理效率

### 4. GPU性能监控
```bash
# 实时监控GPU利用率
nvidia-smi dmon -s pucvmet

# 查看GPU内存使用
nvidia-smi

# 使用高性能配置
python gsm8k_performance_config.py
```

## 故障排除

### 常见问题

#### 1. 内存不足
```bash
# 使用内存优化脚本
bash train_gsm8k_memory_opt.sh

# 或手动设置内存优化参数
python gsm8k_finetune.py --batch_size 1 --lora_r 16 --max_length 512

# 设置环境变量优化内存分配
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 如果仍然内存不足，尝试：
# - 减少max_length到256
# - 使用更小的lora_r (8-16)
# - 禁用多线程数据加载
# - 使用单GPU训练
```

#### 2. 训练速度慢
```bash
# 增加批次大小（如果内存允许）
python gsm8k_finetune.py --batch_size 8 --gradient_accumulation_steps 2
```

#### 3. 模型加载失败
```bash
# 设置HF镜像
export HF_ENDPOINT=https://hf-mirror.com
```

#### 4. 评估结果不理想
- 检查训练数据质量
- 增加训练轮数
- 调整学习率和LoRA参数
- 使用更大的模型

## 实验结果

### 预期性能
在GSM8K测试集上的预期性能：

| 模型 | 基础准确率 | 微调后准确率 | 提升 |
|------|------------|--------------|------|
| Qwen3-30B-A3B | ~45% | ~65-75% | +20-30% |
| Qwen3-Next-80B | ~55% | ~75-85% | +20-30% |

### 训练时间估算
- **Qwen3-30B-A3B**: 约4-6小时（2x A100 80GB）
- **Qwen3-Next-80B**: 约8-12小时（2x A100 80GB）

## 最佳实践

### 1. 数据准备
- 确保数据质量
- 适当的预处理
- 合理的训练/验证划分

### 2. 模型选择
- 根据任务复杂度选择模型
- 考虑硬件限制
- 平衡性能和效率

### 3. 训练策略
- 从较小的学习率开始
- 逐步调整超参数
- 监控训练过程

### 4. 评估方法
- 使用多个评估指标
- 进行人工评估
- 对比不同模型版本

## 扩展功能

### 1. 自定义数据集
可以适配其他数学推理数据集：
- MATH数据集
- MathQA
- AQuA-RAT

### 2. 多任务学习
结合其他任务：
- 数学问题生成
- 解题步骤评分
- 数学概念解释

### 3. 模型融合
- 集成多个微调模型
- 知识蒸馏
- 模型压缩

## 更新日志

- v1.0.0: 初始版本，支持GSM8K微调
- v1.1.0: 添加多模型支持和配置管理
- v1.2.0: 优化内存使用和训练效率

## 贡献指南

欢迎提交Issue和Pull Request来改进代码。

## 许可证

本项目采用MIT许可证，详见LICENSE文件。