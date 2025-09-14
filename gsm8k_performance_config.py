#!/usr/bin/env python3
"""
GSM8K 性能优化配置
专门用于提高GPU利用率和训练速度
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class HighPerformanceConfig:
    """高性能训练配置 - 优化GPU利用率"""
    
    # 模型配置
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    # 数据配置
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    max_length: int = 1024
    train_split: float = 0.9
    
    # LoRA配置 - 平衡性能和内存
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05  # 降低dropout以提高速度
    target_modules: List[str] = None
    
    # 训练配置 - 优化GPU利用率，同时控制内存使用
    num_epochs: int = 3
    per_device_train_batch_size: int = 2  # 小批次大小，避免内存溢出
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # 使用梯度累积来模拟大批次
    learning_rate: float = 5e-4  # 稍高的学习率
    warmup_steps: int = 50  # 减少warmup步数
    max_grad_norm: float = 1.0
    
    # 生成配置
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # 输出配置
    output_dir: str = "./gsm8k_high_perf"
    logging_dir: str = "./logs"
    save_total_limit: int = 2
    
    # 性能优化配置
    fp16: bool = True
    dataloader_num_workers: int = 4  # 适中的数据加载线程
    dataloader_pin_memory: bool = True  # 启用内存锁定
    gradient_checkpointing: bool = True  # 启用梯度检查点以节省内存
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class GPUOptimizedConfig:
    """GPU优化配置 - 针对高GPU利用率"""
    
    # 模型配置
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    # 数据配置
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    max_length: int = 768  # 稍短的序列长度以提高吞吐量
    train_split: float = 0.9
    
    # LoRA配置
    lora_r: int = 32  # 较小的秩以提高速度
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # 训练配置 - 平衡GPU利用率和内存使用
    num_epochs: int = 3
    per_device_train_batch_size: int = 1  # 最小批次大小，避免内存溢出
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16  # 使用大量梯度累积来模拟大批次
    learning_rate: float = 8e-4  # 更高的学习率
    warmup_steps: int = 30
    max_grad_norm: float = 1.0
    
    # 系统配置
    max_new_tokens: int = 384  # 减少生成长度
    temperature: float = 0.7
    top_p: float = 0.9
    
    # 输出配置
    output_dir: str = "./gsm8k_gpu_opt"
    logging_dir: str = "./logs"
    save_total_limit: int = 1  # 减少保存频率
    
    # 性能配置
    fp16: bool = True
    dataloader_num_workers: int = 4  # 适中的数据加载线程
    dataloader_pin_memory: bool = True
    gradient_checkpointing: bool = True  # 启用梯度检查点以节省内存
    
    def __post_init__(self):
        if self.target_modules is None:
            # 只训练关键模块以提高速度
            self.target_modules = ["q_proj", "v_proj", "o_proj"]

@dataclass
class MemoryEfficientConfig:
    """内存高效配置 - 在有限内存下最大化性能"""
    
    # 模型配置
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    # 数据配置
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    max_length: int = 512  # 较短的序列长度
    train_split: float = 0.9
    
    # LoRA配置
    lora_r: int = 16  # 最小的秩
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # 训练配置
    num_epochs: int = 3
    per_device_train_batch_size: int = 4  # 适中的批次大小
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # 使用梯度累积
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # 输出配置
    output_dir: str = "./gsm8k_memory_efficient"
    logging_dir: str = "./logs"
    save_total_limit: int = 2
    
    # 内存优化配置
    fp16: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    gradient_checkpointing: bool = True  # 启用梯度检查点以节省内存

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

# 性能优化训练参数
PERFORMANCE_TRAINING_ARGS = {
    "logging_steps": 1,  # 更频繁的日志记录以监控GPU利用率
    "eval_steps": 25,    # 更频繁的评估
    "save_steps": 100,   # 更频繁的保存
    "save_total_limit": 1,  # 减少磁盘I/O
    "load_best_model_at_end": False,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "report_to": "tensorboard",
    "fp16": True,
    "dataloader_num_workers": 8,
    "remove_unused_columns": False,
    "dataloader_pin_memory": True,
    "gradient_checkpointing": False,
}

def get_performance_config(config_type: str = "high_perf", model_name: str = None):
    """获取性能优化配置"""
    configs = {
        "high_perf": HighPerformanceConfig,
        "gpu_opt": GPUOptimizedConfig,
        "memory_efficient": MemoryEfficientConfig,
    }
    
    config_class = configs.get(config_type, HighPerformanceConfig)
    
    if model_name:
        return config_class(model_name=model_name)
    else:
        return config_class()

def print_performance_tips():
    """打印性能优化建议"""
    print("🔧 GPU性能优化建议:")
    print("1. 增大批次大小 (per_device_train_batch_size)")
    print("2. 增加数据加载线程 (dataloader_num_workers)")
    print("3. 启用内存锁定 (dataloader_pin_memory=True)")
    print("4. 使用FP16训练 (fp16=True)")
    print("5. 减少序列长度 (max_length)")
    print("6. 禁用梯度检查点 (gradient_checkpointing=False)")
    print("7. 减少保存频率 (save_steps, save_total_limit)")
    print("8. 使用较小的LoRA秩 (lora_r)")
    print("\n📊 监控GPU利用率:")
    print("nvidia-smi dmon -s pucvmet")

if __name__ == "__main__":
    print_performance_tips()
    
    print("\n📋 可用配置:")
    for name, config_class in [
        ("high_perf", HighPerformanceConfig),
        ("gpu_opt", GPUOptimizedConfig),
        ("memory_efficient", MemoryEfficientConfig),
    ]:
        config = config_class()
        print(f"\n{name}:")
        print(f"  批次大小: {config.per_device_train_batch_size}")
        print(f"  序列长度: {config.max_length}")
        print(f"  LoRA秩: {config.lora_r}")
        print(f"  数据加载线程: {config.dataloader_num_workers}")