#!/usr/bin/env python3
"""
GSM8K 微调配置文件
包含不同规模的训练配置
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BaseConfig:
    """基础配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    # 数据配置
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    max_length: int = 1024
    train_split: float = 0.9
    
    # LoRA配置
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # 训练配置
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # 生成配置
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # 输出配置
    output_dir: str = "./gsm8k_finetuned"
    logging_dir: str = "./logs"
    save_total_limit: int = 3
    
    # 硬件配置
    fp16: bool = True
    dataloader_num_workers: int = 4

@dataclass
class QuickTestConfig(BaseConfig):
    """快速测试配置 - 用于验证代码"""
    num_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    max_length: int = 512
    lora_r: int = 16
    output_dir: str = "./gsm8k_quick_test"

@dataclass
class StandardConfig(BaseConfig):
    """标准训练配置"""
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 64
    output_dir: str = "./gsm8k_standard"

@dataclass
class HighPerformanceConfig(BaseConfig):
    """高性能训练配置"""
    num_epochs: int = 5
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    lora_r: int = 128
    lora_alpha: int = 256
    max_length: int = 2048
    output_dir: str = "./gsm8k_high_perf"

@dataclass
class LowResourceConfig(BaseConfig):
    """低资源配置 - 适用于GPU内存有限的情况"""
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    lora_r: int = 32
    lora_alpha: int = 64
    max_length: int = 512
    output_dir: str = "./gsm8k_low_resource"

# 不同模型的配置映射
MODEL_CONFIGS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": StandardConfig,
    "Qwen/Qwen3-30B-A3B-Thinking-2507": StandardConfig,
    "Qwen/Qwen3-Next-80B-A3B-Instruct": LowResourceConfig,  # 大模型使用低资源配置
    "Qwen/Qwen3-Next-80B-A3B-Thinking": LowResourceConfig,
}

def get_config(model_name: str, config_type: str = "standard") -> BaseConfig:
    """获取配置"""
    if config_type == "quick":
        return QuickTestConfig(model_name=model_name)
    elif config_type == "standard":
        config_class = MODEL_CONFIGS.get(model_name, StandardConfig)
        return config_class(model_name=model_name)
    elif config_type == "high_perf":
        return HighPerformanceConfig(model_name=model_name)
    elif config_type == "low_resource":
        return LowResourceConfig(model_name=model_name)
    else:
        raise ValueError(f"未知的配置类型: {config_type}")

# 预定义的实验配置
EXPERIMENT_CONFIGS = {
    "experiment_1": {
        "description": "基础LoRA微调",
        "config": StandardConfig,
        "notes": "使用标准的LoRA配置进行微调"
    },
    "experiment_2": {
        "description": "高秩LoRA微调",
        "config": HighPerformanceConfig,
        "notes": "使用更高的LoRA秩以获得更好的性能"
    },
    "experiment_3": {
        "description": "低资源微调",
        "config": LowResourceConfig,
        "notes": "适用于GPU内存有限的环境"
    },
    "experiment_4": {
        "description": "快速验证",
        "config": QuickTestConfig,
        "notes": "快速验证代码是否正确运行"
    }
}

def list_experiments():
    """列出所有预定义实验"""
    print("可用的实验配置:")
    for exp_id, exp_info in EXPERIMENT_CONFIGS.items():
        print(f"\n{exp_id}: {exp_info['description']}")
        print(f"  说明: {exp_info['notes']}")
        config = exp_info['config']()
        print(f"  LoRA秩: {config.lora_r}")
        print(f"  批次大小: {config.per_device_train_batch_size}")
        print(f"  学习率: {config.learning_rate}")
        print(f"  训练轮数: {config.num_epochs}")

if __name__ == "__main__":
    list_experiments()