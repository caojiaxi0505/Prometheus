#!/usr/bin/env python3
"""
GSM8K 完整数据集训练脚本 - 完全复制成功版本
基于fix_gradient_training.py的成功经验，只修改数据集部分
"""

import os
import torch
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数 - 完整数据集训练"""
    logger.info("=== GSM8K 完整数据集训练（成功版本复制）===")
    
    # 模型配置
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    output_dir = "./gsm8k_full_finetuned"
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    
    # 加载tokenizer
    logger.info("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型（不使用量化以避免复杂性）
    logger.info("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # 配置LoRA - 使用成功验证的配置
    logger.info("配置LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # 只选择最基本的模块
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 验证模型梯度
    logger.info("验证模型梯度...")
    model.train()
    
    # 创建简单的测试输入
    test_input = torch.randint(0, min(1000, tokenizer.vocab_size), (1, 10)).to(model.device)
    
    with torch.enable_grad():
        outputs = model(test_input)
        if isinstance(outputs, dict):
            logits = outputs.get('logits')
        else:
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        if logits is not None and logits.requires_grad:
            logger.info("✅ 模型可以产生梯度")
            
            # 计算简单损失
            loss = logits.mean()
            logger.info(f"测试损失: {loss.item():.6f}")
            
            # 反向传播
            loss.backward()
            
            # 检查是否有梯度
            has_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grad = True
                    logger.info(f"✅ 参数 {name} 有梯度")
                    break
            
            if has_grad:
                logger.info("✅ 梯度验证成功")
            else:
                logger.warning("⚠️ 没有找到梯度")
        else:
            logger.error("❌ 模型无法产生梯度")
            return False
    
    # 加载完整数据集 - 这是唯一修改的地方
    logger.info("加载完整GSM8K数据集...")
    dataset = load_dataset("gsm8k", "main", split="train")  # 完整数据集，不再是子集
    
    def format_example(example):
        question = example["question"]
        answer = example["answer"]
        return {"text": f"Question: {question}\nAnswer: {answer}"}
    
    # 格式化数据
    logger.info("格式化数据...")
    formatted_dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="格式化数据"
    )
    
    logger.info(f"完整数据集大小: {len(formatted_dataset)}")  # 应该是7473个样本
    
    # 关键的token化函数 - 使用return_tensors=None避免设备问题
    def tokenize_function(examples):
        # 使用return_tensors=None，让Trainer处理设备转移
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,  # 保持与成功版本相同
            return_tensors=None  # 关键：不指定张量类型
        )
        
        # 创建labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        # 处理pad_token
        if tokenizer.pad_token_id is not None:
            for i in range(len(tokenized["labels"])):
                tokenized["labels"][i] = [
                    label if label != tokenizer.pad_token_id else -100
                    for label in tokenized["labels"][i]
                ]
        
        return tokenized
    
    # Token化 - 完整数据集
    logger.info("Token化完整数据集...")
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Token化数据"
    )
    
    logger.info(f"Token化完成，数据集大小: {len(tokenized_dataset)}")
    
    # 数据收集器 - 关键配置（与成功版本完全相同）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"  # 让数据收集器处理张量
    )
    
    # 训练参数 - 与成功版本完全相同的配置
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # 改为3轮完整训练
        per_device_train_batch_size=1,  # 小批次
        gradient_accumulation_steps=4,  # 梯度累积
        learning_rate=3e-4,
        warmup_steps=10,
        max_grad_norm=1.0,
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to=None,  # 禁用外部日志
        fp16=True,
        dataloader_num_workers=1,
        remove_unused_columns=False,
        # 关键：不设置max_steps，让训练跑完整个数据集
        # 如果需要测试，可以添加：max_steps=100
    )
    
    # 创建训练器
    logger.info("创建训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("开始完整数据集训练...")
    try:
        trainer.train()
        logger.info("✅ 完整数据集训练完成！")
        
        # 保存模型
        logger.info("保存模型...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"🎉 完整数据集训练成功完成！模型保存在: {output_dir}")
        logger.info(f"训练样本数: {len(tokenized_dataset)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)