#!/usr/bin/env python3
"""
GSM8K 推理和评估脚本
用于评估微调后的模型在GSM8K上的性能
"""

import os
import json
import torch
import argparse
import logging
import re
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GSM8KEvaluator:
    """GSM8K评估器"""
    
    def __init__(self, base_model_path: str, finetuned_model_path: str, device: str = "auto"):
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_model(self, use_finetuned: bool = True):
        """加载模型"""
        logger.info(f"正在加载模型 (使用微调模型: {use_finetuned})...")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map=self.device,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if use_finetuned:
                # 加载微调后的模型
                try:
                    # 检查适配器配置文件是否存在
                    import os
                    adapter_config_path = os.path.join(self.finetuned_model_path, "adapter_config.json")
                    adapter_model_path = os.path.join(self.finetuned_model_path, "adapter_model.bin")
                    
                    if not os.path.exists(adapter_config_path):
                        logger.warning(f"未找到适配器配置文件: {adapter_config_path}")
                        # 尝试查找其他可能的适配器文件
                        possible_files = [f for f in os.listdir(self.finetuned_model_path) if f.startswith("adapter")]
                        logger.info(f"找到的适配器相关文件: {possible_files}")
                        
                        if not possible_files:
                            logger.error("未找到任何适配器文件，回退到基础模型")
                            self.model = base_model
                            return
                    
                    # 尝试加载LoRA适配器
                    self.model = PeftModel.from_pretrained(
                        base_model,
                        self.finetuned_model_path
                    )
                    logger.info(f"✅ 已加载微调模型: {self.finetuned_model_path}")
                    
                except Exception as e:
                    logger.error(f"加载微调模型失败: {e}")
                    logger.error(f"错误类型: {type(e).__name__}")
                    logger.info("回退到基础模型")
                    self.model = base_model
            else:
                self.model = base_model
                logger.info("已加载基础模型")
            
            # 确保模型和tokenizer可用
            if self.model is None:
                raise RuntimeError("模型加载失败")
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer加载失败")
                
            logger.info("模型加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise RuntimeError(f"模型加载失败: {str(e)}")
    
    def extract_numeric_answer(self, text: str) -> Optional[str]:
        """从文本中提取数值答案"""
        # 尝试多种格式提取答案
        patterns = [
            r"####\s*([-\d.,]+)",  # GSM8K标准格式
            r"答案是\s*[:：]?\s*([-\d.,]+)",  # 中文格式
            r"answer is\s*[:：]?\s*([-\d.,]+)",  # 英文格式
            r"最终答案\s*[:：]?\s*([-\d.,]+)",  # 最终答案
            r"因此.*?([-\d.,]+)",  # 因此...
            r"所以.*?([-\d.,]+)",  # 所以...
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                # 清理答案（移除逗号等）
                answer = matches[-1].replace(",", "").strip()
                # 确保是有效的数字
                try:
                    float(answer)
                    return answer
                except ValueError:
                    continue
        
        return None
    
    def generate_response(self, question: str, max_new_tokens: int = 512) -> str:
        """生成回答"""
        # 检查模型和tokenizer是否已加载
        if not hasattr(self, 'model') or self.model is None:
            # 尝试重新加载模型
            try:
                self.load_model(use_finetuned=True)
            except Exception as e:
                raise RuntimeError(f"模型加载失败: {e}")
        
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            raise RuntimeError("tokenizer未正确加载")
        
        try:
            # 构建prompt
            prompt = f"""请解决以下数学问题：

问题：{question}

请提供详细的解题步骤，并在最后给出答案。

解题步骤：
"""
            
            # token化
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # 确保输入在正确的设备上
            try:
                if hasattr(self.model, 'device'):
                    device = self.model.device
                else:
                    device = next(self.model.parameters()).device
            except Exception as e:
                logger.warning(f"无法确定模型设备，使用默认设备: {e}")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = self.model.to(device)
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
                )
            
            # 解码
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的部分
            input_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            generated_part = response[len(input_text):]
            
            return generated_part.strip()
            
        except Exception as e:
            logger.error(f"生成回答时出错: {e}")
            logger.error(f"错误详情: {type(e).__name__}: {str(e)}")
            return f"生成回答时出错: {type(e).__name__}: {str(e)}"
    
    def evaluate_single(self, question: str, true_answer: str) -> Tuple[str, str, bool]:
        """评估单个问题"""
        # 生成回答
        response = self.generate_response(question)
        
        # 提取预测答案
        pred_answer = self.extract_numeric_answer(response)
        
        # 提取真实答案
        true_numeric = self.extract_numeric_answer(true_answer)
        if true_numeric is None and "####" in true_answer:
            true_numeric = true_answer.split("####")[-1].strip()
        
        # 比较答案
        is_correct = False
        if pred_answer is not None and true_numeric is not None:
            try:
                # 数值比较（考虑浮点数精度）
                pred_val = float(pred_answer)
                true_val = float(true_numeric)
                is_correct = abs(pred_val - true_val) < 1e-6
            except ValueError:
                # 字符串比较
                is_correct = pred_answer.strip() == true_numeric.strip()
        
        return response, pred_answer, is_correct
    
    def evaluate_dataset(self, num_samples: int = None) -> Dict:
        """评估数据集"""
        logger.info("正在加载GSM8K测试集...")
        
        # 加载测试集
        test_dataset = load_dataset("gsm8k", "main", split="test")
        
        if num_samples:
            test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))
        
        logger.info(f"评估数据集大小: {len(test_dataset)}")
        
        results = []
        correct_count = 0
        
        # 评估每个样本
        for i, example in enumerate(tqdm(test_dataset, desc="评估中")):
            question = example["question"]
            answer = example["answer"]
            
            response, pred_answer, is_correct = self.evaluate_single(question, answer)
            
            results.append({
                "index": i,
                "question": question,
                "true_answer": answer,
                "predicted_response": response,
                "predicted_answer": pred_answer,
                "is_correct": is_correct
            })
            
            if is_correct:
                correct_count += 1
            
            # 定期记录进度
            if (i + 1) % 10 == 0:
                current_accuracy = correct_count / (i + 1)
                logger.info(f"进度: {i+1}/{len(test_dataset)}, 当前准确率: {current_accuracy:.2%}")
        
        # 计算最终指标
        accuracy = correct_count / len(results)
        
        metrics = {
            "total_samples": len(results),
            "correct_predictions": correct_count,
            "accuracy": accuracy,
            "results": results
        }
        
        return metrics
    
    def compare_models(self, num_samples: int = 100) -> Dict:
        """比较基础模型和微调模型"""
        logger.info(f"比较基础模型和微调模型 (样本数: {num_samples})...")
        
        # 加载测试集
        test_dataset = load_dataset("gsm8k", "main", split="test")
        test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))
        
        comparison_results = []
        
        for i, example in enumerate(tqdm(test_dataset, desc="比较中")):
            question = example["question"]
            answer = example["answer"]
            
            # 基础模型预测
            self.load_model(use_finetuned=False)
            base_response, base_answer, base_correct = self.evaluate_single(question, answer)
            
            # 微调模型预测
            self.load_model(use_finetuned=True)
            ft_response, ft_answer, ft_correct = self.evaluate_single(question, answer)
            
            comparison_results.append({
                "index": i,
                "question": question,
                "true_answer": answer,
                "base_response": base_response,
                "base_answer": base_answer,
                "base_correct": base_correct,
                "ft_response": ft_response,
                "ft_answer": ft_answer,
                "ft_correct": ft_correct,
                "improvement": ft_correct and not base_correct
            })
        
        # 计算比较指标
        base_correct = sum(1 for r in comparison_results if r["base_correct"])
        ft_correct = sum(1 for r in comparison_results if r["ft_correct"])
        improvements = sum(1 for r in comparison_results if r["improvement"])
        
        comparison_metrics = {
            "total_samples": len(comparison_results),
            "base_model_accuracy": base_correct / len(comparison_results),
            "finetuned_model_accuracy": ft_correct / len(comparison_results),
            "improvements": improvements,
            "comparison_results": comparison_results
        }
        
        return comparison_metrics
    
    def save_results(self, results: Dict, output_file: str):
        """保存评估结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="GSM8K模型评估")
    parser.add_argument("--base_model", type=str, required=True,
                       help="基础模型路径")
    parser.add_argument("--finetuned_model", type=str, required=True,
                       help="微调模型路径")
    parser.add_argument("--mode", type=str, choices=["evaluate", "compare"], 
                       default="evaluate", help="评估模式")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="评估样本数量（None表示全部）")
    parser.add_argument("--output_file", type=str, default="gsm8k_results.json",
                       help="输出文件路径")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备（auto, cuda, cpu）")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = GSM8KEvaluator(
        base_model_path=args.base_model,
        finetuned_model_path=args.finetuned_model,
        device=args.device
    )
    
    if args.mode == "evaluate":
        # 评估微调模型
        logger.info("开始评估微调模型...")
        results = evaluator.evaluate_dataset(num_samples=args.num_samples)
        
        # 打印结果摘要
        logger.info(f"评估完成！")
        logger.info(f"总样本数: {results['total_samples']}")
        logger.info(f"正确预测数: {results['correct_predictions']}")
        logger.info(f"准确率: {results['accuracy']:.2%}")
        
    elif args.mode == "compare":
        # 比较基础模型和微调模型
        logger.info("开始比较模型...")
        results = evaluator.compare_models(num_samples=args.num_samples or 100)
        
        # 打印比较结果
        logger.info(f"比较完成！")
        logger.info(f"总样本数: {results['total_samples']}")
        logger.info(f"基础模型准确率: {results['base_model_accuracy']:.2%}")
        logger.info(f"微调模型准确率: {results['finetuned_model_accuracy']:.2%}")
        logger.info(f"改进样本数: {results['improvements']}")
    
    # 保存结果
    evaluator.save_results(results, args.output_file)
    
    logger.info("评估完成！")

if __name__ == "__main__":
    main()