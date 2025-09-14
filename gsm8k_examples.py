#!/usr/bin/env python3
"""
GSM8K 微调使用示例
展示如何使用微调后的模型进行数学推理
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def load_finetuned_model(base_model_path: str, peft_model_path: str, device: str = "auto"):
    """加载微调后的模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    
    return model, tokenizer

def solve_math_problem(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """解决数学问题"""
    # 构建prompt
    prompt = f"""请解决以下数学问题：

问题：{question}

请提供详细的解题步骤，并在最后给出答案。

解题步骤：
"""
    
    # token化
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取生成的部分
    generated_part = response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
    
    return generated_part.strip()

def batch_solve_problems(model, tokenizer, questions: list, max_new_tokens: int = 512) -> list:
    """批量解决数学问题"""
    results = []
    
    for i, question in enumerate(questions):
        print(f"正在解决问题 {i+1}/{len(questions)}...")
        answer = solve_math_problem(model, tokenizer, question, max_new_tokens)
        results.append({
            "question": question,
            "answer": answer
        })
    
    return results

# 示例问题
EXAMPLE_PROBLEMS = [
    "小明有15个苹果，他给了小红6个，又买了8个，现在小明有多少个苹果？",
    "一个长方形的长是12厘米，宽是8厘米，它的面积是多少平方厘米？",
    "商店里一本书原价45元，现在打8折出售，现价是多少元？",
    "火车上午8点从北京出发，下午3点到达上海，全程用了多少小时？",
    "一个水池有进水管和出水管，单开进水管6小时可以注满，单开出水管8小时可以放空，如果同时打开进水管和出水管，多少小时可以注满水池？"
]

def main():
    """主函数 - 使用示例"""
    
    # 模型路径（请根据实际情况修改）
    base_model_path = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    peft_model_path = "./gsm8k_finetuned"
    
    print("正在加载模型...")
    model, tokenizer = load_finetuned_model(base_model_path, peft_model_path)
    print("模型加载完成！")
    
    print("\n" + "="*60)
    print("示例1: 单个问题求解")
    print("="*60)
    
    # 单个问题求解
    question = "小明有24支铅笔，他用了1/3，还剩下多少支？"
    answer = solve_math_problem(model, tokenizer, question)
    
    print(f"问题: {question}")
    print(f"解答:\n{answer}")
    
    print("\n" + "="*60)
    print("示例2: 批量问题求解")
    print("="*60)
    
    # 批量问题求解
    results = batch_solve_problems(model, tokenizer, EXAMPLE_PROBLEMS[:3])
    
    for i, result in enumerate(results):
        print(f"\n问题 {i+1}: {result['question']}")
        print(f"解答: {result['answer'][:200]}...")  # 显示前200字符
    
    print("\n" + "="*60)
    print("示例3: 复杂问题求解")
    print("="*60)
    
    # 复杂问题
    complex_question = """
    甲、乙两地相距360千米。一辆汽车从甲地开往乙地，计划9小时到达。
    因天气原因，实际每小时比计划少行4千米，实际多少小时到达乙地？
    """
    
    complex_answer = solve_math_problem(model, tokenizer, complex_question, max_new_tokens=1024)
    
    print(f"复杂问题: {complex_question}")
    print(f"详细解答:\n{complex_answer}")
    
    print("\n" + "="*60)
    print("示例4: 对比基础模型和微调模型")
    print("="*60)
    
    # 加载基础模型进行对比
    from transformers import AutoModelForCausalLM
    
    print("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    test_question = "一个数的3倍比这个数大18，这个数是多少？"
    
    print(f"\n测试问题: {test_question}")
    
    # 基础模型回答
    base_answer = solve_math_problem(base_model, tokenizer, test_question)
    print(f"\n基础模型回答:\n{base_answer}")
    
    # 微调模型回答
    ft_answer = solve_math_problem(model, tokenizer, test_question)
    print(f"\n微调模型回答:\n{ft_answer}")

def interactive_mode():
    """交互模式"""
    base_model_path = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    peft_model_path = "./gsm8k_finetuned"
    
    print("正在加载模型...")
    model, tokenizer = load_finetuned_model(base_model_path, peft_model_path)
    print("模型加载完成！输入'quit'退出交互模式。")
    
    while True:
        question = input("\n请输入数学问题: ").strip()
        
        if question.lower() in ['quit', 'exit', '退出']:
            break
        
        if not question:
            continue
        
        try:
            answer = solve_math_problem(model, tokenizer, question)
            print(f"\n解答:\n{answer}")
        except Exception as e:
            print(f"生成回答时出错: {e}")

def save_examples():
    """保存示例结果"""
    base_model_path = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    peft_model_path = "./gsm8k_finetuned"
    
    print("正在加载模型...")
    model, tokenizer = load_finetuned_model(base_model_path, peft_model_path)
    
    print("正在生成示例结果...")
    results = batch_solve_problems(model, tokenizer, EXAMPLE_PROBLEMS)
    
    # 保存结果
    output_file = "gsm8k_example_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"示例结果已保存到: {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_mode()
        elif sys.argv[1] == "save_examples":
            save_examples()
        else:
            print("用法: python gsm8k_examples.py [interactive|save_examples]")
    else:
        main()