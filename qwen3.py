#!/usr/bin/env python3
"""
Qwen3 模型调用脚本 - 128GB内存优化版
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import os
import torch
import gc

def load_model_and_tokenizer(model_name):
    model_dir = os.path.join(os.getcwd(), "models", model_name.replace("/", "-"))
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"模型将保存到: {model_dir}")
    
    # 检查系统资源
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"GPU {i}: {gpu_mem:.1f}GB")
    
    import psutil
    total_mem = psutil.virtual_memory().total / 1e9
    available_mem = psutil.virtual_memory().available / 1e9
    print(f"系统内存: {total_mem:.1f}GB (可用: {available_mem:.1f}GB)")
    
    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=model_dir,
        trust_remote_code=True
    )
    
    loading_strategies = []
    
    # 策略1：FP16直接加载
    if not "Next" in model_name:
        loading_strategies.append({
            "name": "双GPU FP16直接加载",
            "params": {
                "dtype": torch.float16,
                "device_map": "auto",
                "cache_dir": model_dir,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "max_memory": {
                    0: "44GB",
                    1: "44GB",
                    "cpu": "100GB"
                }
            }
        })

    try:
        import bitsandbytes
        print("✓ bitsandbytes可用，开始量化加载")
        
        # 策略2：4bit量化 - 最快最稳定
        bnb_config_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        loading_strategies.append({
            "name": "双GPU 4bit量化",
            "params": {
                "dtype": torch.float16,
                "device_map": "auto",
                "quantization_config": bnb_config_4bit,
                "cache_dir": model_dir,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "max_memory": {
                    0: "40GB",
                    1: "40GB",
                    "cpu": "100GB"
                }
            }
        })
        
        # 策略3：8bit量化 - 更高精度
        bnb_config_8bit = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        loading_strategies.append({
            "name": "双GPU 8bit量化",
            "params": {
                "dtype": torch.float16,
                "device_map": "auto",
                "quantization_config": bnb_config_8bit,
                "cache_dir": model_dir,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
                "max_memory": {
                    0: "40GB",
                    1: "40GB", 
                    "cpu": "100GB"
                }
            }
        })
        
    except ImportError:
        print("✗ bitsandbytes不可用，请安装: pip install bitsandbytes")
    
    for strategy in loading_strategies:
        try:
            print(f"\n🚀 尝试: {strategy['name']}")
            
            # 清理内存
            gc.collect()
            torch.cuda.empty_cache()
            
            model = AutoModelForCausalLM.from_pretrained(model_name, **strategy['params'])
            
            print(f"✅ 成功加载: {strategy['name']}")
            
            # 显示加载结果
            current_mem = psutil.virtual_memory().percent
            print(f"📊 当前系统内存使用: {current_mem:.1f}%")
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    total = torch.cuda.get_device_properties(i).total_memory / 1e9
                    print(f"📊 GPU {i}: {allocated:.1f}GB / {total:.1f}GB ({allocated/total*100:.1f}%)")
            
            # 显示模型分布
            if hasattr(model, 'hf_device_map'):
                gpu_0_layers = sum(1 for k, v in model.hf_device_map.items() if "layers" in k and v == 0)
                gpu_1_layers = sum(1 for k, v in model.hf_device_map.items() if "layers" in k and v == 1)
                cpu_layers = sum(1 for k, v in model.hf_device_map.items() if "layers" in k and v == "cpu")
                
                print(f"🔧 模型分布: GPU0({gpu_0_layers}层) + GPU1({gpu_1_layers}层) + CPU({cpu_layers}层)")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ {strategy['name']} 失败: {str(e)[:200]}...")
            gc.collect()
            torch.cuda.empty_cache()
            continue
    
    raise RuntimeError("🚫 所有加载策略都失败了")

def generate_response(model, tokenizer, prompt, max_new_tokens=None):
    """生成响应 - 高性能版"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # 智能设备检测
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # 优先使用embed_tokens的设备
        input_device = model.hf_device_map.get('model.embed_tokens', 0)
    else:
        input_device = next(model.parameters()).device
    
    model_inputs = tokenizer([text], return_tensors="pt").to(input_device)
    
    if max_new_tokens is None:
        max_new_tokens = 32768 if "Thinking" in model.name_or_path else 16384
    
    print(f"🎯 开始生成 (最大token: {max_new_tokens}, 设备: {input_device})")
    
    # 优化的生成参数
    generation_kwargs = {
        **model_inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.8,
        "repetition_penalty": 1.05,
        "pad_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id,
        "use_cache": True,
    }
    
    with torch.no_grad():
        generated_ids = model.generate(**generation_kwargs)
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    # 解析输出
    if "Thinking" in model.name_or_path:
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        return thinking_content, content
    else:
        content = tokenizer.decode(output_ids, skip_special_tokens=True)
        return None, content

def main():
    parser = argparse.ArgumentParser(description="Qwen3模型调用脚本")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    
    args = parser.parse_args()
    
    print(f"🚀 正在加载模型: {args.model}")
    print(f"🖥️  检测到 {torch.cuda.device_count()} 个GPU")
    
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    thinking_content, content = generate_response(model, tokenizer, args.prompt, args.max_new_tokens)
    
    if thinking_content:
        print("\n" + "="*50)
        print("🧠 THINKING CONTENT")
        print("="*50)
        print(thinking_content)
    
    print("\n" + "="*50)
    print("💬 RESPONSE") 
    print("="*50)
    print(content)

if __name__ == "__main__":
    main()
