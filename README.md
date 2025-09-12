# Prometheus

Prometheus设计的初衷是为大模型提供一套统一的后训练框架，方便用户使用不同的大模型在不同的领域进行后训练

目前支持的模型有：
- Qwen/Qwen3-30B-A3B-Instruct-2507
- Qwen/Qwen3-30B-A3B-Thinking-2507
- Qwen/Qwen3-Next-80B-A3B-Instruct
- Qwen/Qwen3-Next-80B-A3B-Thinking

目前支持的后训练方法有：

# 推荐环境
```
Ubuntu 20.04
CUDA 12.8
Python 3.10
```
在配置环境前执行`sudo apt update && sudo apt upgrade`以确保包是最新的

# 环境配置
```
conda create -n prometheus python=3.10 -y
conda activate prometheus
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


# 使用教程

## 验证环境是否能够实现推理
```
python qwen3.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --prompt "你是什么模型？"
# python qwen3.py --model Qwen/Qwen3-30B-A3B-Thinking-2507 --prompt "你是什么模型？"
# python qwen3.py --model Qwen/Qwen3-Next-80B-A3B-Instruct --prompt "你是什么模型？"
# python qwen3.py --model Qwen/Qwen3-Next-80B-A3B-Thinking --prompt "你是什么模型？"
```
如果遇到无法连接`https://huggingface.co`的情况，请执行：
```
export HF_ENDPOINT=https://hf-mirror.com
```
Qwen/Qwen3-Next-80B-A3B-Instruct和Qwen/Qwen3-Next-80B-A3B-Thinking需要更新版本的transformers库。执行：
```
pip install git+https://github.com/huggingface/transformers.git@main
```