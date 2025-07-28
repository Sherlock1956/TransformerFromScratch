import torch
import numpy as np
from tqdm import tqdm
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.modules import *
def choose_next_token(logits, tempereture, top_p, top_k):
    next_token_logits = logits.squeeze()[-1]
    
    # 如果温度为0，直接返回最大概率的token（贪心解码）
    if tempereture == 0:
        return torch.argmax(next_token_logits).item()
    
    # 应用温度缩放
    next_token_logits = next_token_logits / tempereture
    
    # Top-k 采样：只保留概率最高的k个token
    if top_k > 0:
        # 获取top-k的值和索引
        top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
        # 创建一个mask，只保留top-k的logits
        logits_mask = torch.full_like(next_token_logits, float('-inf'))
        logits_mask[top_k_indices] = top_k_values
        next_token_logits = logits_mask
    
    # 计算概率分布
    probabilities = torch.softmax(next_token_logits, dim=0)
    
    # Top-p 采样（Nucleus Sampling）
    if top_p < 1.0:
        # 按概率从大到小排序
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        # 找到累积概率超过top_p的位置
        cutoff_index = torch.where(cumulative_probs > top_p)[0]
        if len(cutoff_index) > 0:
            # 保留累积概率在top_p范围内的token
            cutoff_index = cutoff_index[0].item()
            # 创建新的概率分布
            nucleus_probs = torch.zeros_like(probabilities)
            nucleus_probs[sorted_indices[:cutoff_index + 1]] = sorted_probs[:cutoff_index + 1]
            # 重新归一化
            nucleus_probs = nucleus_probs / nucleus_probs.sum()
            probabilities = nucleus_probs
    
    # 从概率分布中采样
    next_token = torch.multinomial(probabilities, num_samples=1).item()
    
    return next_token
stream = True
# 1. prepare model
model_path = "models/test_6/checkpoint_80000"
config_path = os.path.join(os.path.dirname(model_path),'config.json')
config = json.load(open(config_path,'r'))
device = config['device']
transformer_lm = My_transformer_lm(vocab_size=config['vocab_size'], context_length=config['context_length'], d_model=config['d_model'], num_layers=config['num_layers'], num_heads=config['num_heads'], d_ff=config['d_ff'], rope_theta=config['rope_theta'])
transformer_lm.to(device)
trainable_parameters = sum(param.numel() for param in transformer_lm.parameters() if param.requires_grad)
print(f"Transformer lm has {trainable_parameters / 1e9} B trainable parameters")
My_load_checkpoint(model_path,transformer_lm,None)
# 2. prepare data
tokenizer = My_tokenizer(special_tokens=['<|endoftext|>'])
data = 'Tim and Sue were friends who liked to play with their bicycles. They'
prefill_token = tokenizer.encode(data)
prefill_token = torch.tensor(prefill_token).unsqueeze(0)

# 3. generation
max_token_len = 256
generated_tokens = []

# 将prefill_token移动到设备
prefill_token = prefill_token.to(device)

# 自回归生成
if stream == True:
    print(data, end='')
    stream_code = []
for _ in (range(max_token_len - prefill_token.size(1))):
    # 前向传播
    with torch.no_grad():
        probabilities = transformer_lm(prefill_token)
    
    # 取最后一个位置的logits并选择下一个token
    next_token = choose_next_token(probabilities, tempereture=0.8, top_p=0.9, top_k=50)
    
    # 检查是否为结束token
    if next_token == 256:  # <|endoftext|> token
        break
    
    # 将新token添加到序列末尾
    next_token_tensor = torch.tensor([[next_token]], device=device)
    prefill_token = torch.cat([prefill_token, next_token_tensor], dim=1)
        
    if stream == True:
        stream_code.append(next_token)
        next_letter = tokenizer.decode(stream_code)
        if '\uFFFD' not in next_letter:
            print(next_letter, end='')
            stream_code = []
    else:
        generated_tokens.append(next_token)

if not stream:
    generated_text = tokenizer.decode(generated_tokens)
    data += generated_text
    print(f"Generated text: {data}")


