# Transformer Language Model from Scratch

在llm coding和vibe coding盛行的时代，逐渐感觉到自己的代码能力越来越弱，同时本着想深入学习一个基础的大模型是如何构建成的，所以本项目诞生了。

项目完成后发现手写一个完整的大模型和训练过程并不复杂，希望可以帮助更多的人学习大模型。

欢迎对本人的代码批评指正，欢迎提出Github Issue / Github PR : )

本项目是一个参考CS336课程，在允许使用Pytorch中的基础库的情况下，从零开始手写实现的Transformer大语言模型，包括了以下内容：

- 分词器，Tokenizer
  - BPE分词器训练
  - Tokenizer保存与加载
- 大模型结构
  - pre-rmsnorm
  - RoPE
  - SwiGLU
- 模型训练
  - Cross Entropy损失函数
  - AdamW优化器
  - cosine_schedule学习率控制
  - gradient_norm_clipping模型梯度控制
- 模型推理
  - 温度采样，核采样实现
  - 在测试集上测试模型损失函数

该项目特别关注在TinyStories数据集上的预训练和测试，并对模型结构和超参数进行了详细的实验和分析，具体分析文档见test_logs，包括：

- 探究模型的不同超参数对模型训练结果的影响
- 探究不同的模型结构（Layer norm位置，RoPE，SwiGLU or SiLU等）对模型训练结果的影响

## 项目结构
```
TransformerFromScratch/
├── modules/                    # 核心模块
│   ├── __init__.py
│   └── modules.py             # 所有核心组件实现
├── train/                     # 训练相关脚本
│   ├── config.py             # 训练配置
│   ├── train_my_bpe.py       # BPE分词器训练
│   ├── tokenize_corpus.py    # 语料库分词
│   └── train_my_lm.py        # 语言模型训练
├── test/                      # 测试和推理脚本
│   ├── test_my_lm.py         # 模型文本生成测试
│   └── val_my_lm.py          # 模型验证评估
├── data/                      # 数据文件夹
│   ├── TinyStoriesV2-GPT4-train.txt
│   ├── TinyStoriesV2-GPT4-valid.txt
│   └── *.tokenized.npy       # 分词后的数据
├── models/                    # 模型保存目录
└── test_logs/                # 实验日志
```

## 核心组件详解

### 1. BPE分词器 (`My_tokenizer`)

实现了完整的BPE（Byte Pair Encoding）分词算法：

- **预分词处理**：支持正则表达式预分词和特殊标记处理
- **并行化训练**：使用多进程加速BPE训练过程
- **特殊标记支持**：正确处理`<|endoftext|>`等特殊标记
- **编码/解码**：支持文本到token ID的双向转换

核心特性：
```python
# 分词器初始化
tokenizer = My_tokenizer(special_tokens=['<|endoftext|>'])

# 文本编码
token_ids = tokenizer.encode("Hello, world!")

# 解码还原
text = tokenizer.decode(token_ids)
```

### 2. Transformer架构组件

#### 基础层
- **`My_Linear`**: 自定义线性层，使用截断正态分布初始化
- **`My_Embedding`**: 词嵌入层
- **`My_rmsnorm`**: RMSNorm归一化层，相比LayerNorm更稳定

#### 注意力机制
- **`My_RoPE`**: 旋转位置编码（Rotary Position Embedding）
- **`My_scaled_dot_product_attention`**: 缩放点积注意力
- **`My_multihead_attention`**: 多头注意力机制，集成RoPE位置编码

#### 前馈网络
- **`My_SwiGLU`**: SwiGLU激活函数的前馈网络
- **`My_SiLU`**: Swish激活函数实现

#### 完整模型
- **`My_transformer_block`**: 单个Transformer层（注意力+前馈+残差连接）
- **`My_transformer_lm`**: 完整的语言模型架构

### 3. 训练组件

#### 优化器
- **`My_AdamW`**: AdamW优化器实现，支持权重衰减

#### 学习率调度
- **`My_lr_cosine_schedule`**: 余弦退火学习率调度，支持warmup

#### 训练工具
- **`My_cross_entropy`**: 数值稳定的交叉熵损失
- **`My_gradient_clipping`**: 梯度裁剪防止梯度爆炸
- **`My_get_batch`**: 高效的批次数据采样
- **`My_save_checkpoint`/`My_load_checkpoint`**: 模型检查点保存和加载

## 使用方法

### 1. 环境配置

- 推荐使用uv管理项目环境

```bash
# 使用uv管理项目虚拟环境
cd TransformerFromScratch
uv sync
source .venv/bin/activate
```

- 使用conda管理项目环境

```bash
# 使用conda管理项目虚拟环境
cd TransformerFromScratch
conda create --name transformer_lm python=3.12
conda activate transformer_lm
pip install .
```

- 数据集下载

使用TinyStories数据集：

**训练集**: 2.12M tokens

**验证集**: 22K tokens

TinyStories是专门为小型语言模型设计的数据集，包含简单的儿童故事，适合用作简单模型的预训练语料。

```bash
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

### 2. 训练BPE分词器

```bash
python train/train_my_bpe.py
```

这将在TinyStories数据集上训练一个10000词汇量的BPE分词器。

### 3. 数据预处理

```bash
python train/tokenize_corpus.py
```

将原始文本转换为token序列并保存为numpy数组。

### 4. 训练语言模型

```bash
python train/train_my_lm.py
```

使用config.py配置文件中的超参数训练Transformer语言模型。

### 5. 模型测试

```bash
python test/test_my_lm.py    # 文本生成测试，可自定文本前缀生成后续文本
python test/val_my_lm.py     # 在测试集上测试交叉熵损失
```

⚠️ Academic Honesty Notice  
This repository contains my solutions to assignments for CS336 (2025).  
It is shared **for educational and reference purposes only**.

- ✅ You are welcome to:  
  - Study the code to understand concepts  
  - Run experiments locally  
  - Cite this work (with attribution)  

- ❌ Please do **NOT**:  
  - Submit this code (or minor modifications) as your own coursework  
  - Use it to violate your institution's academic integrity policy  

If you're taking CS336 (or similar), try solving problems yourself first — you’ll learn more! 😊

