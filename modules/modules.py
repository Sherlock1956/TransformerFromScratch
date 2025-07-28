import regex
import json
import re
import torch.nn as nn
import torch
import numpy as np
from collections.abc import Iterable, Iterator, Callable, Iterable
from typing import Optional
import math
import os
from tqdm import tqdm
from typing import BinaryIO
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pickle
def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(chunk, special_tokens=["<|endoftext|>"]):
    """
    处理一个文本块，先按特殊标记分割，然后对每个子块进行预分词
    
    Args:
        chunk: 要处理的文本块
        special_tokens: 特殊标记列表，默认为["<|endoftext|>"]
        
    Returns:
        预分词后的token列表
    """
    re_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # 构建用于分割的正则表达式，转义特殊字符并用|连接
    split_pattern = "|".join(map(re.escape, special_tokens))
    
    # 按特殊标记分割文本
    sub_chunks = regex.split(split_pattern, chunk)
    
    # 对每个子块进行预分词并合并结果
    all_tokens = []
    for sub_chunk in sub_chunks:
        if sub_chunk:  # 忽略空子块
            tokens = regex.findall(re_pattern, sub_chunk)
            all_tokens.extend(tokens)
    
    return all_tokens

def pre_tokenize(file_path: str, num_process: int, split_special_token: str = "<|endoftext|>") -> dict[str, int]:
    ## Usage
    with open(file_path, "rb") as f:  # 使用传入的file_path参数
        boundaries = find_chunk_boundaries(
            f, num_process, split_special_token.encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.

        # parallelize implementation
        chunks = []
        token_counts = Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
        
        special_tokens = [split_special_token]  # 可以根据需要添加更多特殊标记
        before_time = time.time()
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_chunk, chunk, special_tokens) for chunk in chunks}
            for future in tqdm(as_completed(futures),total=len(futures)):
                result = future.result()  # 添加括号调用方法
                token_counts.update(result)
        after_time = time.time()
        print(f"It takes {after_time - before_time} seconds on pre-tokenize with {num_process} processes")
        return token_counts
def My_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    counter_save_path = f"{input_path}.counter.pkl"
    if os.path.exists(counter_save_path):
        counter = pickle.load(open(counter_save_path, "rb"))
    else:
        counter = pre_tokenize(str(input_path),num_process=32)
        pickle.dump(counter, open(counter_save_path, "wb"))
    # counter = pre_tokenize(str(input_path),num_process=32)
    # 一定注意初始化corpus的时候，需要先将整体编码成utf-8，然后拆分成单个的bytes
    corpus = {tuple(bytes([b]) for b in token.encode('utf-8')): counter[token] for token in counter}
    vocab = {i: bytes([i]) for i in range(256)}
    for i, item in enumerate(special_tokens):
        vocab[256 + i] = item.encode('utf-8')
    merge_start_id = 256 + len(special_tokens)
    merges = []
    pbar = tqdm(total = vocab_size - len(vocab))
    merge_id = merge_start_id
    updated_pairs = set(corpus.keys())
    pair_frequency = {}
    while(len(vocab) < vocab_size):
        for key, value in corpus.items():
            if key not in updated_pairs:
                continue
            for i in range(len(key) - 1):
                pair_key = (key[i],key[i+1])
                pair_frequency[pair_key] = pair_frequency.get(pair_key,0) + value
        most_frequency = max(pair_frequency.values())
        most_frequent_pairs = [key for key in pair_frequency.keys() if pair_frequency[key] == most_frequency]
        most_frequent_pair = max(most_frequent_pairs)
        vocab[merge_id] = most_frequent_pair[0] + most_frequent_pair[1]
        if most_frequent_pair[0] + most_frequent_pair[1] == b'Hello':
            print(most_frequent_pair[0].decode('utf-8') +'----' + most_frequent_pair[1].decode('utf-8'))
        merges.append(most_frequent_pair)
        new_corpus = {}
        updated_pairs = set()
        for word, freq in corpus.items():
            if most_frequent_pair[0] not in word or most_frequent_pair[1] not in word:
                new_corpus[word] = freq
                continue
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == most_frequent_pair:
                    new_word.append(most_frequent_pair[0] + most_frequent_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            if new_word != word:
                updated_pairs.add(new_word)
                for i in range(len(word) - 1):
                    pair_key = (word[i],word[i+1])
                    pair_frequency[pair_key] = pair_frequency.get(pair_key,0) - freq
            new_corpus[new_word] = freq
        corpus = new_corpus
        merge_id += 1
        pbar.update()
    return vocab, merges
class My_tokenizer():
    def __init__(self, vocab=None, merges=None, special_tokens: list=None):
        assert isinstance(special_tokens, list), "special_tokens must be a list"
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        if not self.special_tokens:
            self.special_tokens = []
        if not vocab or not merges:
            self.vocab, self.merges = self.from_files()
        self.merges_dict = {(item[0], item[1]): i for i, item in enumerate(self.merges)}
        self.reverse_vocab = {v:k for k, v in self.vocab.items()}
    def from_files(self, vocab_path = None, merges_path = None):
        vocab_path = "data/TinyStoriesV2-GPT4-train.txt.vocab.json"
        merges_path = "data/TinyStoriesV2-GPT4-train.txt.merges.json"
        vocab = json.load(open(vocab_path, 'r',encoding='utf-8'))
        vocab = {int(k):v.encode('latin1') for k, v in vocab.items()}
        with open(merges_path, "r", encoding="utf-8") as f:
            merges_utf = json.load(f)
            merges = [(a.encode('latin1'), b.encode('latin1')) for a, b in merges_utf]
        return vocab, merges
    def split_text_with_tokens(self, text):
        """对special tokens需要特殊处理，先以special tokens分隔，保存special tokens的分隔记录"""
        result = []
        if self.special_tokens:
            pattern = '|'.join(map(re.escape, sorted(self.special_tokens,key=len,reverse=True)))
            pattern = f'({pattern})'
            
            # 使用 re.split 进行分割
            parts = re.split(pattern, text)
            for i in range(1, len(parts), 2):
                # 每个分隔符前面的文本段是 parts[i-1]
                # 当前分隔符是 parts[i]
                text_segment = parts[i-1]
                separator = parts[i]
                result.append((text_segment, separator))
            if parts[-1]:
                result.append((parts[-1], None))
        else:
            result.append((text,None))
        return result
    def pre_tokenize(self, text: str) -> list[str]:
        """正确处理了special tokens的问题"""
        re_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        split_text = self.split_text_with_tokens(text)
        tokens = []
        for pair in split_text:
            tokens.extend(regex.findall(re_pattern, pair[0]))
            if pair[1]:
                tokens.append(pair[1])
        return tokens
    def find_pairs_in_merges(self, token_list):
        pairs_in_merges = {}
        for i in range(len(token_list) - 1):
            if (token_list[i],token_list[i+1]) in self.merges_dict:
                pairs_in_merges[i] = self.merges_dict[(token_list[i],token_list[i+1])]
        return pairs_in_merges
    def tokenize_single_token(self, token):
        """注意在merges时，不应该直接判断是否在字典中，而是应该按照merges顺序来合并"""
        token_list = [bytes([b]) for b in token.encode('utf-8')]
        i = 0
        result = []
        pairs_in_merges = self.find_pairs_in_merges(token_list)
        while(len(pairs_in_merges) > 0):
            earliest_index = min(pairs_in_merges.items(),key=lambda x: x[1])[0]
            token_list[earliest_index:earliest_index+2] = [token_list[earliest_index]+token_list[earliest_index+1]]
            pairs_in_merges = self.find_pairs_in_merges(token_list)
        result = [self.reverse_vocab[token] for token in token_list]
        return result
    def encode(self, text: str) -> list[int]:
        result = []
        tokens = self.pre_tokenize(text)
        for token in tokens:
            if token in self.special_tokens:
                result.extend([self.reverse_vocab[token.encode('utf-8')]])
            else:
                result.extend(self.tokenize_single_token(token))
        return result
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pos = iterable.tell()
        line_count = sum(1 for _ in iterable)
        iterable.seek(pos)  # 回到原位置
        pbar = tqdm(total = line_count)
        for line in (iterable):
            pbar.update(1)
            for token in self.encode(line):
                yield token
    def decode(self, ids: list[int]) -> str:
        result = b''
        for index in ids:
            result += self.vocab.get(index)
        """注意这里需要使用errors='replace'来保证可以解决非法字符的问题"""
        result = result.decode('utf-8',errors='replace')
        return result

class My_Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        Weight = torch.empty(self.out_features,self.in_features,dtype=dtype,device=device)
        self.weight = nn.Parameter(Weight)
        std = (2/(self.in_features+self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight,0,std,-3*std,3*std)
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x @ self.weight.transpose(-1,-2)
        return res

class My_Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings,self.embedding_dim,dtype=self.dtype,device=self.device)
        )
        nn.init.trunc_normal_(self.weight, 0,1,-3,3)
    def forward(self, x):
        return self.weight[x]

class My_rmsnorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x):
        in_type = x.dtype
        x = x.to(torch.float32)
        # 需要 keepdim=True 保持维度一致，方便广播
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        rmsnorm = x / rms * self.weight
        return rmsnorm.to(in_type)

class My_SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)
# feedforward Network
class My_SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if self.d_ff == None:
            self.d_ff = (8 * self.d_model) // 3
        self.device = device
        self.dtype = dtype
        self.SiLU = My_SiLU()
        # 注意这里的W_1 2 3是一个类，不是一个参数矩阵了，调用的时候会调用forward函数，也不需要手动转换输入输出维度，因为在Linear的实现中已经转换了
        self.w1 = My_Linear(d_model, d_ff, self.device, self.dtype)
        self.w2 = My_Linear(d_ff, d_model, self.device, self.dtype)
        self.w3 = My_Linear(d_model, d_ff, self.device, self.dtype)
    def forward(self, x):
        result = self.w2((self.SiLU(self.w1(x)) * self.w3(x)))
        return result
class My_RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000, token_positions=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.token_positions = token_positions
        # 计算 inv_freq
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # 注册 inv_freq，不需要保存到 state_dict
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 预先缓存 cos/sin 编码
        self._build_cache()
    
    def _build_cache(self):
        # 生成 [max_seq_len, dim//2] 的位置编码
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device).float()
        freqs = t.unsqueeze(1) * self.inv_freq.unsqueeze(0) # [seq, dim//2]
        emb = freqs
        # 缓存 cos/sin
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    def forward(self, x, seq_dim=1):
        # x: [batch, seq, dim]
        seq_len = x.shape[seq_dim]
        if self.token_positions is not None:
            pos = self.token_positions.squeeze(0)  # [seq]
            cos = self._cos_cached[pos]
            sin = self._sin_cached[pos]
        else:
            cos = self._cos_cached[:seq_len]
            sin = self._sin_cached[:seq_len]
        # 旋转操作
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x_rot = x_rot.flatten(-2)
        return x_rot
class My_softmax(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, dimension=-1):
        max_vec = torch.max(x, dim=dimension, keepdim=True)[0]
        x -= max_vec
        result = torch.exp(x) / torch.sum(torch.exp(x),dim=dimension,keepdim=True)
        return result
 
class My_scaled_dot_product_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = My_softmax()
    def forward(self, q, k, v, mask=None):
        d_k = q.shape[-1]
        attention = q @ k.transpose(-1,-2) / d_k ** 0.5
        if mask is not None:
            attention = attention.masked_fill(~mask, float('-inf'))
        result = self.softmax(attention, dimension=-1) @ v
        return result
class My_multihead_attention(nn.Module):
    def __init__(self, d_model, num_head, max_seq_len=None, theta=None, token_positions=None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        assert self.d_model % self.num_head == 0
        self.head_dim = d_model // num_head
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.token_positions = token_positions
        self.q_proj = My_Linear(d_model, d_model)
        self.k_proj = My_Linear(d_model, d_model)
        self.v_proj = My_Linear(d_model, d_model)
        self.output_proj = My_Linear(d_model, d_model)
        if self.max_seq_len is not None:
            self.RoPE = My_RoPE(self.head_dim, max_seq_len, theta, token_positions)
        self.attention = My_scaled_dot_product_attention()
        
    def forward(self, x):
        batch_size, seq_len = x.shape[0:2]
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)
        if hasattr(self, 'RoPE'):
            # 只能在Q和K上作用旋转位置编码！！！！
            Q = self.RoPE(Q, seq_dim = -2)
            K = self.RoPE(K, seq_dim = -2)
        mask = torch.ones((seq_len, seq_len),dtype=torch.bool)
        mask = ~torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        mask = mask.expand(batch_size, self.num_head, seq_len, seq_len)  # [batch_size, num_head, seq_len, seq_len]
        mask = mask.to(x.device)
        result = self.attention(Q, K, V, mask).transpose(1,2).contiguous()\
            .view(batch_size, seq_len, self.num_head * self.head_dim)
        return self.output_proj(result)

class My_transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,):
        super().__init__()
        self.ln1 = My_rmsnorm(d_model)
        self.ln2 = My_rmsnorm(d_model)
        self.ffn = My_SwiGLU(d_model, d_ff)
        self.attn = My_multihead_attention(d_model, num_heads, max_seq_len, theta)
    def forward(self, x):
        res_x = x.clone()
        x = self.ln1(x)
        x = self.attn(x)
        x = res_x + x
        res_x = x.clone()
        x = self.ln2(x)
        x = self.ffn(x)
        result = x + res_x
        return result

class My_transformer_lm(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.token_embeddings = My_Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.ln_final = My_rmsnorm(d_model=d_model)
        self.lm_head = My_Linear(d_model, vocab_size)
        self.layers = nn.ModuleList([
            My_transformer_block(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
    def forward(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        # 注意这里不需要加softmax!!!
        return x
def My_cross_entropy(inputs, targets):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # 注意Cancel out log and exp whenever possible.不要先计算softmax再计算log，会造成数值不稳定
    vocab_size = inputs.shape[-1]
    inputs = inputs.view(-1, vocab_size)
    targets = targets.view(-1)
    targets = targets.unsqueeze(1)
    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True)[0]
    logexpsum = torch.log(torch.sum(torch.exp(inputs),dim=-1))
    target_logits = inputs.gather(dim=-1, index=targets)
    loss = logexpsum - target_logits
    return loss.mean()

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

class My_AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), weight_decay=1e-2, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr":lr, "beta": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            weight_decay = group['weight_decay']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p.grad))
                v = state.get("v", torch.zeros_like(p.grad))
                m = beta[0] * m + (1 - beta[0]) * p.grad.data
                v = beta[1] * v + (1 - beta[1]) * p.grad.data ** 2
                t = state.get("t",1)
                lr_t = lr * ((1 - beta[1] ** t) ** 0.5) / (1 - beta[0] ** t)
                p.data -= lr_t * m / (v ** 0.5 + eps)
                p.data -= lr * weight_decay * p.data
                state["m"] = m
                state["v"] = v
                # 注意这里要自增1
                state["t"] = t + 1

def My_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        lr = (it / warmup_iters) * max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        lr = min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    elif it > cosine_cycle_iters:
        lr = min_learning_rate
    return lr

def My_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # 注意l2 norm的计算方法是平方和开根号，并且是在梯度上做计算，不是参数本身
    total_norm = 0
    for param in parameters:
        if param.grad is not None:
            total_norm += ((param.grad.data) ** 2).sum().item()
    total_norm = total_norm ** 0.5
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad.data = param.grad.data * scale
    return total_norm * scale if total_norm > max_l2_norm else total_norm

def My_get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str = "cpu"):
    """
    Samples a batch of training data from a token stream.
    
    Args:
        x (np.ndarray): A 1D numpy array of token IDs.
        batch_size (int): Number of samples in the batch.
        context_length (int): Length of each input sequence.
        device (str): Device to put tensors on ('cpu', 'cuda', 'mps').

    Returns:
        input_batch (torch.Tensor): (batch_size, context_length)
        target_batch (torch.Tensor): (batch_size, context_length)
    """
    corpus_len = len(x)
    random_selected_point = np.random.randint(0, corpus_len - context_length, batch_size)
    random_selected_point_plus = random_selected_point + 1
    input_batch = [x[point:point + context_length] for point in random_selected_point]
    input_batch = np.stack(input_batch, axis=0)
    input_batch = torch.tensor(input_batch, device=device, dtype=torch.long)
    target_batch = [x[point:point + context_length] for point in random_selected_point_plus]
    target_batch = np.stack(target_batch, axis=0)
    target_batch = torch.tensor(target_batch, device=device, dtype=torch.long)
    return input_batch, target_batch

def My_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str,
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(save_dict, out)
def My_load_checkpoint(
    src: str ,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    state_dict = torch.load(src,map_location="cpu")
    model.load_state_dict(state_dict['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    return state_dict['iteration']
if __name__ == "__main__":
    dataset = np.arange(0, 100)
    context_length = 7
    batch_size = 32
    device = "cpu"
    My_get_batch(dataset, batch_size, context_length, device)
