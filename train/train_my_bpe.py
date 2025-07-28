import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.modules import My_train_bpe

if __name__ == '__main__':
    data_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ['<|endoftext|>']
    My_train_bpe(input_path=data_path, vocab_size=vocab_size, special_tokens=special_tokens)