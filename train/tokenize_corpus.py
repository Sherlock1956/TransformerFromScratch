import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.modules import My_tokenizer
import numpy as np
if __name__ == "__main__":
    data_path = ["data/TinyStoriesV2-GPT4-train.txt","data/TinyStoriesV2-GPT4-valid.txt"]
    save_path = ["data/TinyStoriesV2-GPT4-train.txt.tokenized.npy","data/TinyStoriesV2-GPT4-valid.txt.tokenized.npy"]
    all_ids = []
    tokenizer = My_tokenizer(special_tokens=['<|endoftext|>'])
    for i in range(len(data_path)):
        with open(data_path[i], 'r') as f:
            for token in tokenizer.encode_iterable(f):
                all_ids.append(token)
        all_ids = np.array(all_ids, dtype=np.uint16)
        np.save(save_path[i], all_ids)
    