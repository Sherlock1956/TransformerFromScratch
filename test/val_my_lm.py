import torch
import numpy as np
from tqdm import tqdm
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.modules import *


# 1. prepare model
model_path = "models/test_6/checkpoint_80000"
config_path = os.path.join(os.path.dirname(model_path),'config.json')
config = json.load(open(config_path,'r'))
device = config['device']
transformer_lm = My_transformer_lm(vocab_size=config['vocab_size'], context_length=config['context_length'], d_model=config['d_model'], num_layers=config['num_layers'], num_heads=config['num_heads'], d_ff=config['d_ff'], rope_theta=config['rope_theta'])
trainable_parameters = sum(param.numel() for param in transformer_lm.parameters() if param.requires_grad)
print(f"Transformer lm has {trainable_parameters / 1e9} B trainable parameters")
My_load_checkpoint(model_path,transformer_lm,None)
transformer_lm.to(device)

# 2. prepare data
data = np.load(config['val_data_path'])

# 3. generation
loss_list = []
for i in tqdm(range(100)):
    batched_data_x, batched_data_y = My_get_batch(data, 16, config['context_length'], device=device)
    with torch.no_grad():
        batched_data_y_hat = transformer_lm(batched_data_x)
    loss = My_cross_entropy(batched_data_y_hat, batched_data_y)
    loss_list.append(loss.item())
print(f"average loss: {sum(loss_list) / len(loss_list)}")



