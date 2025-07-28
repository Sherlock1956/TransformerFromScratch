import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from modules.modules import *
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
device = config['device']
# 1. prepare model
transformer_lm = My_transformer_lm(vocab_size=config['vocab_size'], context_length=config['context_length'], d_model=config['d_model'], num_layers=config['num_layers'], num_heads=config['num_heads'], d_ff=config['d_ff'], rope_theta=config['rope_theta'])
transformer_lm.to(device)
trainable_parameters = sum(param.numel() for param in transformer_lm.parameters() if param.requires_grad)
print(f"Transformer lm has {trainable_parameters / 1e9} B trainable parameters")
# 2. prepare data
train_data = np.load(config['train_data_path'], mmap_mode='r')
val_data = np.load(config['val_data_path'])
# 3. optimizer
optimizer = My_AdamW(transformer_lm.parameters(), lr=config['max_lr'])
# 4. training loop
writer = SummaryWriter(os.path.join(config['output_root_path'], 'logs'))
pbar = tqdm(total = config['total_iterations'])
cosine_iters = int(config['total_iterations'])
if not os.path.isdir(config['output_root_path']):
    os.mkdir(config['output_root_path'])
config_save_path = os.path.join(config['output_root_path'],'config.json')
json.dump(config,open(config_save_path, 'w'),ensure_ascii=False,indent=4)
for i in range(int(config['total_iterations'])):
    lr = My_lr_cosine_schedule(i, config['max_lr'], config['min_lr'], config['warmup_iters'], config['total_iterations'])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    batched_data_x, batched_data_y = My_get_batch(train_data, config['batch_size'], config['context_length'], device=device)
    batched_data_y_hat = transformer_lm(batched_data_x)
    loss = My_cross_entropy(batched_data_y_hat, batched_data_y)
    optimizer.zero_grad()
    loss.backward()
    gradient_norm = My_gradient_clipping(transformer_lm.parameters(), config['max_grad_norm'])
    optimizer.step()
    writer.add_scalar("train/loss", loss.item(), i)
    writer.add_scalar("train/lr", lr, i)
    writer.add_scalar("train/grad_norm", gradient_norm, i)
    pbar.update(1)
    if i % 10000 == 0:
        save_path = os.path.join(config['output_root_path'],f"checkpoint_{i}")
        My_save_checkpoint(transformer_lm, optimizer, i, save_path)
    if i % config['val_every'] == 0:
        loss_list = []
        for _ in tqdm(range(100)):
            batched_data_x, batched_data_y = My_get_batch(val_data, 8, config['context_length'], device=device)
            with torch.no_grad():
                batched_data_y_hat = transformer_lm(batched_data_x)
            loss = My_cross_entropy(batched_data_y_hat, batched_data_y)
            loss_list.append(loss.item())
        val_loss = sum(loss_list) / len(loss_list)
        print(f"Iteration: {i}, validation loss: {val_loss}")
        writer.add_scalar("val/loss", val_loss, i)
save_path = os.path.join(config['output_root_path'],f"checkpoint_{config['total_iterations']}")
My_save_checkpoint(transformer_lm, optimizer, i, save_path)
pbar.close()
writer.close()
