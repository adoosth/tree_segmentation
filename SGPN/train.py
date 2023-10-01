import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import tqdm
import torch.nn as nn
import torch.nn.init as init
import json
import sys
import os
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
#from dataloader import get_dataloader
from dataloader import get_dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
from architecture.SGPN import SGPN, SGPNLoss

if __name__ == '__main__':
    # for farthest point sampling
    #torch.multiprocessing.set_start_method('spawn')
    if len(sys.argv) > 1:
        config = sys.argv[1]
        with open(config) as f:
            config = json.load(f)
        config_filename = os.path.basename(sys.argv[1])
        config_filename = os.path.splitext(config_filename)[0]
        model_basename = config['model_basename'] + "_" + config_filename
        data_dir = config['data_dir']
        dataset = config['dataset']
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        alpha = config['alpha']
        alpha_step = config['alpha_step']
        lr = config['lr']
        lr_step = config['lr_step']
        save_step = config['save_step']
        margin = config['margin']
        epochs = config['epochs']
        latent_dims = config['latent_dims']
    else:
        print("Usage: python local_train.py <config>")
        exit(0)
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', data_dir)
    print("loading data... from, ", data_dir)
    dataloader = get_dataloader(data_dir, dataset, batch_size, num_workers, max_trees = 50, point_count = 4096, sampling='random', preload=False, center=True, normalize=True, select_trees = None, annotated = True, aug_rot = False)
    model = SGPN(latent_dims=latent_dims).cuda()
    optimizer = optim.Adam([{'params': model.parameters(), 'lr':lr}])
    loss_fn = SGPNLoss()

    writer = SummaryWriter('./runs/' + model_basename)
    print("training...")
    p_start = 1
    for p in range(p_start,epochs + 1):
        lost = []
        for i, (input_tensor, targets) in tqdm.tqdm(enumerate(dataloader)):
            dataloader.dataset.set_epoch(p)
            input_tensor = input_tensor.cuda().float()
            target = targets.cuda().float()
            if (input_tensor.shape[1] > 1000):
                if (input_tensor.shape[1] > 4096):
                    idx = np.random.choice(input_tensor.shape[1], 4096, replace=False)
                    input_tensor = input_tensor[:, idx, :]
                    ## Add noise
                    ## input_tensor = input_tensor + torch.randn(input_tensor.shape).cuda() * 0.00002
                    target = target[:, idx]
                optimizer.zero_grad()
                l0_points, Fsim, conf = model(input_tensor)
                loss = loss_fn(l0_points, Fsim, conf, target, alpha, margin)
                if i % 50 == 0:
                    print("Loss: ", loss.item())
                lost.append(loss.item())
                loss.backward()
                optimizer.step()
            else:
                print("Warning: not enough points. Skipping...")
        mean_loss = torch.tensor(lost).cuda().mean()
        if (p % lr_step == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2
        if (p % alpha_step == 0 and p != 0):
            alpha = alpha + 2.0
        if (p % save_step == 0):
            torch.save(model.cpu(), "./models/" + model_basename + "_" + str(p) + ".ckpt")
            model.cuda()
        print("Epoch ", p, " loss: ", np.array(lost).mean())
        writer.add_scalar('Loss SGPNLoss ', np.array(lost).mean(), p)
    torch.save(model.cpu(), "./models/" + model_basename + ".ckpt")
