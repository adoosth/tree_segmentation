import open3d as opend
import torch
import numpy as np
from architectures.SGPN import SGPN
import torch.optim as optim
import torch.nn as nn
import tqdm
import torch.nn as nn
import torch.nn.init as init
import sys
import os
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
from dataloader import get_dataloader
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == '__main__':
    if len(sys.argv) > 1:
        architecture = sys.argv[1]
        data_dir = os.path.join("../data/", sys.argv[2])
        model = None
        if len(sys.argv) > 3:
            model_basename = sys.argv[3]
            model_filename = model_basename + '.ckpt'
            model_path = './' + architecture + '/models/' + model_filename
            p_start = model_basename.split('_')[3] + 1
            if os.path.exists(model_path):
                model = torch.load(model_path)
            else:
                print("Warning: model not found: ", model_path)
        if not model:
            p_start = 1
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
            model_basename = '{}_{}'.format(timestamp, architecture)
    else:
        print("Usage: python train.py <architecture> <data_dir>")
        exit(0)
    
    seed = list(map(ord, 'toto'))
    seed = map(str, seed)
    seed = ''.join(seed)
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(1)

    torch.set_num_threads(1)
    OMP_NUM_THREADS=1
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    print("loading data...")
    forests = 'all'
    batch_size = 5
    num_workers = 8
    dataloader = get_dataloader(data_dir + "/train", forests, batch_size, num_workers, max_trees = 50, point_count = 4096, sampling='random', preload=False)
    model = SGPN(num_classes=10, latent_dims=3, alpha_step = 400, margin=0.8).cuda()
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.00005}])
    epochs = 1000

    writer = SummaryWriter('./' + architecture + '/runs/' + model_basename)
    print("training...")
    for p in range(p_start,epochs + 1):
        lost = []
        for i, (input_tensor, targets) in tqdm.tqdm(enumerate(dataloader)):
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
                loss = model(input_tensor, target, training=True, epoch=p)
                lost.append(loss.item())
                loss.backward()
                optimizer.step()
            else:
                print("Warning: not enough points. Skipping...")
        if (p % 20 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2
        if (p % 5 == 0):
            torch.save(model.cpu(), "./" + architecture + "/models/" + model_basename + "_" + str(p) + ".ckpt")
            model.cuda()
        print("Epoch ", p, " loss: ", np.array(lost).mean())
        writer.add_scalar('Loss simmat_loss ', np.array(lost).mean(), p)
    torch.save(model.cpu(), "./" + architecture + "/models/" + model_basename + ".ckpt")
