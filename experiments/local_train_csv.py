import torch
import numpy as np
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
#from dataloader import get_dataloader
from dataloader_csv import get_dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
from architectures.SGPN_conf import SGPN, SGPNLoss

if __name__ == '__main__':
    # for farthest point sampling
    #torch.multiprocessing.set_start_method('spawn')
    if len(sys.argv) > 2:
        dataset = None
        model_basename = sys.argv[1]
        architecture = model_basename.split('_')[0]
        model_filename = model_basename + '.ckpt'
        p_start = 1
        try:
            p_start = int(model_basename.split('_')[-1])
        except:
            print("Couldn't extract epoch")
        data_dir = os.path.join("../data/", sys.argv[2])
        model_path = os.path.join('./' + architecture + '/models/', model_filename)
        if len(sys.argv) > 3:
            dataset = sys.argv[3]
        else:
            print("Warning: no dataset specified. Using default train.")
            dataset = "train"
        if not model_path:
            p_start = 1
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
            model_basename = '{}_{}'.format(timestamp, architecture)
            model_path = os.path.join('./' + architecture + '/models/', model_filename)
    else:
        print("Usage: python local_train_csv.py <model_basename> <data_dir>  [dataset]")
        exit(0)

    # Sanity
    # batch_size = 4
    # num_workers = 8
    # alpha = 2
    # alpha_step = 30
    # lr = 0.002 # 0.00005
    # lr_step = 50 # 20
    # save_step = 10
    # margin = 0.8
    # epochs = 1000
    # latent_dims = 4

    # One forest train
    batch_size = 4
    num_workers = 4
    alpha = 2
    alpha_step = 10
    lr = 0.0001 # 0.00005
    lr_step = 20 # 20
    save_step = 2
    margin = 0.8
    epochs = 1000
    latent_dims = 32

    dataset = "sanity" if not dataset else dataset
    print("loading data...")
    dataloader = get_dataloader(data_dir, dataset, batch_size, num_workers, max_trees = 50, point_count = 4096, sampling='random', preload=False, center=True, normalize=True, select_trees = None, annotated = True, aug_rot = False)
    model = SGPN(latent_dims=latent_dims).cuda()
    optimizer = optim.Adam([{'params': model.parameters(), 'lr':lr}])
    loss_fn = SGPNLoss()

    writer = SummaryWriter('./' + architecture + '/runs/' + model_basename)
    print("training...")
    for p in range(p_start,epochs + 1):
        lost = []
        for i, (input_tensor, targets) in tqdm.tqdm(enumerate(dataloader)):
            #print("I got the data")
            dataloader.dataset.set_epoch(p)
            input_tensor = input_tensor.cuda().float()
            target = targets.cuda().float()
            #print("Sending it to cuda")
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
            torch.save(model.cpu(), "./" + architecture + "/models/" + model_basename + "_" + str(p) + ".ckpt")
            model.cuda()
        print("Epoch ", p, " loss: ", np.array(lost).mean())
        writer.add_scalar('Loss SGPNLoss ', np.array(lost).mean(), p)
    torch.save(model.cpu(), "./" + architecture + "/models/" + model_basename + ".ckpt")
