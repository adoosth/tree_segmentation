import open3d as opend
import torch
import numpy as np
from architectures.SGPN import SGPN
from architectures.SGPN_utils import SGPNLoss
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
import socket
from dataloader import get_dataloader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def init():
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

if __name__ == '__main__':
    print("Received sys.argv ")
    print(sys.argv)
    forests = 'all'
    batch_size = 10
    num_workers = 4
    alpha_step = 5
    if len(sys.argv) > 3:
        architecture = sys.argv[1]
        data_dir = os.path.join("../data/", sys.argv[2])
        model_path = None
        in_world_size = int(sys.argv[3])
        if len(sys.argv) > 4:
            model_basename = sys.argv[4]
            model_filename = model_basename + '.ckpt'
            model_path = os.path.join('./' + architecture + '/models/', model_filename)
            p_start = 1
            try:
                p_start = int(model_basename.split('_')[-1])
            except:
                print("Couldn't extract epoch")
            if not os.path.exists(model_path):
                print("Warning: model not found: ", model_path)
        if not model_path:
            p_start = 1
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
            model_basename = '{}_{}'.format(timestamp, architecture)
            model_path = os.path.join('./' + architecture + '/models/', model_filename)
    else:
        print("Usage: python cluster_train.py <architecture> <data_dir> <world_size> [model_basename]")
        exit(0)
    
    init()
    # Get cluster parameters
    world_size = int(os.environ['SLURM_NPROCS'])
    world_rank = int(os.environ['SLURM_PROCID'])
    gpus_per_node = torch.cuda.device_count()
    local_rank = world_rank - gpus_per_node * (world_rank // gpus_per_node)
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    assert in_world_size == world_size

    # Set local GPU device
    torch.cuda.set_device(local_rank)

    if world_rank == 0 and not os.path.exists(model_path):
         model = SGPN(num_classes=10, latent_dims=3, margin=0.8).cuda()
         torch.save(model.cpu(), model_path)
         print("Saved to " + model_path)

    # Initiate cluster process
    print("Rank %d on %s using local GPU #%d initialized. Waiting" % (world_rank, hostname, local_rank))
    dist.init_process_group("nccl", rank=world_rank, world_size = world_size)
    print("Wait finished")
    if world_rank == 0:
        print("All nodes online. Start!")
    
    model = DDP(torch.load(model_path).to(local_rank), device_ids=[local_rank])#, find_unused_parameters=True)
    
    dataloader = get_dataloader(data_dir + "/train", forests, batch_size, num_workers, max_trees = 50, point_count = 4096, sampling='random', preload=False, distributed = True, world_size = world_size, world_rank = world_rank)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.00005}])
    epochs = 1000
    alpha = 2.0
    loss_fn = SGPNLoss()
    if (world_rank == 0):
        writer = SummaryWriter('./' + architecture + '/runs/' + model_basename)
    print("training...")
    for p in range(p_start,epochs + 1):
        dataloader.sampler.set_epoch(p)
        print("Rank ", world_rank, " Epoch ", p, " starting")
        lost = []
        for i, (input_tensor, targets) in enumerate(dataloader):
            #input_tensor = input_tensor.cuda(non_blocking=True).float()
            #target = targets.cuda(non_blocking=True).float()
            # Get data without issues
            input_tensor = input_tensor.to(local_rank).float()
            target = targets.to(local_rank).float()
            if i % 5 == 0:
                print("Rank ", world_rank, " Epoch ", p, " batch ", i)
            if (input_tensor.shape[1] > 1000):
                if (input_tensor.shape[1] > 4096):
                    idx = np.random.choice(input_tensor.shape[1], 4096, replace=False)
                    input_tensor = input_tensor[:, idx, :]
                    ## Add noise
                    ## input_tensor = input_tensor + torch.randn(input_tensor.shape).cuda() * 0.00002
                    target = target[:, idx]
                l0_points, Fsim = model(input_tensor)
                loss = loss_fn(l0_points, Fsim, target, alpha = alpha)
                lost.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                print("Warning: not enough points. Skipping...")
        mean_loss = torch.tensor(lost).cuda().mean()
        #torch.distributed.all_reduce(correct, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(mean_loss, op=torch.distributed.ReduceOp.AVG)
        #mean_loss = 0
        print("Rank ", world_rank, " Finishing up")
        if (p % 20 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2
        if (p % alpha_step == 0 and p != 0):
            alpha = alpha + 2.0
        if (p % 2 == 0 and world_rank == 0):
            # Save DistributedDataParallel model in a way that can be loaded in a non distributed context on a single GPU
            torch.save(model.module, "./" + architecture + "/models/" + model_basename + "_{}.ckpt".format(p))
            #print("Saving model...")
            #torch.save(model, "./" + architecture + "/models/" + model_basename + "_" + str(p) + ".ckpt")
            #print("Saving model state dict...")
            #torch.save(model.module.cpu().state_dict(), "./" + architecture + "/models/" + model_basename + "_" + str(p) + "_single_state_dict_m2.pt")

        if (world_rank == 0):
            print("Epoch ", p, " loss: ", mean_loss)
            writer.add_scalar('Loss simmat_loss ', mean_loss, p)
            writer.flush()
        #dist.barrier()
    torch.save(model.cpu(), "./" + architecture + "/models/" + model_basename + ".ckpt")
