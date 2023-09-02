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

def get_boxe_from_p(boxes):
    vertices = []
    lines = np.array( [[0,0]])
    i = 0
    for b in boxes:
        first_p = b[1]
        second_p = b[0]
        width = first_p[0] - second_p[0]
        height = first_p[1] - second_p[1]
        depth = first_p[2] - second_p[2]

        vertices.append(first_p) # top front right
        vertices.append(first_p-[width,0,0]) # top front left
        vertices.append(first_p-[width,height,0]) # bottom front left
        vertices.append(first_p-[0,height,0]) # botton front right

        vertices.append(second_p) # bottom back left
        vertices.append(first_p-[width,0,depth]) # top back left
        vertices.append(first_p-[0,height,depth]) # bottom back right
        vertices.append(first_p-[0,0,depth]) # top back right

        edges = [[0+(i*8),1+(i*8)],[1+(i*8),2+(i*8)],[2+(i*8),3+(i*8)],[3+(i*8),0+(i*8)]
                ,[4+(i*8),5+(i*8)],[4+(i*8),6+(i*8)],[6+(i*8),7+(i*8)],[7+(i*8),5+(i*8)]
                ,[0+(i*8),7+(i*8)],[1+(i*8),5+(i*8)],[4+(i*8),2+(i*8)],[3+(i*8),6+(i*8)]]
        lines = np.concatenate([lines,edges],axis = 0)
        i = i+1


    line_set = opend.geometry.LineSet()
    line_set.points = opend.utility.Vector3dVector(vertices)
    line_set.lines = opend.utility.Vector2iVector(lines[1:])
    line_set.colors = opend.utility.Vector3dVector([[1, 0, 0] for i in range(lines[1:].shape[0])])
        # i = i + 1

    return line_set

if __name__ == '__main__':
    if len(sys.argv) > 1:
        architecture = sys.argv[1]
        data_dir = os.path.join("../data/", sys.argv[2])
        model = None
        if len(sys.argv) > 3:
            model_basename = sys.argv[3]
            model_filename = model_basename + '.ckpt'
            model_path = './' + architecture + '/models/' + model_filename
            p_start = int(model_basename.split('_')[3]) + 1
            model_basename = '_'.join(model_basename.split('_')[:3])
            if os.path.exists(model_path):
                print("loading model: ", model_path)
                model = torch.load(model_path).cuda()
            else:
                print("Warning: model not found: ", model_path)
        if not model:
            p_start = 1
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
            model_basename = '{}_{}'.format(timestamp, architecture)
    else:
        print("Usage: python train.py <architecture> <data_dir> [model_basename]")
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

    # reshape and add batch dim
    # points = points.reshape((1, points.shape[0], points.shape[1]))
    # targets = targets.reshape((1, targets.shape[0]))
    #points, targets = torch.rand(100, 10000, 3).numpy(), torch.randint(0, 2, (100, 10000)).numpy()
    if not model:
        print("creating model...")
        model = SGPN(num_classes=10, latent_dims=3, alpha_step = 5, margin=0.8).cuda()
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.00005}])
    epochs = 1000

    writer = SummaryWriter('./' + architecture + '/runs/' + model_basename)
    print("training...")
    for p in range(p_start,epochs + 1):
        lost = []
        for i, (input_tensor, targets) in tqdm.tqdm(enumerate(dataloader)):
            # input_tensor = torch.from_numpy(points[i]).float().unsqueeze(0).cuda()
            # target = torch.from_numpy(targets[i]).unsqueeze(0).cuda().float()
            input_tensor = input_tensor.cuda().float()
            target = targets.cuda().float()
            if (input_tensor.shape[1] > 1000):
                # select 4096 points randomly
                if (input_tensor.shape[1] > 4096):
                    print("Warning: too many points. Selecting 4096 randomly...")
                    idx = np.random.choice(input_tensor.shape[1], 4096, replace=False)
                    input_tensor = input_tensor[:, idx, :]
                    # add noise
                    #input_tensor = input_tensor + torch.randn(input_tensor.shape).cuda() * 0.00002
                    target = target[:, idx]
                optimizer.zero_grad()
                loss = model(input_tensor, target, training=True, epoch=p)
                lost.append(loss.item())
                loss.backward()
                optimizer.step()
            else:
                print("Warning: not enough points. Skipping...")

        print("Epoch ", p, " loss: ", np.array(lost).mean())
        writer.add_scalar('Loss simmat_loss ', np.array(lost).mean(), p)
        if (p % 20 == 0 and p != 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2
        if (p % 5 ==0 and p !=0 ):
            torch.save(model.cpu(), "./" + architecture + "/models/" + model_basename + "_" + str(p) + ".ckpt")
            model.cuda()
    torch.save(model.cpu(), "./" + architecture + "/models/" + model_basename + ".ckpt")
