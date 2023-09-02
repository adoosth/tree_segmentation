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
import socket
from dataloader import get_dataloader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

model = (SGPN(num_classes=10, latent_dims=3, alpha_step = 400, margin=0.8))