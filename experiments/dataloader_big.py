import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import laspy
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from architectures.net_utils import farthest_point_sample_gpu

# define Dataset
class ForestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, forests = 'all', preload=True, max_trees = 100, center=True, normalize=True, point_count=4096, sampling='random', select_trees = None, annotated = None, box_size=(40,40), aug_rot = False, box_cnt = None, forest_size = (1000, 1000), chunk_size = 4000000):
        self.data_dir = data_dir
        self.forests = forests
        self.preload = preload
        self.point_count = point_count
        self.sampling = sampling
        self.max_trees = max_trees
        self.select_trees = select_trees
        self.normalize = normalize
        self.center = center
        self.annotated = annotated
        self.box_cnt = box_cnt
        self.box_size = box_size
        self.has_aug_rot = aug_rot
        self.forest_size = forest_size
        if forests == 'all':
            self.forests = [os.path.basename(f)[:-4] for f in os.listdir(data_dir) if f.endswith('.laz') or f.endswith('.las')]
        # TODO REMOVE
        self.forests = self.forests[:1]
        print("Loading {} forests from {}".format(len(self.forests), data_dir))

        #print(self.forests)
        if self.preload:
            self.forest_data = []
            for forest_id in self.forests:
                print("Checking forest {}".format(forest_id))
                laz_path = os.path.join(self.data_dir, forest_id + '.laz')
                if not os.path.exists(laz_path):
                    laz_path = os.path.join(self.data_dir, forest_id + '.las')
                with laspy.open(laz_path) as f:
                    print("Total points: ", f.header.point_count)
                    print("Min: ", f.header.min)
                    print("Max: ", f.header.max)
                    cur_forest_size = (f.header.max[0] - f.header.min[0], f.header.max[1] - f.header.min[1])
                    self.forest_size = (min(forest_size[0], cur_forest_size[0]), min(forest_size[1], cur_forest_size[1]))
            # set boxes
            if box_cnt is not None:
                self.set_boxes(box_cnt, 0)
            for forest_id in self.forests:
                print("Loading forest {}".format(forest_id))
                laz_path = os.path.join(self.data_dir, forest_id + '.laz')
                if not os.path.exists(laz_path):
                    laz_path = os.path.join(self.data_dir, forest_id + '.las')
                xyz = []
                target = []
                annotated = True
                with laspy.open(laz_path) as f:  
                    for i, points in enumerate(f.chunk_iterator(chunk_size)):
                        # print las header information
                        #print("las header: ", f.header)
                        #print("las header min: ", f.header.min)
                        #print("las header max: ", f.header.max)
                        forest_size = (f.header.max[0] - f.header.min[0], f.header.max[1] - f.header.min[1])
                        #print("las header scale: ", f.header.scale)
                        #print("las header offset: ", f.header.offset)

                        points_xyz = np.array([points.x, points.y, points.z]).T
                        xyz.append(np.array(points_xyz))
                        if annotated:
                            try:
                                target.append(np.array(points.hitObjectId))
                            except:
                                try:
                                    target.append(np.array(points.point_source_id))
                                except:
                                    print("No hitObjectId or point_source_id in {}".format(laz_path))
                                    exit(1)
                        else:
                            target.append(np.zeros(points.xyz.shape[0]))
                        if i % 10 == 0:
                            print("Loaded {} points from {}".format(points_xyz.shape[0], laz_path))
                xyz = np.concatenate(xyz, axis=0)
                # center x,y but not z
                xyz[:,0] = xyz[:,0] - np.mean(xyz[:,0])
                xyz[:,1] = xyz[:,1] - np.mean(xyz[:,1])
                target = np.concatenate(target, axis=0)
                print("Loaded {} points from {}".format(xyz.shape[0], laz_path))
                self.forest_data.append((xyz, target))

                # del las so that it doesn't take up memory
                #del las

    def set_boxes(self, box_cnt, seed):
            np.random.seed(seed)
            box_forests = np.random.choice(len(self.forests), box_cnt, replace=True)
            # box forests should be int
            self.box_forests = box_forests.astype(int).reshape(-1)
            box_xys = np.random.rand(box_cnt, 2) * np.array(self.forest_size) - np.array(self.box_size) / 2
            # box_xys should be centered to forest center
            box_xys = box_xys - np.array(self.forest_size) / 2
            self.boxes = np.concatenate((box_xys, box_xys + np.array(self.box_size)), axis=1)

    def center_xyz(self, xyz):
        return xyz - np.mean(xyz, axis=0)
    
    def normalize_xyz(self, xyz):
        return xyz / np.max(np.abs(xyz))
    
    def load_laz(self, forest_id):
        if self.preload:
            return self.data[self.forests.index(forest_id)]
        laz_path = os.path.join(self.data_dir, forest_id + '.laz')
        if not os.path.exists(laz_path):
            laz_path = os.path.join(self.data_dir, forest_id + '.las')
        las = laspy.read(laz_path)
        #print("Loaded {} points from {}".format(las.xyz.shape[0], laz_path))
        # select trees
        if self.select_trees is not None:
            idx = np.isin(las.hitObjectId, self.select_trees)
            las = las[idx]
        xyz = las.xyz
        target = las.hitObjectId if self.annotated else np.zeros(las.xyz.shape[0])
        if self.center:
            #print("Bounding box before centering: ", las.header.min, las.header.max)
            xyz = las.xyz - np.mean(las.xyz, axis=0)
            #print("Bounding box after centering: ", np.min(xyz, axis=0), np.max(xyz, axis=0))
        if self.bbox is not None:
            print("Selecting points in bounding box {}".format(self.bbox))
            idx = np.logical_and(np.logical_and(xyz[:,0] >= self.bbox[0], xyz[:,0] <= self.bbox[3]), np.logical_and(xyz[:,1] >= self.bbox[1], xyz[:,1] <= self.bbox[4]))
            idx = np.logical_and(idx, np.logical_and(xyz[:,2] >= self.bbox[2], xyz[:,2] <= self.bbox[5]))
            xyz, target = xyz[idx, :], target[idx]
        if self.normalize:
            xyz = xyz / np.max(np.abs(las.xyz))
        if self.annotated is None:
            #print("Determining if forest {} is annotated".format(forest_id))
            try:
                target = las.hitObjectId
                annotated = True
            except:
                annotated = False
        if annotated:
            target = las.hitObjectId
        else:
            target = np.zeros(xyz.shape[0])
        return xyz, target
    
    def sample_trees(self, xyz, target):
        tree_ids = np.unique(target)
        if len(tree_ids) > self.max_trees:
            tree_ids = np.random.choice(tree_ids, self.max_trees, replace=False)
            #print("Selected trees {}".format(tree_ids))
        idx = np.isin(target, tree_ids)
        # reset indices by mapping to range [0, len(tree_ids)]
        new_target = np.zeros(target.shape)
        for i, tree_id in enumerate(tree_ids):
            new_target[target == tree_id] = i
        return xyz[idx, :], new_target[idx]
    
    def sample_points(self, xyz, target):
        if self.sampling == 'random':
            idx = np.random.choice(xyz.shape[0], self.point_count, replace=False)
            return xyz[idx, :], target[idx]
        elif self.sampling == 'farthest':
            idx = farthest_point_sample_gpu(torch.from_numpy(xyz.astype(np.float32)).unsqueeze(0).cuda(), self.point_count)[0].squeeze(0).cpu().numpy()
            return xyz[idx, :], target[idx]

    def aug_rot(self, xyz, target):
        # randomly rotate around z axis
        angle = np.random.rand() * 2 * np.pi
        xyz = np.dot(xyz, np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]))
        return xyz, target
    
    def takebox(self, xyz, target, bbox):
        idx = np.logical_and(np.logical_and(xyz[:,0] >= bbox[0], xyz[:,0] <= bbox[2]), np.logical_and(xyz[:,1] >= bbox[1], xyz[:,1] <= bbox[3]))
        #print("idx is " + str(idx))
        return xyz[idx, :], target[idx]

    def __getitem__(self, index):
        forest_id = self.box_forests[index]
        bbox = self.boxes[index]
        #print("Forest id: " + str(forest_id) + " bbox: " + str(bbox))
        if not self.preload:
            xyz, target = self.load_lazy(forest_id)
        else:
            xyz, target = self.forest_data[forest_id]
        xyz, target = self.takebox(xyz, target, bbox)
        xyz, target = self.sample_trees(xyz, target)
        xyz = self.center_xyz(xyz) if self.center else xyz
        xyz = self.normalize_xyz(xyz) if self.normalize else xyz
        xyz, target = self.sample_points(xyz, target)
        if self.has_aug_rot:
            xyz, target = self.aug_rot(xyz, target)
        return xyz, target

    def __len__(self):
        #return len(self.forests)
        return len(self.boxes)
    

# define DataLoader
def get_dataloader(data_dir, forests, batch_size, num_workers, max_trees = 10, point_count=4096, sampling='random', select_trees = None, annotated = None, bbox = None, preload=True, distributed=False, world_rank = 0, world_size = 1, aug_rot = True, box_cnt = None, box_size = (40,40), forest_size = (1000, 1000)):
    dataset = ForestDataset(data_dir, forests, max_trees=max_trees, point_count=point_count, sampling=sampling, select_trees = select_trees, annotated = annotated, preload=preload, aug_rot = aug_rot, box_cnt = box_cnt, box_size = box_size, forest_size = forest_size)
    if distributed:
        data_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=world_rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=data_sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader