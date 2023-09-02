import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import laspy
import numpy as np

# define Dataset
class ForestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, forests, preload=True, max_trees = 10, center=True, normalize=True, point_count=4096, sampling='random', select_trees = None, annotated = None, bbox = None):
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
        self.bbox = bbox
        if forests == 'all':
            self.forests = [os.path.basename(f)[:-4] for f in os.listdir(data_dir) if f.endswith('.laz') or f.endswith('.las')]
        print("Loading {} forests from {}".format(len(self.forests), data_dir))
        print(self.forests)
        if self.preload:
            self.data = []
            for forest_id in self.forests:
                laz_path = os.path.join(self.data_dir, forest_id + '.laz')
                if not os.path.exists(laz_path):
                    laz_path = os.path.join(self.data_dir, forest_id + '.las')
                las = laspy.read(laz_path)
                # select trees
                if select_trees is not None:
                    idx = np.isin(las.hitObjectId, self.select_trees)
                    las = las[idx]
                xyz = las.xyz
                #print("Loaded {} points from {}".format(las.xyz.shape[0], laz_path))
                #print("Bounding box before centering: ", las.header.min, las.header.max)
                # center points
                if center:
                    xyz = xyz - np.mean(xyz, axis=0)
                #print("Bounding box after centering: ", np.min(xyz, axis=0), np.max(xyz, axis=0))
                if bbox is not None:
                    #print("Selecting points in bounding box {}".format(bbox))
                    idx = np.logical_and(np.logical_and(xyz[:,0] >= bbox[0], xyz[:,0] <= bbox[3]), np.logical_and(xyz[:,1] >= bbox[1], xyz[:,1] <= bbox[4]))
                    idx = np.logical_and(idx, np.logical_and(xyz[:,2] >= bbox[2], xyz[:,2] <= bbox[5]))
                    las = las[idx]
                # normalize
                if normalize:
                    xyz = xyz / np.max(np.abs(xyz))
                if annotated is None:
                    print("Determining if forest {} is annotated".format(forest_id))
                    try:
                        target = las.hitObjectId
                        annotated = True
                    except:
                        target = np.zeros(xyz.shape[0])
                        annotated = False
                if annotated:
                    target = las.hitObjectId
                else:
                    target = np.zeros(xyz.shape[0])
                self.data.append((xyz, target))

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
            # calculate distance to nearest neighbor
            # select point with max distance
            # repeat
            # TODO
            pass

    def __getitem__(self, index):
        forest_id = self.forests[index]
        xyz, target = self.load_laz(forest_id)
        xyz, target = self.sample_trees(xyz, target)
        xyz, target = self.sample_points(xyz, target)
        return xyz, target

    def __len__(self):
        return len(self.forests)
    

# define DataLoader
def get_dataloader(data_dir, forests, batch_size, num_workers, max_trees = 10, point_count=4096, sampling='random', select_trees = None, annotated = None, bbox = None, preload=True):
    dataset = ForestDataset(data_dir, forests, max_trees=max_trees, point_count=point_count, sampling=sampling, select_trees = select_trees, annotated = annotated, bbox = bbox, preload=preload)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader