import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import laspy
import numpy as np
import pandas as pd
from copy import deepcopy
from torch.utils.data.distributed import DistributedSampler
from architecture.net_utils import farthest_point_sample_gpu

# define Dataset
class ForestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv, preload=True, max_trees = 100, center=True, normalize=True, point_count=4096, sampling='random', select_trees = None, annotated = None, aug_rot = False, chunk_size = 4000000, max_recent_tiles = 100, max_recent_boxes = 10):
        self.data_dir = data_dir
        self.preload = preload
        self.point_count = point_count
        self.sampling = sampling
        self.max_trees = max_trees
        self.select_trees = select_trees
        self.normalize = normalize
        self.center = center
        self.annotated = annotated
        self.has_aug_rot = aug_rot
        self.max_recent_tiles = max_recent_tiles
        self.recent_tiles = dict()
        self.max_recent_boxes = max_recent_boxes
        self.recent_boxes = dict()
        #print("Loading {} forests from {}".format(len(self.forests), data_dir))
        csv_path = os.path.join(data_dir, csv + '.csv')
        if not os.path.exists(csv_path):
            print("Error: csv not found: ", csv_path)
            exit(0)
        self.full_dataset = pd.read_csv(csv_path)
        self.set_epoch(1)
        self.forests = self.boxes.forest_id.unique()
        self.index_tiles()
        #print(self.forests)
        if self.preload:
            self.forest_data = []
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
                target = np.concatenate(target, axis=0)
                print("Loaded {} points from {}".format(xyz.shape[0], laz_path))
                self.forest_data.append((xyz, target))

    def index_tiles(self):
        self.forest_tiles = pd.DataFrame(columns=['forest_id', 'tile', 'minx', 'maxx', 'miny', 'maxy'])
        for forest_id in self.forests:
            tiles_dir = os.path.join(self.data_dir, forest_id)
            tiles = os.listdir(tiles_dir)
            tiles.sort()
            for tile in tiles:
                with laspy.open(os.path.join(tiles_dir, tile)) as f:
                    metadata = f.header
                    # append to dataframe
                    self.forest_tiles = self.forest_tiles.append({'forest_id': forest_id, 'tile': tile, 'minx': metadata.min[0], 'maxx': metadata.max[0], 'miny': metadata.min[1], 'maxy': metadata.max[1]}, ignore_index=True)
            print("Indexed {} tiles for forest {}".format(len(tiles), forest_id))
            print(self.forest_tiles.tail(5))

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.boxes = self.full_dataset[self.full_dataset.epoch == epoch]
        # drop epoch column
        self.boxes = self.boxes.drop(columns=['epoch'])
        
    def center_xyz(self, xyz):
        return xyz - np.mean(xyz, axis=0)
    
    def normalize_xyz(self, xyz):
        return xyz / np.max(np.abs(xyz))
    
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
            try:
                idx = np.random.choice(xyz.shape[0], self.point_count, replace=False)
            except:
                # append 0s to xyz and target
                print("WARNING: Not enough points in box: ", xyz.shape[0], " at epoch ", self.epoch, " and index ", index)
                xyz = np.concatenate((xyz, np.zeros((self.point_count - xyz.shape[0], 3))), axis=0)
            return xyz[idx, :], target[idx]
        elif self.sampling == 'farthest':
            idx = farthest_point_sample_gpu(torch.from_numpy(xyz.astype(np.float32)).unsqueeze(0).cuda(), self.point_count)[0].squeeze(0).cpu().numpy()
            return xyz[idx, :], target[idx]

    def aug_rot(self, xyz, target):
        # randomly rotate around z axis
        angle = np.random.rand() * 2 * np.pi
        xyz = np.dot(xyz, np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]))
        return xyz, target
    
    def tiles_to_read(self, forest_id, x, y, width, height):
        bounds = [(x, x + width), (y, y + height)]
        # filter tiles without any overlap
        tiles = self.forest_tiles[self.forest_tiles.forest_id == forest_id]
        tiles = tiles[(tiles.minx < bounds[0][1]) & (tiles.maxx > bounds[0][0]) & (tiles.miny < bounds[1][1]) & (tiles.maxy > bounds[1][0])]
        return [os.path.join(self.data_dir, forest_id, tile) for tile in tiles.tile]    


    def load_lazy(self, forest_id, x, y, width, height):
        # Define the bounding box [minx, maxx, miny, maxy]
        bounds = [(x, x + width), (y, y + height)]
        all_xyz = []
        all_target = []
        tiles = self.tiles_to_read(forest_id, x, y, width, height)
        if len(tiles) == 0:
            print("No tiles to read for bounding box: ", bounds)
            exit(1)
        #print("Reading %d tiles" % len(tiles), end="... ")
        for tile in tiles:
            if tile not in self.recent_tiles.keys():
                las = laspy.read(tile)
                xyz = las.xyz
                try:
                    target = las.hitObjectId if self.annotated else np.zeros(las.xyz.shape[0])
                except:
                    target = las.point_source_id if self.annotated else np.zeros(las.xyz.shape[0])
                self.recent_tiles[tile] = (xyz, target)
                # filter points outside of bounding box
                idx = np.logical_and(np.logical_and(xyz[:,0] >= bounds[0][0], xyz[:,0] <= bounds[0][1]), np.logical_and(xyz[:,1] >= bounds[1][0], xyz[:,1] <= bounds[1][1]))
                xyz = xyz[idx, :]
                target = target[idx]
                all_xyz.append(xyz)
                all_target.append(target)
                if len(self.recent_tiles) > self.max_recent_tiles:
                    self.recent_tiles.popitem()
            else:
                xyz, target = self.recent_tiles[tile]
                all_xyz.append(xyz)
                all_target.append(target)
        #print("Done")
        xyz = np.concatenate(all_xyz, axis=0)
        target = np.concatenate(all_target, axis=0)
        if len(target) < 4096:
            print("Bounding box ERROR: ", bounds)
            print("Too few points found in tiles: ", tiles, " points: ", len(target))
            exit(1)
        return xyz, target

    def __getitem__(self, index):
        forest_id, x, y, width, height = self.boxes.iloc[index]
        #print("Forest id: " + str(forest_id) + " bbox: " + str(bbox))
        try:
            if not self.preload:
                # load boxes with pdal
                if (forest_id, x, y, width, height) not in self.recent_boxes.keys():
                    xyz, target = self.load_lazy(forest_id, x, y, width, height)
                    #self.recent_boxes[(forest_id, x, y, width, height)] = (xyz, target)
                    #if len(self.recent_boxes) > self.max_recent_boxes:
                    #    self.recent_boxes.popitem()
                else:
                    print("Found box in recent boxes")
                    xyz, target = self.recent_boxes[(forest_id, x, y, width, height)]
            else:
                # find forest in numpy array
                print("Loading forest tile from memory")    
                xyz, target = self.forest_data[np.where(self.forests == forest_id)[0][0]]
        except:
            print("Error loading forest: ", forest_id, " at epoch ", self.epoch, " and index ", index)
            print("x: {}, y: {}, width: {}, height: {}".format(x, y, width, height))
            exit(1)
        assert xyz.shape[0] > 0, "No points found in BEFORE final filtering : " + str((forest_id, x, y, width, height)) + " at epoch " + str(self.epoch) + " and index " + str(index)
        idx = np.where((xyz[:,0] >= x) & (xyz[:,0] < x + width) & (xyz[:,1] >= y) & (xyz[:,1] < y + height))[0]
        if len(idx) == 0:
            print("ERROR DEBUG")
            print("x: {}, y: {}, width: {}, height: {}".format(x, y, width, height))
            print("x_min" , np.min(xyz[:,0]))
            print("x_max" , np.max(xyz[:,0]))
            print("y_min" , np.min(xyz[:,1]))
            print("y_max" , np.max(xyz[:,1]))
        xyz, target = xyz[idx, :], target[idx]
        assert xyz.shape[0] > 0, "No points found in box: " + str((forest_id, x, y, width, height)) + " at epoch " + str(self.epoch) + " and index " + str(index)
        xyz, target = self.sample_trees(xyz, target)
        xyz = self.center_xyz(xyz) if self.center else xyz
        xyz = self.normalize_xyz(xyz) if self.normalize else xyz
        xyz, target = self.sample_points(xyz, target)
        if self.has_aug_rot:
            xyz, target = self.aug_rot(xyz, target)
        return xyz, target

    def __len__(self):
        return len(self.boxes)
    

# define DataLoader
def get_dataloader(data_dir, csv, batch_size = 8, num_workers = 8, max_trees = 100, point_count = 4096, sampling='random', preload=True, center=True, normalize=True, select_trees = None, annotated = None, aug_rot = False, chunk_size = 4000000, distributed=False, world_size=1, world_rank=0, shuffle=True):
    dataset = ForestDataset(data_dir, csv = csv, preload = preload, max_trees=max_trees, center=center, normalize=normalize, point_count=point_count, sampling=sampling,  select_trees = select_trees,  annotated = annotated,aug_rot = aug_rot, chunk_size=chunk_size)
    if distributed:
        data_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=world_rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=data_sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader