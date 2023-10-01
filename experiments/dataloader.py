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

def sample_farthest_points(
    points,
    lengths = None,
    K = 50,
    random_start_point = False,
):
    """
    Iterative farthest point sampling algorithm [1] to subsample a set of
    K points from a given pointcloud. At each iteration, a point is selected
    which has the largest nearest neighbor distance to any of the
    already selected points.

    Farthest point sampling provides more uniform coverage of the input
    point cloud compared to uniform random sampling.

    [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        on Point Sets in a Metric Space", NeurIPS 2017.

    Args:
        points: (N, P, D) array containing the batch of pointclouds
        lengths: (N,) number of points in each pointcloud (to support heterogeneous
            batches of pointclouds)
        K: samples required in each sampled point cloud (this is typically << P). If
            K is an int then the same number of samples are selected for each
            pointcloud in the batch. If K is a tensor is should be length (N,)
            giving the number of samples to select for each element in the batch
        random_start_point: bool, if True, a random point is selected as the starting
            point for iterative sampling.

    Returns:
        selected_points: (N, K, D), array of selected values from points. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            0.0 for batch elements where k_i < max(K).
        selected_indices: (N, K) array of selected indices. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            -1 for batch elements where k_i < max(K).
    """
    N, P, D = points.shape
    device = points.device

    # Validate inputs
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)
    else:
        if lengths.shape != (N,):
            raise ValueError("points and lengths must have same batch dimension.")
        if lengths.max() > P:
            raise ValueError("Invalid lengths.")

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.int64, device=device)

    if K.shape[0] != N:
        raise ValueError("K and points must have the same batch dimension")

    # Find max value of K
    max_K = torch.max(K)

    # List of selected indices from each batch element
    all_sampled_indices = []

    for n in range(N):
        # Initialize an array for the sampled indices, shape: (max_K,)
        sample_idx_batch = torch.full(
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
            (max_K,),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )

        # Initialize closest distances to inf, shape: (P,)
        # This will be updated at each iteration to track the closest distance of the
        # remaining points to any of the selected points
        closest_dists = points.new_full(
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
            (lengths[n],),
            float("inf"),
            dtype=torch.float32,
        )

        # Select a random point index and save it as the starting point
        selected_idx = np.randint(0, lengths[n] - 1) if random_start_point else 0
        sample_idx_batch[0] = selected_idx

        # If the pointcloud has fewer than K points then only iterate over the min
        # pyre-fixme[6]: For 1st param expected `SupportsRichComparisonT` but got
        #  `Tensor`.
        # pyre-fixme[6]: For 2nd param expected `SupportsRichComparisonT` but got
        #  `Tensor`.
        k_n = min(lengths[n], K[n])

        # Iteratively select points for a maximum of k_n
        for i in range(1, k_n):
            # Find the distance between the last selected point
            # and all the other points. If a point has already been selected
            # it's distance will be 0.0 so it will not be selected again as the max.
            dist = points[n, selected_idx, :] - points[n, : lengths[n], :]
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            dist_to_last_selected = (dist**2).sum(-1)  # (P - i)

            # If closer than currently saved distance to one of the selected
            # points, then updated closest_dists
            closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

            # The aim is to pick the point that has the largest
            # nearest neighbour distance to any of the already selected points
            selected_idx = torch.argmax(closest_dists)
            sample_idx_batch[i] = selected_idx

        # Add the list of points for this batch to the final list
        all_sampled_indices.append(sample_idx_batch)

    idx = torch.stack(all_sampled_indices, dim=0)

    # Select the points based on the indices
    selected_points = torch.gather(
        points,  # (N, P, D)
        1,  # dim
        idx.unsqueeze(-1).expand(-1, -1, D),  # (N, K, D)
    )

    return selected_points, idx

# define Dataset
class ForestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, forests, preload=True, max_trees = 10, center=True, normalize=True, point_count=4096, sampling='random', select_trees = None, annotated = None, bbox = None, aug_rot = True):
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
        self.has_aug_rot = aug_rot
        if forests == 'all':
            self.forests = [os.path.basename(f)[:-4] for f in os.listdir(data_dir) if f.endswith('.laz') or f.endswith('.las')]
        print("Loading {} forests from {}".format(len(self.forests), data_dir))
        #print(self.forests)
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
            idx = farthest_point_sample_gpu(torch.from_numpy(xyz.astype(np.float32)).unsqueeze(0).cuda(), self.point_count)[0].squeeze(0).cpu().numpy()
            return xyz[idx, :], target[idx]

    def aug_rot(self, xyz, target):
        # randomly rotate around z axis
        angle = np.random.rand() * 2 * np.pi
        xyz = np.dot(xyz, np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]))
        return xyz, target

    def __getitem__(self, index):
        forest_id = self.forests[index]
        xyz, target = self.load_laz(forest_id)
        xyz, target = self.sample_trees(xyz, target)
        xyz, target = self.sample_points(xyz, target)
        if self.has_aug_rot:
            xyz, target = self.aug_rot(xyz, target)
        return xyz, target

    def __len__(self):
        return len(self.forests)
    

# define DataLoader
def get_dataloader(data_dir, forests, batch_size, num_workers, max_trees = 10, point_count=4096, sampling='random', select_trees = None, annotated = None, bbox = None, preload=True, distributed=False, world_rank = 0, world_size = 1, aug_rot = True):
    dataset = ForestDataset(data_dir, forests, max_trees=max_trees, point_count=point_count, sampling=sampling, select_trees = select_trees, annotated = annotated, bbox = bbox, preload=preload, aug_rot = aug_rot)
    if distributed:
        data_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=world_rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=data_sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader