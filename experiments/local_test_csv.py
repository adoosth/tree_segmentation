import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import laspy
import threading
# import kmeans as cluster
from sklearn.cluster import KMeans, DBSCAN, MeanShift
# gaussian mixture model
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
# 3d with matplotlib
from mpl_toolkits.mplot3d import Axes3D
# import tsne
from sklearn.manifold import TSNE
#from dataloader_big import get_dataloader
from dataloader_csv import get_dataloader

# pca
from sklearn.decomposition import PCA

model_basename = None
model_path = None
architecture = None

from architectures.SGPN_conf import SGPN

def visualize2d(points, labels, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(points[:, 0], points[:, 1], c=labels)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()


def visualize(points, labels, figsize=(10,10), save_path=None, latents = None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, s=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # rotate points
    ax = fig.add_subplot(222, projection='3d')
    ax.view_init(90, 0)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, s=0.1)

    if latents is not None:
        ax = fig.add_subplot(223, projection='3d')
        ax.scatter(latents[:, 0], latents[:, 1], latents[:, 2], c=labels, s=0.1)
        # rotate points
        ax = fig.add_subplot(224, projection='3d')
        ax.view_init(90, 0)
        ax.scatter(latents[:, 0], latents[:, 1], latents[:, 2], c=labels, s=0.1)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_mappping(points, labels, targets, mapping, figsize=(10,10), save_path=None):
    # green: correct 
    # red: incorrect
    truth_labels = np.array([mapping[l] for l in labels])
    correct = targets == truth_labels
    incorrect = targets != truth_labels
    fig = plt.figure(figsize=figsize)
    fig.tight_layout()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(points[correct, 0], points[correct, 1], points[correct, 2], c='g', s=0.1)
    ax.scatter(points[incorrect, 0], points[incorrect, 1], points[incorrect, 2], c='r', s=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Correct vs. Incorrect')
    ax.legend(['Correct', 'Incorrect'])
    # rotate points
    ax = fig.add_subplot(122, projection='3d')
    ax.view_init(90, 0)
    ax.scatter(points[correct, 0], points[correct, 1], points[correct, 2], c='g', s=0.1)
    ax.scatter(points[incorrect, 0], points[incorrect, 1], points[incorrect, 2], c='r', s=0.1)
    # save
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()


def get_labels(model, data):
    xyz, target = data
    xyz = xyz.float().cuda()
    target = target.float().cuda()
    # take 4096 points randomly
    if (xyz.shape[1] > 4096):
        idx = np.random.choice(xyz.shape[1], 4096, replace=False)
        xyz = xyz[:, idx, :]
        target = target[:, idx]
    #output = model(xyz, test=True)
    output = model.segment(xyz)
    return xyz.detach().cpu().numpy(), target.detach().cpu().numpy(), output#.reshape(-1)

def get_features(model, data):
    xyz, target = data
    xyz = xyz.float().cuda()
    target = target.float().cuda()
    # take 4096 points randomly
    if (xyz.shape[1] > 4096):
        idx = np.random.choice(xyz.shape[1], 4096, replace=False)
        xyz = xyz[:, idx, :]
        target = target[:, idx]
    output = model(xyz, test=True)
    return xyz.detach().cpu().numpy(), target.detach().cpu().numpy(), output.detach().cpu().numpy()

def cluster(features, target = None, n_clusters=10, save_path=None, method = 'gmm'):
    print("Clustering for save_path = ", save_path)
    # First apply PCA to reduce dimensionality
    pca = PCA(n_components=3)
    pca.fit(features)
    features = pca.transform(features)
    print(features.shape)

    # Then apply k-means
    if method == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(features[:, :2])
        labels = clustering.labels_
    # visualize features
        if save_path is not None:
            visualize2d(features[:, :2], labels, save_path=os.path.join(save_path, 'pca_kmeans.png'))
            if target is not None:
                visualize2d(features[:, :2], target, save_path=os.path.join(save_path, 'pca_kmeans_truth.png'))
        return labels

    # Apply GMM
    if method == 'gmm':
        # initialize with kmeans
        #clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(features[:, :3])
        #labels = clustering.labels_
        # fit GMM
        clustering = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(features[:, :5])

        labels = clustering.predict(features[:, :5])
        os.makedirs('./' + architecture + '/results/' + model_basename, exist_ok=True)
        # visualize features
        if save_path is not None:
            visualize2d(features[:, :2], labels, save_path=os.path.join(save_path, 'pca_gmm.png'))
            if target is not None:
                visualize2d(features[:, :2], target, save_path=os.path.join(save_path, 'pca_gmm_truth.png'))
        return labels
    
    # Apply DBScan
    if method.lower() == 'dbscan':
        print("Clustering with DBScan", end=' ')
        clustering = DBSCAN(eps=0.3, min_samples=20).fit(features[:, :3])
        labels = clustering.labels_
        print("Done")
        os.makedirs('./' + architecture + '/results/' + model_basename, exist_ok=True)
        # visualize features
        if save_path is not None:
            visualize2d(features[:, :2], labels, save_path=os.path.join(save_path, 'pca_dbscan.png'))
            if target is not None:
                visualize2d(features[:, :2], target, save_path=os.path.join(save_path, 'pca_dbscan_truth.png'))
        return labels

    if method.lower() == 'meanshift':
        print("Clustering with MeanShift", end=' ')
        clustering = MeanShift(bandwidth=0.5).fit(features[:, :3])    
        # higher bandwidth causes higher or lower number of clusters??? the answer is 
        labels = clustering.labels_
        print("Done")
        os.makedirs('./' + architecture + '/results/' + model_basename, exist_ok=True)
        # visualize features
        if save_path is not None:
            visualize2d(features[:, :2], labels, save_path=os.path.join(save_path, 'pca_meanshift.png'))
            if target is not None:
                visualize2d(features[:, :2], target, save_path=os.path.join(save_path, 'pca_meanshift_truth.png'))
        return labels
    
    if method.lower() == 'tsne':
        # get tsne embedding
        features = TSNE(n_components=3, perplexity=30.0).fit_transform(features[:, :5])
        features = features[:, :3]

        # Then apply k-means
        clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        labels = clustering.labels_
        
        # visualize features
        if save_path is not None:
            visualize2d(features[:, :2], labels, save_path=os.path.join(save_path, 'pca_tsne_kmeans.png'))
            if target is not None:
                visualize2d(features[:, :2], target, save_path=os.path.join(save_path, 'pca_tsne_kmeans_truth.png')) 
        return labels

def calc_iou(labels, targets):
    # for each target cluster, find the maximium iou with any label cluster
    # return all ious
    ious = []
    for t in np.unique(targets):
        all_ious = []
        for l in np.unique(labels):
            intersection = np.sum((targets == t) & (labels == l))
            union = np.sum((targets == t) | (labels == l))
            iou = intersection / union
            all_ious.append(iou)
        ious.append(np.max(all_ious))
    return ious

# def evaluate(labels, targets, mapping):
#     # find the best match between labels and targets
#     # return the accuracy

#     # compute accuracy
#     accuracy = np.sum(targets == np.array([mapping[l] for l in labels])) / len(targets)
#     print("Accuracy: ", accuracy)
#     # compute mAP
#     mAP = 0
#     for l in np.unique(labels):
#         mAP += np.sum(targets[labels == l] == mapping[l]) / np.sum(labels == l)
#     mAP /= len(np.unique(labels))
#     print("mAP: ", mAP)
#     # compute mIoU
#     mIoU = 0
#     for l in np.unique(labels):
#         mIoU += np.sum(targets[labels == l] == mapping[l]) / (np.sum(targets == mapping[l]) + np.sum(labels == l) - np.sum(targets[labels == l] == mapping[l]))
#     mIoU /= len(np.unique(labels))
#     print("mIoU: ", mIoU)
#     return accuracy, mAP, mIoU

def get_ap_threshold(ious, threshold = 0.5):
    # compute average precision
    # ious: list of ious
    # threshold: iou threshold for positive detection
    # return: average precision
    ious = np.array(ious)
    tps = np.sum(ious >= threshold)
    fns = np.sum(ious < threshold)
    return tps / (tps + fns)

def get_ap_50 (ious):
    return get_ap_threshold(ious, threshold = 0.5)

def get_ap_25 (ious):
    return get_ap_threshold(ious, threshold = 0.25)

def get_ap(ious):
    # average from 0.5 to 0.95
    return np.mean([get_ap_threshold(ious, threshold = t) for t in np.arange(0.5, 1.0, 0.05)])

results = [None] * 100

def test(xyz, target, features, save_path, method, focus_ratio, iter = None):
    # Center xyz in x,y
        xyz[:, 0] -= np.min(xyz[:, 0]) + (np.max(xyz[:, 0]) - np.min(xyz[:, 0])) / 2
        xyz[:, 1] -= np.min(xyz[:, 1]) + (np.max(xyz[:, 1]) - np.min(xyz[:, 1])) / 2

        n_true_clusters = len(np.unique(target))
        #if n_true_clusters == 1:
        #    n_true_clusters = 10
        print(features.shape)
        labels = cluster(features, n_clusters=n_true_clusters, save_path=save_path, target=target, method = method)
        print("SGPN found {} clusters".format(len(np.unique(labels))))
        print("Ground truth has {} clusters".format(len(np.unique(target))))

        # save results as las
        labels = labels.reshape(-1)
        xyz = xyz.reshape(-1, 3)
        labels = labels.astype(np.uint8)
        xyz = xyz.astype(np.float32)
        las = laspy.create(point_format=2)
        las.x = xyz[:, 0]
        las.y = xyz[:, 1]
        las.z = xyz[:, 2]
        # add extra data as hitObjectId
        las.user_data = labels
        if iter is not None:
            las.write(os.path.join(save_path, dataset + '_' + str(iter) + '_' + 'SGPN' + '.laz'))
        else:
            las.write(os.path.join(save_path, dataset + '_' + 'SGPN' + '.laz'))

        if focus_ratio is not None:
            # print("Focus on center ratio: ", focus_ratio)
            total_x = np.max(xyz[:, 0]) - np.min(xyz[:, 0])
            total_y = np.max(xyz[:, 1]) - np.min(xyz[:, 1])
            # print("Total x: ", total_x, " | min: ", np.min(xyz[:, 0]), " | max: ", np.max(xyz[:, 0]))
            # print("Total y: ", total_y, " | min: ", np.min(xyz[:, 1]), " | max: ", np.max(xyz[:, 1]))
            focus_side_x = total_x * focus_ratio
            focus_side_y = total_y * focus_ratio
            # print("Focus side x: ", focus_side_x)
            # print("Focus side y: ", focus_side_y)
            idx = np.where((xyz[:, 0] >= -focus_side_x/2) & (xyz[:, 0] <= focus_side_x/2) & (xyz[:, 1] >= -focus_side_y/2) & (xyz[:, 1] <= focus_side_y/2))[0]
            # print("Found {} points".format(len(idx)))
            xyz = xyz[idx]
            target = target[idx]
            labels = labels[idx]
            features = features[idx]

        #mapping = best_mapping(labels, target)
        ious = calc_iou(labels, target)
        miou = np.mean(ious) * 100
        miou = np.round(miou, 2)
        # print("IoUs: ", ious)
        
        AP = get_ap(ious)*100
        AP_50 = get_ap_threshold(ious, threshold = 0.5)*100
        AP_25 = get_ap_threshold(ious, threshold = 0.25)*100

        AP = np.round(AP, 2)
        AP_50 = np.round(AP_50, 2)
        AP_25 = np.round(AP_25, 2)

        # print and delimit with tabs
        # print("miou: %.2f" % miou, end='\t')
        # print("AP: %.2f" % (AP), end='\t')
        # print("AP@25: %.2f" % (AP_25), end='\t')
        # print("AP@50: %.2f" % (AP_50))

        
        # save metrics as csv
        # with open(os.path.join(save_path, 'metrics.csv'), 'w') as f:
        #     f.write("mIoU,AP,AP@25,AP@50\n")
        #     f.write("{},{},{},{}".format(miou, AP, AP_25, AP_50))

        # visualize
        visualize(xyz, labels, save_path=os.path.join(save_path,f'{iter}_pred.png'))#, latents = features)
        visualize(xyz, target, save_path=os.path.join(save_path,f'{iter}_truth.png'))
        if iter is not None:
            results[iter] = (miou, AP, AP_50, AP_25)

        return miou, AP, AP_50, AP_25

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_basename = sys.argv[1]
        architecture = model_basename.split('_')[0]
        model_filename = model_basename + '.ckpt'
        model_path = './' + architecture + '/models/' + model_filename
        dataset = sys.argv[2]
        subfolder = sys.argv[3] if len(sys.argv) > 3 else 'test'
        data_dir = os.path.join("../data", dataset)
        csv_path = os.path.join(data_dir, subfolder + '.csv')
        #bbox_side = int(sys.argv[4]) if len(sys.argv) > 4 else None
        #bbox = [-bbox_side/2, -bbox_side/2, -bbox_side/2, bbox_side/2, bbox_side/2, bbox_side/2] if bbox_side is not None else None
        method = sys.argv[4] if len(sys.argv) > 4 else 'gmm'
        focus_ratio = float(sys.argv[5]) if len(sys.argv) > 5 else None
        if focus_ratio is not None:
            print("Focus ratio: ", focus_ratio)
        else:
            print("WARNING: No focus ratio given")
    else:
        print("Usage: python test.py <model_basename> <data_dir> [<subfolder>] [method]")
        exit(0)
    # spawn cuda
    #torch.multiprocessing.set_start_method('spawn')
    model = torch.load(model_path)
    # get latent dims from model
    #latent_dims = model.latent_dims
    try:
        model_new = SGPN(num_classes=10, latent_dims=10, alpha_step = 400, margin=0.8).cuda()
        model_new.load_state_dict(model.state_dict())
        model = model_new
    except:
        try:
            model_new = SGPN(num_classes=10, latent_dims=3, alpha_step = 400, margin=0.8).cuda()
            model_new.load_state_dict(model.state_dict())
            model = model_new
        except:
            try:
                model_new = SGPN(latent_dims=128).cuda()
                model_new.load_state_dict(model.state_dict())
                model = model_new
            except:
                try:
                    model_new = SGPN(latent_dims=4).cuda()
                    model_new.load_state_dict(model.state_dict())
                    model = model_new
                except:
                    print("Warning: could not load model with latent_dims")
    save_path = os.path.join('./', architecture, 'results', model_basename, dataset + '-' + subfolder)
    model.cuda().eval()
    print("Clustering method: ", method)
    
    #dataloader = get_dataloader('../data/real', 'all', 1, 1, max_trees = 50, point_count = 4096, sampling='random')
    batch_size = 8
    dataloader = get_dataloader(data_dir, subfolder, batch_size, 1, max_trees = 50, point_count = 4096, sampling='random', preload=False, center=True, normalize=True, select_trees = None, annotated = True, aug_rot = False)
    data = next(iter(dataloader))
    # xyz, target, labels = get_labels(model, data)
    # xyz, target = xyz[0], target[0]

    # Manual clustering
    xyzs, targets, featuress = get_features(model, data)

    # with threading
    threads = []
    for i, (xyz, target, features) in enumerate(zip(xyzs, targets, featuress)):
        print("Sending thread {} to test".format(i))
        t = threading.Thread(target=test, args=(xyz, target, features, save_path, method, focus_ratio, i))
        threads.append(t)
        t.start()
    df = pd.DataFrame(columns=['mIoU', 'AP', 'AP@50', 'AP@25'])
    for i,t in enumerate(threads):
        # get return values
        t.join()
        miou, AP, AP_50, AP_25 = results[i]
        df.loc[i] = [miou, AP, AP_50, AP_25]
    df.to_csv(os.path.join(save_path, 'metrics.csv'))
    print(df)
    # print means in one line
    print("Mean: ", df.mean(axis=0))
    #test(xyz, target, features, save_path, method, focus_ratio)
         # test(xyz, target, features, save_path, method, focus_ratio)
            
        
    #visualize_mappping(xyz, labels, target, mapping, save_path=os.path.join(save_path,'mapping.png'))
    #accuracy, mAP, mIoU = evaluate(labels, target, mapping)
    #with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
    #    f.write("Accuracy: {}\nmAP: {}\nmIoU: {}".format(accuracy, mAP, mIoU))

    

    
