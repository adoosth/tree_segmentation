import sys
import os
import numpy as np
import torch
import torch.nn as nn
import laspy
# import kmeans as cluster
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# 3d with matplotlib
from mpl_toolkits.mplot3d import Axes3D
# import tsne
from sklearn.manifold import TSNE
from dataloader import get_dataloader

# pca
from sklearn.decomposition import PCA

model_basename = None
model_path = None
architecture = None


def visualize2d(points, labels, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(points[:, 0], points[:, 1], c=labels)
    if save_path is not None:
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

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path is not None:
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


def get_features(model, data):
    xyz, target = data
    xyz = xyz.float().cuda()
    target = target.float().cuda()
    # take 4096 points randomly
    if (xyz.shape[1] > 4096):
        idx = np.random.choice(xyz.shape[1], 4096, replace=False)
        xyz = xyz[:, idx, :]
        target = target[:, idx]
    output = model(xyz, target, training=False)
    print(output)
    print(output.shape)
    return xyz.detach().cpu().numpy(), target.detach().cpu().numpy(), output.detach().cpu().numpy()

def cluster(features, target = None, n_clusters=10):
    # First apply PCA to reduce dimensionality
    pca = PCA(n_components=3)
    pca.fit(features)
    features = pca.transform(features)
    print(features.shape)

    # Then apply k-means
    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(features[:, :2])
    labels = clustering.labels_
    os.makedirs('./' + architecture + '/results/' + model_basename, exist_ok=True)
    # visualize features
    visualize2d(features[:, :2], labels, save_path='./' + architecture + '/results/' + model_basename + '/pca_kmeans.png')
    if target is not None:
        visualize2d(features[:, :2], target, save_path='./' + architecture + '/results/' + model_basename + '/pca_kmeans_truth.png')
    # get tsne embedding
    features = TSNE(n_components=3, perplexity=30.0).fit_transform(features[:, :20])
    features = features[:, :3]

    # Then apply k-means
    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = clustering.labels_
    
    # visualize features
    visualize2d(features[:, :2], labels, save_path='./' + architecture + '/results/' + model_basename + '/pca_tsne_kmeans.png')
    if target is not None:
        visualize2d(features[:, :2], target, save_path='./' + architecture + '/results/' + model_basename + '/pca_tsne_kmeans_truth.png')
    return labels

def best_mapping(labels, targets):
    # find the best match between labels and targets
    # return the accuracy
    mapping = {}
    for l in np.unique(labels):
        # mapping[l] = np.argmax(np.bincount(targets[labels == l]))
        # fix cannot cast array from dtype('float32') to dtype('int64') according to the rule 'safe'
        mapping[l] = np.argmax(np.bincount(targets[labels == l].astype('int64')))
    return mapping

def evaluate(labels, targets, mapping):
    # find the best match between labels and targets
    # return the accuracy

    # compute accuracy
    accuracy = np.sum(targets == np.array([mapping[l] for l in labels])) / len(targets)
    print("Accuracy: ", accuracy)
    # compute mAP
    mAP = 0
    for l in np.unique(labels):
        mAP += np.sum(targets[labels == l] == mapping[l]) / np.sum(labels == l)
    mAP /= len(np.unique(labels))
    print("mAP: ", mAP)
    # compute mIoU
    mIoU = 0
    for l in np.unique(labels):
        mIoU += np.sum(targets[labels == l] == mapping[l]) / (np.sum(targets == mapping[l]) + np.sum(labels == l) - np.sum(targets[labels == l] == mapping[l]))
    mIoU /= len(np.unique(labels))
    print("mIoU: ", mIoU)
    return accuracy, mAP, mIoU

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_basename = sys.argv[1]
        architecture = model_basename.split('_')[2]
        model_filename = model_basename + '.ckpt'
        model_path = './' + architecture + '/models/' + model_filename
        dataset = sys.argv[2]
        subfolder = sys.argv[3] if len(sys.argv) > 3 else 'test'
        data_dir = os.path.join("../data", dataset, subfolder)
        bbox_side = int(sys.argv[4]) if len(sys.argv) > 4 else None
        bbox = [-bbox_side/2, -bbox_side/2, -bbox_side/2, bbox_side/2, bbox_side/2, bbox_side/2] if bbox_side is not None else None
    else:
        print("Usage: python test.py <model_basename> <data_dir> [<subfolder>]")
        exit(0)
    model = torch.load(model_path)
    save_path = os.path.join('./', architecture, 'results', model_basename, dataset + '-' + subfolder)
    model.cuda().eval()
    #dataloader = get_dataloader('../data/real', 'all', 1, 1, max_trees = 50, point_count = 4096, sampling='random')
    dataloader = get_dataloader(data_dir, 'all', 1, 1, max_trees = 50, point_count = 4096, sampling='random', annotated = None, preload=False, bbox = bbox)
    data = next(iter(dataloader))
    xyz, target, features = get_features(model, data)
    xyz, target, features = xyz[0], target[0], features[0]
    n_true_clusters = len(np.unique(target))
    if n_true_clusters == 1:
        n_true_clusters = 10
    #labels = cluster(features, target, n_clusters=50)
    labels = cluster(features, n_clusters=n_true_clusters)

    print(labels)
    print(labels.shape)
    print(np.unique(labels))
    mapping = best_mapping(labels, target)


    # visualize
    visualize(xyz, labels, save_path=os.path.join(save_path,'pred.png'), latents = features)
    visualize(xyz, target, save_path=os.path.join(save_path,'truth.png'))
    visualize_mappping(xyz, labels, target, mapping, save_path=os.path.join(save_path,'mapping.png'))
    accuracy, mAP, mIoU = evaluate(labels, target, mapping)
    with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
        f.write("Accuracy: {}\nmAP: {}\nmIoU: {}".format(accuracy, mAP, mIoU))
    
