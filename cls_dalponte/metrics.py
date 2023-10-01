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

# pca
from sklearn.decomposition import PCA

model_basename = None
model_path = None
architecture = None

def prefix_ax(ax):
    ax.grid(False)
    # remove background color except for ground
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    # make ground color light green
    ax.zaxis.pane.fill = True
    #ax.zaxis.pane.set_color('green')
    # make it creamish brown and more transparent
    ax.zaxis.pane.set_color((0.95, 0.9, 0.8, 0.1))
    # remove background lines
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    # remove background lines
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0.5)
    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # set axis labels
    #ax.set_xlabel('X', fontdict={'size': 16, 'family': 'sans-serif'})
    #ax.set_ylabel('Y', fontdict={'size': 16, 'family': 'sans-serif'})
    #ax.set_zlabel('Z (m)', fontdict={'size': 16, 'family': 'sans-serif'})
    # remove lines
    ax.w_zaxis.line.set_lw(0.)
    ax.w_xaxis.line.set_lw(0.)
    ax.w_yaxis.line.set_lw(0.)
    # remove z tick labels
    ax.set_zticklabels([])
    # remove z ticks
    ax.set_zticks([])

def visualize(points, labels, targets, figsize = (30,30), s = 100, save_path = None, a = 1):
    # set colormap to viridis
    # mak
    fig = plt.figure(figsize=figsize)
    
    fig.tight_layout()
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, s=s, cmap='viridis', alpha=a)
    prefix_ax(ax)

    # BeV
    ax = fig.add_subplot(222, projection='3d')
    ax.view_init(90, 0)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, s=s, cmap='viridis', alpha=a)
    prefix_ax(ax)

    ax = fig.add_subplot(223, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=targets, s=s, cmap='viridis', alpha=a)
    prefix_ax(ax)
    # BeV
    ax = fig.add_subplot(224, projection='3d')
    ax.view_init(90, 0)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=targets, s=s, cmap='viridis', alpha=a)
    prefix_ax(ax)

    # figure should be tight
    fig.tight_layout()
    # make sure no white space
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close(fig)


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
        os.makedirs('./results/' + model_basename, exist_ok=True)
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
        os.makedirs('./results/' + model_basename, exist_ok=True)
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
        os.makedirs('./results/' + model_basename, exist_ok=True)
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
    # dimensions: n_targets x n_labels
    ious = np.array([[0.0] * len(np.unique(labels))] * len(np.unique(targets)))
    print(ious.shape)
    for t in np.unique(targets):
        for l in np.unique(labels):
            intersection = np.sum((targets == t) & (labels == l))
            union = np.sum((targets == t) | (labels == l))
            iou = intersection / union
            ious[t, l] = iou
    print(ious.shape)
    # in each iteration, find the maximum iou in the matrix and set the row and column to 0
    best_ious = []
    for i in range(len(np.unique(labels))):
        idx = np.argmax(ious)   
        t, l = idx // len(np.unique(labels)), idx % len(np.unique(labels))
        #print(idx, ious[t, l])
        best_ious.append(ious[t, l])
        ious[t, :] = 0
        ious[:, l] = 0
    return best_ious

def get_ap_threshold(ious, threshold = 0.5):
    # compute average precision
    # ious: list of ious
    # threshold: iou threshold for positive detection
    # return: average precision
    ious = np.array(ious)
    tps = np.sum(ious >= threshold)
    fps = np.sum(ious < threshold)
    # each true positive is worth as much as its number of points
    
    #return tps / (tps + fps) * (tps / len(ious))
    return tps / (tps + fps)

def get_ap_50 (ious):
    return get_ap_threshold(ious, threshold = 0.5)

def get_ap_25 (ious):
    return get_ap_threshold(ious, threshold = 0.25)

def get_ap(ious):
    # average from 0.5 to 0.95
    return np.mean([get_ap_threshold(ious, threshold = t) for t in np.arange(0.5, 1.0, 0.05)])

results = [None] * 100

def reindex(target):
    unique_targets = np.unique(target)
    mapping = {}
    for i, t in enumerate(unique_targets):
        mapping[t] = i
    target = np.array([mapping[t] for t in target])
    return target


def test(xyz, target, labels, save_path, focus_ratio, iter = None, lock = None):
    # make plt work with multiprocessing
    # plt.switch_backend('agg')
    # get labels
    # Center xyz in x,y
    print("Centering xyz")
    xyz[:, 0] -= np.min(xyz[:, 0]) + (np.max(xyz[:, 0]) - np.min(xyz[:, 0])) / 2
    xyz[:, 1] -= np.min(xyz[:, 1]) + (np.max(xyz[:, 1]) - np.min(xyz[:, 1])) / 2

    n_true_clusters = len(np.unique(target))
    #if n_true_clusters == 1:
    #    n_true_clusters = 10
    print("Dalponte found {} clusters".format(len(np.unique(labels))))
    print("Ground truth has {} clusters".format(len(np.unique(target))))

    # save results as las
    labels = labels.reshape(-1)
    xyz = xyz.reshape(-1, 3)
    labels = labels.astype(np.uint8)
    xyz = xyz.astype(np.float32)

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
        # reindex
        target = reindex(target)
        labels = reindex(labels)
        

    #mapping = best_mapping(labels, target)
    print("Calculating metrics")
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
    if len(xyz) > 40000:
        idx = np.random.choice(xyz.shape[0], 40000, replace=False)
        xyz = xyz[idx]
        labels = labels[idx]
        target = target[idx]
    results[iter] = (miou, AP, AP_50, AP_25)


    return miou, AP, AP_50, AP_25


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        subfolder = sys.argv[2] if len(sys.argv) > 2 else 'test'
        iter = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        focus_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else None
        if focus_ratio is not None:
            print("Focus ratio: ", focus_ratio)
        else:
            focus_ratio = 0.8
            print("WARNING: No focus ratio given, using default value of ", focus_ratio)
        data_dir = os.path.join("../data", dataset)
        csv_path = os.path.join(data_dir, subfolder + '.csv')
        
    else:
        print("Usage: python test.py <dataset> <subfolder> <iter> <focus_ratio>")
        exit(0)
    
    xyzs, targets, labels = [], [], []
    for i in range(1, iter + 1):
        # load las files
        las_path = os.path.join('./results/', dataset, subfolder + '-'+ str(i) + '-dalponte.laz')
        if os.path.exists(las_path):
            las = laspy.read(las_path)
            xyz, target, label = las.xyz, las.hitObjectId, las.treeID
            xyzs.append(xyz)
            targets.append(target)
            labels.append(label)
        else:
            break
    print("Found {} las files".format(len(xyzs)))
    # with threading
    save_path = os.path.join('./results/', dataset, subfolder + '-dalponte')
    threads = []
    lock = threading.Lock()
    for i, (xyz, target, label) in enumerate(zip(xyzs, targets, labels)):
        print("Sending thread {} to test".format(i))
        t = threading.Thread(target=test, args=(xyz, target, label, save_path, focus_ratio, i), kwargs={'lock': lock})
        threads.append(t)
        t.start()
        #print(len(label))
        #test(xyz, target, label, save_path, focus_ratio, i)
    df = pd.DataFrame(columns=['mIoU', 'AP', 'AP@50', 'AP@25'])
    
    for i,t in enumerate(threads):
        # get return values
        t.join()
        miou, AP, AP_50, AP_25 = results[i]
        df.loc[i] = [miou, AP, AP_50, AP_25]

    df.to_csv(save_path + '-metrics.csv')
    print(df)
    # print means in one line
    print("Mean: ", df.mean(axis=0))
    print(dataset + " " + subfolder + " " + str(iter) + " " + str(focus_ratio))
    # print means in one line with 2 decimal places and put & between them and standard deviation in parenthesis
    print("Dalponte & ", end='')
    for i in range(4):
        print("{:.2f} ({:.2f})".format(df.mean(axis=0)[i], df.std(axis=0)[i]), end=' & ' if i < 3 else '')
    # add \\ at the end
    print(" \\\\")
    print()



    

    
