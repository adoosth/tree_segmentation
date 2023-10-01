import laspy
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plot3d(xyz, id, ax = None):
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
    # remove background grid
    ax.grid(False)
    # remove background color except for ground
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    # make ground color light green
    ax.zaxis.pane.fill = True
    #ax.zaxis.pane.set_color('green')
    # make it creamish brown and more transparent
    ax.zaxis.pane.set_color((0.8, 0.6, 0.4, 0.1))
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
    # set axis limits
    #ax.set_xlim(np.min(xyz[:, 0]), np.max(xyz[:, 0]))
    #ax.set_ylim(np.min(xyz[:, 1]), np.max(xyz[:, 1]))
    #ax.set_zlim(np.min(xyz[:, 2]), np.max(xyz[:, 2]))
    # set axis limits to whatever is in the data
    ax.set_xlim(xyz[:, 0].min(), xyz[:, 0].max())
    ax.set_ylim(xyz[:, 1].min(), xyz[:, 1].max())
    ax.set_zlim(0, 40)
    # rotate to face corner
    # ax.view_init(elev=30, azim=45)
    # choose the other corner
    #ax.view_init(elev=28, azim=250)
    # make it closer to the image and zoomed in
    ax.view_init(elev=20, azim=250)
    # plot
    # choose color map viridis
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=id, cmap=plt.get_cmap('viridis'), s = 0.1)

    #ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=id, cmap=plt.get_cmap('jet'), s = 0.1)

def bev_plot(xyz, id, ax = None):
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
    # remove background grid
    ax.grid(False)
    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis labels
    #ax.set_xlabel('X', fontdict={'size': 16, 'family': 'sans-serif'})
    #ax.set_ylabel('Y', fontdict={'size': 16, 'family': 'sans-serif'})
    # set axis limits to whatever is in the data
    ax.set_xlim(np.min(xyz[:, 0]), np.max(xyz[:, 0]))
    ax.set_ylim(np.min(xyz[:, 1]), np.max(xyz[:, 1]))
    # remove lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # plot
    ax.scatter(xyz[:, 0], xyz[:, 1], c=id, cmap=plt.get_cmap('viridis'), s = 0.1)

def shuffle_ids(ids):
    unique_ids = np.unique(ids)
    np.random.shuffle(unique_ids)
    for i, unique_id in enumerate(unique_ids):
        ids[ids == unique_id] = i
    return ids

def main(las):
    # select columns
    xyz = np.vstack([las.x, las.y, las.z]).transpose()
    id_true = shuffle_ids(np.vstack([las.hitObjectId]).transpose().reshape(-1))
    id_pred = shuffle_ids(np.vstack([las.treeID]).transpose().reshape(-1))
    # plot in 3D
    fig = plt.figure(figsize=(28, 12))
    fig.tight_layout()
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax2 = fig.add_subplot(1,2,2)
    # plot
    plot3d(xyz, id_true, ax1)
    bev_plot(xyz, id_true, ax2)
    plt.savefig('results/{}.png'.format(las_basename), dpi=300)
    fig = plt.figure(figsize=(28, 12))
    fig.tight_layout()
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax2 = fig.add_subplot(1,2,2)
    # plot
    plot3d(xyz, id_pred, ax1)
    bev_plot(xyz, id_pred, ax2)
    plt.savefig('results/{}_pred.png'.format(las_basename), dpi=300)


if __name__ == '__main__':
    las_basename = sys.argv[1]
    las_path = os.path.join('.', 'data', las_basename + '.las')
    if not os.path.exists(las_path):
        las_path = os.path.join('.', 'data', las_basename + '.laz')
    if not os.path.exists(las_path):
        print('No LAS file found for basename: {}'.format(las_basename))
        sys.exit(1)
    print('Reading LAS file: {}'.format(las_path))
    las = laspy.read(las_path)
    print('LAS file read successfully.')
    print('Number of points: {}'.format(len(las.x)))
    # center the point cloud
    las.x = las.x - np.min(las.x)
    las.y = las.y - np.min(las.y)
    main(las)