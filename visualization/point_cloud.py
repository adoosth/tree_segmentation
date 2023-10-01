import laspy
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def main(las):
    # select columns
    xyz = np.vstack([las.x, las.y, las.z]).transpose()
    raw_id = np.vstack([las.hitObjectId]).transpose().reshape(-1)
    # convert raw_id to id by unique values and then index
    id = np.zeros(len(raw_id))
    unique_ids = np.unique(raw_id)
    # shuffle
    np.random.shuffle(unique_ids)
    for i, unique_id in enumerate(unique_ids):
        id[raw_id == unique_id] = i
    # plot in 3D
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
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
    # set axis limits to (-20, 20, 0) to (70, 70, 60)
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 70)
    ax.set_zlim(0, 50)
    # rotate to face corner
    # ax.view_init(elev=30, azim=45)
    # choose the other corner
    ax.view_init(elev=28, azim=250)
    # plot
    # choose color map viridis
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=id, cmap=plt.get_cmap('viridis'), s = 0.1)
    #ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=id, cmap=plt.get_cmap('jet'), s = 0.1)
    plt.savefig('results/{}.png'.format(las_basename), dpi=300)
    # also plot a BeV 
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
    plt.savefig('results/{}_bev.png'.format(las_basename), dpi=300)


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
    main(las)