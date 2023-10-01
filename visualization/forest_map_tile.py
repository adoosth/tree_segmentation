import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main(df, side):
    print(df.head())
    # select side
    df = df[(df['X'] >= 0) & (df['X'] <= side)]
    df = df[(df['Y'] >= 0) & (df['Y'] <= side)]
    # select columns
    df = df[['X', 'Y', 'D', 'Grp', 'H']]
    colors = ['brown', 'blue', 'red', 'orange', 'purple', 'yellow', 'pink', 'cyan', 'magenta']
    # \textit{Pinus sylvestris}, \textit{Picea abies}, \textit{Fagus sylvatica}, \textit{Quercus robur}, \textit{Populus marilandica}, \textit{Fraxinus excelsior}, \textit{Betula pendula} and \textit{Pseudotsuga menziesii}.
    species = ['P. sylvestris', 'P. abies', 'F. sylvatica', 'Q. robur', 'P. marilandica', 'F. excelsior', 'B. pendula', 'P. menziesii']
    # save figure
    fig_path = os.path.join('.', 'results', forest_basename + '_' + str(side) + '_tile.png')
    fig, ax = plt.subplots(figsize=(12, 8))
    # visualize forest as circles, D is diameter (width), H is height which is represented by alpha
    for i, row in df.iterrows():
        x, y, d, grp, h = row
        #X,Y, D, H = x/side, y/side, d/side, h/60.0
        X,Y, D, H = x, y, d, h/60.0
        #print(X, Y, D, H)
        ax.add_patch(plt.Circle((X, Y), D, color=colors[int(grp)], alpha=H))
    ax.add_patch(plt.Rectangle((0, 0), side, side, color='green', alpha=0.5, label='Forest'))

    # add a hollow rectangle from 0, 0, to 50, 50
    #ax.add_patch(plt.Rectangle((0, 0), 50, 50, fill=False, color='red', alpha=1, label='Tile Center'))
    # make it thickekr
    ax.add_patch(plt.Rectangle((1, 1), 49, 49, fill=False, color='red', alpha=1, label='Tile Center', linewidth=3))
    # add a hollow rectangle from -20, -20, to 70, 70
    ax.add_patch(plt.Rectangle((-20, -20), 90, 90, fill=False, color='blue', alpha=1, label='Tile Boundary', linewidth=2))
    # add a hollow rectangle for the second tile
    #ax.add_patch(plt.Rectangle((1, 51), 49, 49, fill=False, color='red', alpha=1))
    #ax.add_patch(plt.Rectangle((-20, 30), 90, 90, fill=False, color='yellow', alpha=1))
    # add a hollow rectangle for the third tile
    ax.add_patch(plt.Rectangle((101, 1), 49, 49, fill=False, color='red', alpha=1, linewidth=3))
    ax.add_patch(plt.Rectangle((80, -20), 90, 90, fill=False, color='yellow', alpha=1, linewidth=2))


    # set font to academic
    #plt.rcParams['font.family']  = 'Calibri'
    ax.set_aspect('equal')
    # prettify
    ax.set_xlabel('X (m)', fontdict={'size': 16, 'family': 'sans-serif'})
    ax.set_ylabel('Y (m)', fontdict={'size': 16, 'family': 'sans-serif'})
    # set limits
    ax.set_xlim(0, side, auto=True)
    ax.set_ylim(0, side, auto=True)
    #ax.set_title('Forest Map')
    # add legend for tile center and boundary
    legend_patches = []
    legend_patches.append(plt.Rectangle((0, 0), 1, 1, fill=False, color='red', alpha=1, label='Tile Centers', linewidth=3))
    legend_patches.append(plt.Rectangle((0, 0), 1, 1, fill=False, color='blue', alpha=1, label='Tile Boundary', linewidth=2))
    legend_patches.append(plt.Rectangle((0, 0), 1, 1, fill=False, color='yellow', alpha=1, label='Tile Boundary', linewidth=2))
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=16)

    # remove borders and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # set tick size
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xticks([0, side/4, side/2, 3*side/4, side])
    ax.set_yticks([0, side/4, side/2, 3*side/4, side])
    # save
    fig.savefig(fig_path)

if __name__ == '__main__':
    forest_basename = sys.argv[1]
    if len(sys.argv) > 2:
        side = int(sys.argv[2])
    else:
        side = 1000
    forest_path = os.path.join('.', 'data', forest_basename + '.res')
    df = pd.read_csv(forest_path, skiprows=2, header=0, sep='\t+', engine='python')
    print(df.head())
    main(df, side)