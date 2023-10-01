import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def main(df, side):
    print(df.head())
    # select side
    df = df[(df['X'] >= 0) & (df['X'] <= side)]
    df = df[(df['Y'] >= 0) & (df['Y'] <= side)]
    # select columns
    df = df[['X', 'Y', 'D', 'Grp', 'H']]
    #colors = ['brown', 'blue', 'red', 'orange', 'purple', 'yellow', 'pink', 'cyan', 'magenta']
    # Make colors viridis
    colors = sns.color_palette('viridis', 8)
    print(colors)
    # \textit{Pinus sylvestris}, \textit{Picea abies}, \textit{Fagus sylvatica}, \textit{Quercus robur}, \textit{Populus marilandica}, \textit{Fraxinus excelsior}, \textit{Betula pendula} and \textit{Pseudotsuga menziesii}.
    #species = ['P. sylvestris', 'P. abies', 'F. sylvatica', 'Q. robur', 'P. marilandica', 'F. excelsior', 'B. pendula', 'P. menziesii']
    
    # I want full species names
    species = ['Pinus sylvestris', 'Picea abies', 'Fagus sylvatica', 'Quercus robur', 'Populus marilandica', 'Fraxinus excelsior', 'Betula pendula', 'Pseudotsuga menziesii']
    # save figure
    fig_path = os.path.join('.', 'results', forest_basename + '_' + str(side) + '_species.png')
    fig, ax = plt.subplots(figsize=(15, 8))
    # cream color is 
    # color=()
    ax.add_patch(plt.Rectangle((0, 0), side, side, color=(0.8, 0.6, 0.4), alpha=0.1, label='Forest'))
    # visualize forest as circles, D is diameter (width), H is height which is represented by alpha
    for i, row in df.iterrows():
        x, y, d, grp, h = row
        #X,Y, D, H = x/side, y/side, d/side, h/60.0
        X,Y, D, H = x, y, d, h/60.0
        #print(X, Y, D, H)
        ax.add_patch(plt.Circle((X, Y), 4*D, color=colors[int(grp)-1], alpha=H))
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
    # add legend for species
    #plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    legend_patches = []
    for i, species_name in enumerate(species):
        legend_patches.append(plt.Circle((0, 0), 1, color=colors[i], alpha=1, label=species_name))
    # scientific name has to be italicized
    font = FontProperties()
    font.set_style('italic')
    #font.set_family('serif')
    font.set_size(16)
    # set cool font
    # font.set_family('sans-serif') # not cool enough
    # set times new roman
    #font.set_family('Times New Roman')
    ax.legend(handles=legend_patches, loc='upper left', fontsize=20, frameon=False, prop=font, bbox_to_anchor=(1, 1))
    


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