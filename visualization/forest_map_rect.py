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
    # save figure
    fig_path = os.path.join('.', 'results', forest_basename + '_' + str(side) + '.png')
    fig, ax = plt.subplots()
    # visualize forest as rectangles, D is diameter (width), H is height
    for i, row in df.iterrows():
        x, y, d, grp, h = row
        X,Y, D, H = (x - d/2)/side, (y - h/2)/side, d/side, h/side
        #print(X, Y, D, H)
        ax.add_patch(plt.Rectangle((X, Y), D, H, color='brown', alpha=0.5))
    ax.add_patch(plt.Rectangle((0, 0), side, side, color='green', alpha=0.5, label='Forest'))
    ax.set_aspect('equal')
    # prettify
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Forest Map')
    ax.legend()
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