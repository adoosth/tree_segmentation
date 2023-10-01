import sys
import os
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

flight_path = [
                {"x": 100, "y": 300},
                {"x": 300, "y": 100},
                {"x": 250, "y": 50},
                {"x": 50, "y": 250},
                {"x": 0, "y": 200},
                {"x": 200, "y": 0},
                {"x": 150, "y": -50},
                {"x": -50, "y": 150},
                {"x": -100, "y": 100},
                {"x": 100, "y": -100},
                {"x": 300, "y": 100},
                {"x": 250, "y": 150},
                {"x": 50, "y": -50},
                {"x": 0, "y": 0},
                {"x": 200, "y": 200},
                {"x": 150, "y": 250},
                {"x": -50, "y": 50},
                {"x": -100, "y": 100},
                {"x": 100, "y": 300}
            ]

def main(df, side, overlap = 0):
    global flight_path
    print(df.head())
    # select side
    df = df[(df['X'] >= -overlap) & (df['X'] <= side + overlap)]
    df = df[(df['Y'] >= -overlap) & (df['Y'] <= side + overlap)]
    # select columns
    df = df[['X', 'Y', 'D', 'Grp', 'H']]
    # save figure
    fig_path = os.path.join('.', 'results', forest_basename + '_' + str(side) + '_flight_path.png')
    fig, ax = plt.subplots(figsize=(8.5, 8))
    # visualize forest as circles, D is diameter (width), H is height which is represented by alpha
    for i, row in df.iterrows():
        x, y, d, grp, h = row
        #X,Y, D, H = x/side, y/side, d/side, h/60.0
        X,Y, D, H = x, y, d, h/60.0
        #print(X, Y, D, H)
        ax.add_patch(plt.Circle((X, Y), 4*D, color='brown', alpha=H))
        #ax.add_patch(plt.Rectangle((0, 0), side, side, color='green', alpha=0.01, label='Forest'))
    ax.add_patch(plt.Rectangle((-overlap, -overlap), side + 2*overlap, side + 2*overlap, color='green', alpha=0.12, label='Forest'))

    flight_path = [{"x": p["x"]/4.0, "y": p["y"]/4.0} for p in flight_path]

    # draw flight path
    for i in range(len(flight_path) - 1):
        
        ax.plot([flight_path[i]["x"], flight_path[i+1]["x"]], [flight_path[i]["y"], flight_path[i+1]["y"]], color='red', linewidth=2)
    

    # set font to academic
    #plt.rcParams['font.family']  = 'Calibri'
    ax.set_aspect('equal')
    # prettify
    ax.set_xlabel('X (m)', fontdict={'size': 16, 'family': 'sans-serif'})
    ax.set_ylabel('Y (m)', fontdict={'size': 16, 'family': 'sans-serif'})
    # set limits to include flight path
    ax.set_xlim(-25, 80)
    ax.set_ylim(-25, 80)
    fig.tight_layout()

    #ax.set_title('Forest Map')
    #ax.legend()
    # remove borders and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # set tick size
    ax.tick_params(axis='both', which='major', labelsize=16)
    #ax.set_xticks([0, side/4, side/2, 3*side/4, side])
    #ax.set_yticks([0, side/4, side/2, 3*side/4, side])
    # save
    fig.savefig(fig_path)

if __name__ == '__main__':
    forest_basename = sys.argv[1]
    if len(sys.argv) > 2:
        side = int(sys.argv[2])
    else:
        side = 1000
    if len(sys.argv) > 3:
        overlap = int(sys.argv[3])
    else:
        overlap = 0
    forest_path = os.path.join('.', 'data', forest_basename + '.res')
    df = pd.read_csv(forest_path, skiprows=2, header=0, sep='\t+', engine='python')
    print(df.head())
    main(df, side, overlap)