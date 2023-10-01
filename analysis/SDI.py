import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

data_dir = './data/maps/'

def load_data(data_dir, cnt = 3):
# all directories in data_dir
    dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    # take the first cnt files in each directory
    files = []
    for d in dirs:
        files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.laz') or f.endswith('.las')][:cnt]
    return files

if __name__ == '__main__':
    dataset_files = load_data(data_dir)
    dfs = []
    for f in dataset_files:
        df = pd.read_csv(f, sep="\t+", skiprows=2, engine="python")
        dfs.append(df)
    # # two rows of headers
    # diameter = df["D"]
    # # calculate basal area
    # basal_areas = (diameter / 2) ** 2 * 3.14159
    # forest_area = 1000 * 1000
    # # calculate density
    # forest_basal_area = basal_areas.sum()
    # # density to m^2/ha
    # forest_density = forest_basal_area / forest_area * 10000
    # # calculate SDI
    # sdi = len(diameter) * forest_density ** 2 / forest_basal_area