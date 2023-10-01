import os
import sys
import numpy as np
import pandas as pd
import laspy
import tqdm

def generate_boxes(cnt, size, forest_size):
    boxes = []
    for i in range(cnt):
        x = np.random.randint(0, forest_size[0] - size[0])
        y = np.random.randint(0, forest_size[1] - size[1])
        boxes.append([x, y, size[0], size[1]])
    return boxes

def build_dataset(forests, dataset_dir, cnt, size, out_file = 'boxes.csv', test = False, sanity=False, epochs = 1000):
    forest_sizes = []
    forest_shifts = []
    for forest_id in forests:
        print("Checking forest {}".format(forest_id))
        laz_path = os.path.join(dataset_dir, forest_id + '.laz')
        if not os.path.exists(laz_path):
            laz_path = os.path.join(dataset_dir, forest_id + '.las')
        with laspy.open(laz_path) as f:
            print("Total points: ", f.header.point_count)
            cur_forest_size = (f.header.max[0] - f.header.min[0], f.header.max[1] - f.header.min[1])
            cur_forest_shift = (f.header.min[0], f.header.min[1])
            print("Forest size: ", cur_forest_size)
            forest_sizes.append(cur_forest_size)
            forest_shifts.append(cur_forest_shift)
    # decide how many boxes to generate for each forest
    forest_cnt = 1 if test or sanity else len(forests) - 1 
    df = pd.DataFrame(columns=['epoch', 'forest_id', 'x', 'y', 'width', 'height'])
    if sanity:
        epoch = 0
        cur_boxes = 0
        for i, forest_id in enumerate([forests[-1]] if test or sanity else forests[:-1]):
            if test or sanity:
                i = len(forests) - 1
            cur_cnt = 1
            cur_boxes += cnt
            boxes = generate_boxes(cur_cnt, size, forest_sizes[i])
            # implement shift
            boxes = [[box[0] + forest_shifts[i][0], box[1] + forest_shifts[i][1], box[2], box[3]] for box in boxes]
            #df = df.append(pd.DataFrame([[epoch, forest_id] + box for box in boxes], columns=['epoch', 'forest_id', 'x', 'y', 'width', 'height']))
            df = pd.concat([df, pd.DataFrame([[0, forest_id] + box for box in boxes], columns=['epoch', 'forest_id', 'x', 'y', 'width', 'height'])])
        df = pd.concat([df] * cnt * epochs)
        df.reset_index(drop=True, inplace=True)
        # repeat the same boxes for all epochs, change the epoch number. Every cnt boxes, change the epoch number
        df['epoch']  = (df.index // cnt) + 1
        df.to_csv(os.path.join(dataset_dir, out_file), index=False)
        return
    
    for epoch in tqdm.tqdm(range(1, epochs+1)):
        cur_boxes = 0
        for i, forest_id in enumerate([forests[-1]] if test else forests[:-1]):
            if test:
                i = len(forests) - 1
            cur_cnt = int(cnt / forest_cnt) if cur_boxes + int(cnt / forest_cnt) <= cnt else cnt - cur_boxes
            cur_cnt = cur_cnt if cur_cnt > 0 else 1
            cur_boxes += cur_cnt
            boxes = generate_boxes(cur_cnt, size, forest_sizes[i])
            # implement shift
            boxes = [[box[0] + forest_shifts[i][0], box[1] + forest_shifts[i][1], box[2], box[3]] for box in boxes]
            #df = df.append(pd.DataFrame([[epoch, forest_id] + box for box in boxes], columns=['epoch', 'forest_id', 'x', 'y', 'width', 'height']))
            df = pd.concat([df, pd.DataFrame([[epoch, forest_id] + box for box in boxes], columns=['epoch', 'forest_id', 'x', 'y', 'width', 'height'])])

    df.to_csv(os.path.join(dataset_dir, out_file), index=False)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        dataset_basename = sys.argv[1]
        dataset_dir = os.path.join(data_dir, dataset_basename)
        if not os.path.exists(dataset_dir):
            print("Error: dataset path not found: ", dataset_dir)
            exit(0)
    else:
        print("Usage: python generate_datasets.py <dataset_basename>")
        exit(0)
    forests = [os.path.basename(f)[:-4] for f in os.listdir(dataset_dir) if f.endswith('.laz') or f.endswith('.las')] # Old
    # This is for folders:
    #forests = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    # sort forests alphabetically
    forests.sort()
    #build_dataset(forests, dataset_dir, 400, (30,30), 'train_30.csv', epochs = 1000)
    build_dataset(forests, dataset_dir, 400, (30,30), 'train_30.csv', test=True, epochs = 1000)
    build_dataset(forests, dataset_dir, 8, (30,30), 'test_30.csv', test = True, epochs = 100)
    build_dataset(forests, dataset_dir, 8, (30,30), 'sanity_30.csv', sanity=True, epochs = 1000)
