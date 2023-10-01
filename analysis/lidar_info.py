import laspy
import sys
import os
import numpy as np

# Read in the lidar data


if __name__ == '__main__':
    # Read in the lidar data
    path = __file__
    cur_dir = os.path.dirname(path)
    data_dir = os.path.join(os.path.dirname(cur_dir), 'data')
    dataset_dir = os.path.join(data_dir, sys.argv[1])
    id = int(sys.argv[2])
    laz_path  = os.path.join(dataset_dir, sys.argv[1] + '-' + str(id) + '-ULS.laz')
    # read with chunk size 0
    with laspy.open(laz_path) as f:  
        las = next(f.chunk_iterator(10))
        point_count = f.header.point_count
        # print with , at every three digits
        print("Point count:", "{:,}".format(point_count))
        print("Point format:", f.header.point_format)