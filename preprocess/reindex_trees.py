import laspy
import sys
import os
import numpy as np




if __name__ == "__main__":
    in_path = sys.argv[1]
    if not os.path.exists(in_path):
        raise ValueError("Input path does not exist")
    if not in_path.endswith(".las") and not in_path.endswith(".laz"):
        raise ValueError("Input path must be a LAS file")
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        out_path = in_path.replace(".las", "_reindexed.las")        
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    las = laspy.read(in_path)
    # ground is 9999
    # unclassified is 65436
    tree_ids = np.unique(las.point_source_id)
    # reindex and remove ground and unclassified
    for i, tree_id in enumerate(tree_ids):
        if tree_id in [9999, 65436]:
            continue
        las.point_source_id[las.point_source_id == tree_id] = i
    # remove ground and unclassified
    las = las[las.point_source_id != 9999]
    las = las[las.point_source_id != 65436]
    las.write(out_path)
