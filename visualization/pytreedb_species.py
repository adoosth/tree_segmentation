import json
import os
import matplotlib.pyplot as plt

geojson_dir = '/scratch/projects/forestcare/repos/synforest/pytreedb/geojsons'

tree_files = os.listdir(geojson_dir)

species = [f.split('_')[0] for f in tree_files] 