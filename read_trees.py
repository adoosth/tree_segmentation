import sys
import pandas as pd
import json
from copy import deepcopy

tree_template = {
    "template":"{species}",
    "count":1,
    "placement":{
        "method": "position",
        "x": "{x}",
        "y": "{y}"
    }
}

def main():
    config = json.load(open(sys.argv[1]))
    forest_stand_path = config['forest_stand']
    # Read the data from the file
    data = pd.read_csv(forest_stand_path, sep=config['delimiter'])
    print('Data shape: {}'.format(data.shape))
    print('Data head:\n{}'.format(data.head()))
    # Get the number of trees
    n_trees = data.shape[0]
    print('Number of trees: {}'.format(n_trees))
    # normalize x and y coordinates
    data['//X'] = 25 + (data['//X'] - data['//X'].min()) / (data['//X'].max() - data['//X'].min()) * 50
    data['Y'] = 25 + (data['Y'] - data['Y'].min()) / (data['Y'].max() - data['Y'].min()) * 50
    print(data[["//X", "Y", "Z", "Spec"]].head())
    # load helios template as json
    helios_template = json.load(open(config['helios_template']))
    # add trees to helios template
    for tree in data[["//X", "Y", "Z", "Spec"]].values:
        print(tree)
        cur_tree = deepcopy(tree_template)
        cur_tree['template'] = cur_tree['template'].format(species=tree[3])
        cur_tree['placement']['x'] = cur_tree['placement']['x'].format(x=tree[0])
        cur_tree['placement']['y'] = cur_tree['placement']['y'].format(y=tree[1])
        print(cur_tree)
        helios_template['scene']['trees'].append(cur_tree)
    # write helios template
    with open(config['helios_config_out'], 'w') as f:
        json.dump(helios_template, f, indent=4)

if __name__ == '__main__':
    main()