# Tree Segmentation Project

This repository contains the code for the master thesis titled "*Enhancing Tree Segmentation on Large Forest Point Clouds with Synthetic Data*"

## Data Generation

The synthetic forest generation workflow is available at:

https://gitlab-ce.gwdg.de/hpc-team-public/synforest

## Data Preparation

To generate a dataset from a LAS point cloud, run `generate_datasets.py` under `preprocess/`.

## Tree Segmentation

There is a directory for each of the four segmentation methods: watershed, Dalponte, AMS3D, and SGPN: `cls_watershed/`, `cls_dalponte/`, `cls_ams3d/`, and `SGPN/`, respectively. The watershed, Dalponte, and AMS3D methods are implemented in `R`, while `SGPN` is implemented in `Python` using `PyTorch`.

To train SGPN, run `SPGN/train.py <config>` where `<config>` is the basename of the config file under `SGPN/config/`.

## Evaluation

Each directory has a `test.py` and `metrics.py` file to run the test on a dataset with or without visualization, respectively.

## Pretrained Networks

The pretrained networks are available under `SGPN/models/`

## Citation

If you used this repository in your research, please cite as follows:

```
@mastersthesis{Doosthosseini2023,
  author      = {Ali Doosthosseini},
  title       = {Enhancing Tree Segmentation in Large Forest Point Clouds with Synthetic Data},
  type        = {Master Thesis},
  pages       = {66},
  school      = {Georg August University of Göttingen},
  year        = {2023},
}
```


