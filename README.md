# Edge-Selective Feature Weaving for Point Cloud Matching
This repository contains a PyTorch-lightning implementation of the ESFW module proposed in our paper Edge-Selective Feature Weaving for Point Cloud Matching https://arxiv.org/pdf/2202.02149v1.pdf. 

## Note
- The code with RoITr will be published soon. (4th Dec. 2023)
- Our code is created based on https://github.com/ZENGYIMING-EAMON/CorrNet3D

## Installation
```
conda create --name corrnet3d python=3.8
conda activate corrnet3d
pip install pytorch-lightning==1.1.6
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
conda install torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install h5py
pip install tables
pip install matplotlib
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org whl/torch-1.10.0+cu102.html
```

## Dockerfile
You can use my docker file
```
docker build ./ -t {image_name}
```

## Datasets
Download from https://github.com/ZENGYIMING-EAMON/CorrNet3D


## Train
```python:sample
uncomment 'cli_main()' in lit_corrnet3d_ESFW.py
python lit_corrnet3d_ESFW.py --batch_size=10 --data_dir=./trainset.h5 --test_data_dir=./testset.h5 --num_gpus <gpunum>
```

## Test
To test on the whole testing set, run:
```python:sample
uncomment 'cli_main_test_()' in lit_corrnet3d_ESFW.py
python lit_corrnet3d_ESFW.py --batch_size=1 --ckpt_user=<ckpt_PATH> --data_dir=./trainset.h5 --test_data_dir=./testset.h5 -- num_gpus <gpunum>
```
---
## How to cite
```
@article{yanagi2022edge,
  title={Edge-selective feature weaving for point cloud matching},
  author={Yanagi, Rintaro and Atsushi, Hashimoto and Shusaku, Sone and Naoya, Chiba and Jiaxin, Ma and Yoshitaka, Ushiku},
  journal={arXiv preprint arXiv:2202.02149},
  year={2021}
}
```
