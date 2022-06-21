FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN pip install pytorch-lightning
RUN apt-get update
RUN apt-get install git -y
RUN pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
RUN pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
RUN pip install h5py tables matplotlib

RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html