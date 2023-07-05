# TGraphNet: Exploiting Spatio-Temporal Relationshipts for 3D Pose with Graph Neural Networks

This repo contains the implementation of TGraphNet, a spatio-temporal graph convolutional neural network for sequence to sequence 3D pose estimation from videos. A detailed report about TGraphNet and general human pose estimation can be found at the root of the repository.  

<div align="center">
  <img src="https://github.com/Ea0011/TGraphNet/blob/main/reports/demo/walking_cam3.gif" height="400" width="45%" />
  <img src="https://github.com/Ea0011/TGraphNet/blob/main/reports/demo/sittingdown_cam3.gif" height="400" width="45%" />
  <p>
    <em>Ground truth 3D poses is on the left. Reconstruction is on the right</em>
  </p>
</div>

## TGraphNet

<div align="center">
  <img src="https://github.com/Ea0011/TGraphNet/blob/main/reports/diagrams/architecture.png" height="90%" width="90%" />
</div>

TGraphNet is a U-Shaped spatial-temporal graph convolutional network that estimates 3D pose sequence from an input video. It is a 2-stage model, meaning that first a sequence of 2D postures are created from the video, then 2D poses are lifted to 3D poses by TGraphNet.  
As a U-Shaped network, TGraphNet downsamples and upsamples the input sequence for global temporal feature extraction. To exploit relational features of the input 2D poses, TGraphNet utilizes a non-uniform graph convolutions with learnable affinity matrices.

## Environment

* Python 3.6.9
* CUDA 11.1
* PyTorch 1.10.0
* Ubuntu 18.04

Dependencies are listed in *requirements.txt* file in *src* folder.

## Datasets Setup

*Human3.6M* dataset needs to be downloaded from the official website. D3 Positions and D3 angles are necessary for the dataset class to work. To prepare the dataset, extract the archives for each subject and setup the directory tree to look as follows:

```
/path/to/dataset/convert_cdf_to_mat.m
/path/to/dataset/pose/S1/MyPoseFeatures/D3_Positions/Directions 1.cdf
/path/to/dataset/pose/S1/MyPoseFeatures/D3_Positions/Directions.cdf
...
```

The run `convert_cdf_to_mat.m` from MATLAB.

2D *CPN* detectios for Human3.6M dataset can be downloaded from VideoPose3D repository at [](https://github.com/facebookresearch/VideoPose3D).

*MPI-3DHP* and *MPI-3DPW* dataset files are pre-processed and provided in the `./data/` folder.

## Evaluation

Two model *TGraphNet* and *TGraphNet traj* are trained in the scope of the project. TGraphNet predictes only the root relative pose. TGraphNet traj, additionally, predicts the global trajectory of the subject. To run the scripts, make sure that `./src` folder is included in PYTHONPATH so that python recognizes modules in the directory.

### TGraphNet

The model checkpoint is located in `./models/stgcn/root_rel/root_rel.pth.tar` folder. To evaluate the model on Human3.6M using a checkpoint, please mention the path to the checkpoint in `./models/stgcn/root_rel/params.json` in `restore_file` key. Then run:

```python
cd src
python3 models/train_seq.py --exp_suffix=root_rel --run_suffix=1 --exp_desc="testing" --data_dir="/path/to/h36m/dataset" --seed_value=333 --mode=test
```

For evaluation on MPI-3DPW dataset run:

```python
cd src
python3 models/eval_pw3d.py --exp_suffix=root_rel --run_suffix=1 --exp_desc="testing" --seed_value=333 --mode=test
```

For evaluation on MPI-3DHP dataset run:

```python
cd src
python3 models/eval_dhp.py --exp_suffix=root_rel --run_suffix=1 --exp_desc="testing" --seed_value=333 --mode=test
```

### TGraphNet traj

Additionally, TGraphNet (traj) is trained to predict the global trajectory of the person in the 3D space. The architecture is identical to TGraphNet with only difference being that another regression head is used to estimate the position of the root (Hip) joint.

The model checkpoint is located in `./models/stgcn/root_rel/root_rel.pth.tar` folder. To evaluate the model on Human3.6M using a checkpoint, please mention the path to the checkpoint in `./models/stgcn/root_rel/params.json` in `restore_file` key. Then run:

```python
cd src
python3 models/train_seq_traj.py --exp_suffix=global_pos --run_suffix=1 --exp_desc="testing" --data_dir="/path/to/h36m/dataset" --seed_value=333 --mode=test
```

For evaluation on MPI-3DPW dataset run:

```python
cd src
python3 models/eval_pw3d.py --exp_suffix=global_pos --run_suffix=1 --exp_desc="testing" --seed_value=333 --mode=test
```

For evaluation on MPI-3DHP dataset run:

```python
cd src
python3 models/eval_dhp.py --exp_suffix=global_pos --run_suffix=1 --exp_desc="testing" --seed_value=333 --mode=test
```

## Training from Scratch

To train the root relative 3D pose mode, TGraphNet, remove the checkpoint path in `params.json` and run:

```python
cd src
python3 models/train_seq.py --exp_suffix=root_rel --run_suffix=1 --exp_desc="description" --data_dir="/path/to/h36m/dataset" --seed_value=333 --mode=train
```

To train TGraphNet traj, remove the checkpoint path in `params.json` and run:

```python
cd src
python3 models/train_seq_traj.py --exp_suffix=root_rel --run_suffix=1 --exp_desc="description" --data_dir="/path/to/h36m/dataset" --seed_value=333 --mode=train
```

## Acknowledgements

I am eternally grateful to my supervisor at Technical University of Munich, Ms.C. Soubarna Banik for guidance and support during this project.

Dataset setup codes and sequence data generators are built on top of [MixSTE](https://github.com/JinluZhang1126/MixSTE#readme) and [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).
