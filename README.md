# Sphere-Guided Training of Neural Implicit Surfaces (CVPR 2023)
### [Project Page](https://andreeadogaru.github.io/SphereGuided/) | [Paper](https://arxiv.org/abs/2209.15511)

The official implementation of *Sphere-Guided Training of Neural Implicit Surfaces*. The main code for sphere-guidance is in the `spheres` directory. We include the integration of our method with three different surface reconstruction methods: [NeuS](https://github.com/Totoro97/NeuS), [NeuralWarp](https://github.com/fdarmon/NeuralWarp), and [UNISURF](https://github.com/autonomousvision/unisurf).

![results](teaser.gif)

## Installation

The main dependencies are PyTorch with CUDA and PyTorch3D. We recommend following the official installation instructions from [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). 

Alternatively, we provide a Python=3.9 anaconda environment with PyTorch=1.13, CUDA=11.6, PyTorch3D=0.7, and additional dependencies including numpy, open3d, opencv_python, pymcubes, pyhocon, tensorboard, trimesh:

```
conda env create -f environment.yml
conda activate spheregt
```

Using UNISURF codebase additionally requires compiling some extension modules:
```
cd unisurf
python setup.py build_ext --inplace
```
## Data
For the DTU dataset, as each method used its own version of the data, we recommend using the indications provided in the integrated repositories. For [NeuS](https://drive.google.com/file/d/1zgD-uTLjO8hXcjLqelU444rwS9s9-Syg/view?usp=share_link), the data has to be downloaded from Google Drive and added to the `data` directory. For [NeuralWarp](https://github.com/fdarmon/NeuralWarp#data), it can easily be downloaded with:
```
./NeuralWarp/download_dtu.sh
```
Similar, for [UNISURF](https://github.com/autonomousvision/unisurf#dataset):
```
cd unisurf
./download_dataset.sh
```

For the Realistic Synthetic 360 (Nerf Synthetic) dataset, we preprocess each scene to fit into the bounding sphere of radius one and compute the source views required for NeuralWarp. The processed data can be downloaded from [here](https://drive.google.com/file/d/18R2gc4Pj4jCrGz-_PC3EELh3h5qXjYff/view?usp=sharing) and added to the `data` directory.

For the BlendedMVS dataset, we advise to use the preprocessed data from [NeuS](https://drive.google.com/file/d/1AnMOSOKeIdjbGp-zAK9udb5yFnx6rRby/view?usp=share_link). For reconstruction using NeuralWarp, the source views have to be computed with the suggested method in the original [repository](https://github.com/fdarmon/NeuralWarp/issues/1#issuecomment-1058269029).  

## Training

We provide a straightforward way for launching the training of the methods:
```
./scripts/train.sh METHOD DATASET SCENE EXPNAME [sphere]
```
- `METHOD` specifies one of the three methods: `neus`, `nwarp`, `unisurf`.
- `DATASET` selects one of the datasets downloaded in the previous section: `dtu`, `bmvs`, `nerf`.
- `SCENE` indicates the training scene from the selected dataset.
- `sphere` option uses the proposed sphere-guidance training. 

## Extraction and Evaluation

Using the same parameters as for the training, you can extract the surface from a trained model with:
```
./scripts/extract.sh METHOD DATASET SCENE EXPNAME [sphere]
```
We also provide easy-to-use evaluation scripts for DTU and Nerf Synthetic datasets:
```
./scripts/eval_dtu.sh METHOD SCENE EXPNAME
./scripts/eval_nerf.sh METHOD SCENE EXPNAME
```
 For DTU dataset, the data required for evaluation can be downloaded with:
```
./NeuralWarp/download_dtu_eval.sh
```
## Citation

Should you find this research useful in your work, please cite:

```BibTeX
@article{Dogaru2023SphereGT,
  title={Sphere-Guided Training of Neural Implicit Surfaces},
  author={Dogaru, Andreea and Ardelean, Andrei Timotei and Ignatyev, Savva and Zakharov, Egor and Burnaev, Evgeny},
  journal={arXiv preprint arXiv:2209.15511},
  year={2022}
}
```

## License

Please see the [LICENSE](LICENSE).
