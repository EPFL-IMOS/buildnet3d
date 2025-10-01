# BuildNet3D
Official code for "Exploiting Semantic Scene Reconstruction for Estimating Building Envelope Characteristics" (Building and Environment 2025)
## 3D Semantic Building Reconstruction
This repository extends [nerfstudio](https://docs.nerf.studio/) (v1.1.3) to reconstruct building envelope surface meshes with **appearance**, **geometry**, and **semantic** properties using SDF-based representations.

### Usage
```bash
# Reconstruction
python buildnet3d/scripts/reconstruct.py --model-type semantic-sdf --data <DIR_TO_DATA>
```


## Multi-Modal Imageset Generation
This repository uses `BlenderProc` to generate multi-modal image data from 3D building models. The rendered outputs include RGB, depth maps, surface normals, semantic labels, and instance segmentations. We implement a simple rule-based sampling method to randomly place camera viewpoints while ensuring the entire object remains within the view. More details are provided [here](bproc_generator/README.md). 

The generated **buildnet3d image dataset** is available [here](https://zenodo.org/records/15075790).

## Installation
```bash
# Clone repository
​git clone --recursive git@github.com:EPFL-IMOS/buildnet3d.git

# Install dependencies
cd buildnet3d/nerfstudio
pip install -e .
# Install tinycudann
TCNN_CUDA_ARCHITECTURES=<YOUR_ARCH> \
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch 
```
> *The code was tested on A100 GPU with Python 3.10, PyTorch 2.0.1, and CUDA 11.8.*
## Citation
If you find this repository or the associated dataset useful, please cite:
```
@article{XU2025112731,
      title = {Exploiting semantic scene reconstruction for estimating building envelope characteristics},
      journal = {Building and Environment},
      volume = {275},
      pages = {112731},
      year = {2025},
      issn = {0360-1323},
      doi = {https://doi.org/10.1016/j.buildenv.2025.112731},
      url = {https://www.sciencedirect.com/science/article/pii/S0360132325002136},
      author = {Chenghao Xu and Malcolm Mielle and Antoine Laborde and Ali Waseem and Florent Forest and Olga Fink},
}
```
