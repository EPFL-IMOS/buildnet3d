# BuildNet3D
Official code for "Exploiting Semantic Scene Reconstruction for Estimating Building Envelope Characteristics" (Building and Environment 2025)

## Multi-Modal Imageset Generation
This repository uses `BlenderProc` to generate multi-modal image data from 3D building models. The rendered outputs include RGB, depth maps, surface normals, semantic labels, and instance segmentations. We implement a simple rule-based sampling method to randomly place camera viewpoints while ensuring the entire object remains within the view. More details are provided [here](bproc_generator/README.md). 

The generated **buildnet3d image dataset** is available [here](https://zenodo.org/records/15075790).



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
