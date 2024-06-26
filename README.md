# Goliath

### Part of Codec Avatar Studio

Code for Codec Avatar Family.

### Disclaimer

This is a pre-release.

### Dependencies

See `requirements.txt`/`environment.yaml`

### Repository structure

- `ca_code/` - python source
    * `loss` - loss functions
    * `models` - standalone models
    * `nn` - reusable modules (layers, blocks, learnable, modules, networks)
    * `utils` - reusable utils (functions, modules w/o learnable params)

- `notebooks/` - example notebooks
- `extensions/` - CUDA extensions
- `data/` - location of sample data and checkpoints

### Downloading data

Access to the dataset is currently gated.
Please email `julietamartinez@meta.com`, preferrably from an institutional email, to get access to the data.

### Compiling and installing extensions

```
cd extensions/{mvpraymarch,sgutils,utils}
make
```

### Training

```
python ca_code/scripts/run_train.py <config.yml>
```

### Visualization (Relighting)

```
python ca_code/scripts/run_vis_relight.py <config.yml>
```

### Evaluation

TODO:

```

```



### License

See LICENSE.


### Citation

If you use this repository, please cite relevant paper(s).

Full-body Avatars
```
@article{bagautdinov2021driving,
  title={Driving-signal aware full-body avatars},
  author={Bagautdinov, Timur and Wu, Chenglei and Simon, Tomas and Prada, Fabi{\'a}n and Shiratori, Takaaki and Wei, Shih-En and Xu, Weipeng and Sheikh, Yaser and Saragih, Jason},
  journal={ACM Transactions on Graphics (TOG)},
  volume={40},
  number={4},
  pages={1--17},
  year={2021},
  publisher={ACM New York, NY, USA}
}
```

Relightable Head Avatars
```
@inproceedings{saito2024rgca,
  author = {Shunsuke Saito and Gabriel Schwartz and Tomas Simon and Junxuan Li and Giljoo Nam},
  title = {Relightable Gaussian Codec Avatars},
  booktitle = {CVPR},
  year = {2024},
}
```

Relightable Hand Avatars
```
@inproceedings{iwase2023relightablehands,
  title={Relightablehands: Efficient neural relighting of articulated hand models},
  author={Iwase, Shun and Saito, Shunsuke and Simon, Tomas and Lombardi, Stephen and Bagautdinov, Timur and Joshi, Rohan and Prada, Fabian and Shiratori, Takaaki and Sheikh, Yaser and Saragih, Jason},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16663--16673},
  year={2023}
}
```
