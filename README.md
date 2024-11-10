# Goliath

### Together with [Ava-256](https://github.com/facebookresearch/ava-256), part of Codec Avatar Studio

We provide 4 sets of captures; for each subject:
* 1 relightable head capture
* 1 relightable hands capture
* 1 fully clothed capture
* 1 minimally clothed capture
* 1 mobile head capture
* 1 mobile hands capture
* 1 mobile fully clothed capture
* 1 mobile minimally clothed capture

And code to train personalized decoders:
* Relightable Gaussian Codec Avatar heads,
* Relightable hands
* [Universal relightable hands](https://arxiv.org/abs/2401.05334)
* Mesh-based bodies 

![goliath](https://github.com/facebookresearch/goliath/assets/3733964/887bf0a0-a92a-40b7-90bc-a0f9872c787b)

Please refer to the [samples](https://github.com/facebookresearch/goliath/blob/main/samples/) to get a sense of what the data looks like.

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


### Data

Besides camera, views, we also provide segmentations, 3d keypoints, registered and unregistered meshes, as well as light information when available.



https://github.com/facebookresearch/goliath/assets/3733964/3052d8ee-e2b8-48e0-9715-9f0c2e6f6e72



Access to the dataset is currently gated.
Please email `julietamartinez@meta.com`, preferrably from an institutional email, to get access to the data.


### Compiling and installing extensions

```
cd extensions/{mvpraymarch,sgutils,utils}
make
```

### Training

(You may have to add the directory of the codebase to your `PYTHONPATH`)
```
python ca_code/scripts/run_train.py <config.yml>
```
or simply
```
python -m ca_code.scripts.run_train <config.yml>
```

(URHand training) You may need to unwrap images to get `color_mean.png` for static hand assets before launch training in case there is no `color_mean.png` under `uv_images` for the target identity:
```
python -m ca_code.scripts.run_gen_texmean config/urhand_mesh_example.yml
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

For the data itself
```
@article{martinez2024codec,
  author = {Julieta Martinez and Emily Kim and Javier Romero and Timur Bagautdinov and Shunsuke Saito and Shoou-I Yu and Stuart Anderson and Michael Zollhöfer and Te-Li Wang and Shaojie Bai and Chenghui Li and Shih-En Wei and Rohan Joshi and Wyatt Borsos and Tomas Simon and Jason Saragih and Paul Theodosis and Alexander Greene and Anjani Josyula and Silvio Mano Maeta and Andrew I. Jewett and Simon Venshtain and Christopher Heilman and Yueh-Tung Chen and Sidi Fu and Mohamed Ezzeldin A. Elshaer and Tingfang Du and Longhua Wu and Shen-Chi Chen and Kai Kang and Michael Wu and Youssef Emad and Steven Longay and Ashley Brewer and Hitesh Shah and James Booth and Taylor Koska and Kayla Haidle and Matt Andromalos and Joanna Hsu and Thomas Dauer and Peter Selednik and Tim Godisart and Scott Ardisson and Matthew Cipperly and Ben Humberston and Lon Farr and Bob Hansen and Peihong Guo and Dave Braun and Steven Krenn and He Wen and Lucas Evans and Natalia Fadeeva and Matthew Stewart and Gabriel Schwartz and Divam Gupta and Gyeongsik Moon and Kaiwen Guo and Yuan Dong and Yichen Xu and Takaaki Shiratori and Fabian Prada and Bernardo R. Pires and Bo Peng and Julia Buffalini and Autumn Trimble and Kevyn McPhail and Melissa Schoeller and Yaser Sheikh},
  title = {{Codec Avatar Studio: Paired Human Captures for Complete, Driveable, and Generalizable Avatars}},
  year = {2024},
  journal = {NeurIPS Track on Datasets and Benchmarks},
}
```

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

Universal Relightable Hand Avatars
```
@inproceedings{chen2024urhand,
  title={{U}{R}Hand: Universal Relightable Hands},
  author={Zhaoxi Chen and Gyeongsik Moon and Kaiwen Guo and Chen Cao and Stanislav Pidhorskyi and Tomas Simon and Rohan Joshi and Yuan Dong and Yichen Xu and Bernardo Pires and He Wen and Lucas Evans and Bo Peng and Julia Buffalini and Autumn Trimble and Kevyn McPhail and Melissa Schoeller and Shoou-I Yu and Javier Romero and Michael Zollhöfer and Yaser Sheikh and Ziwei Liu and Shunsuke Saito}
  booktitle={CVPR},
  year={2024}
}
```
