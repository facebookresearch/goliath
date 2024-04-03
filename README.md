# ca_body

Codec Avatar Body

### Disclaimer

This is a pre-release.

### Dependencies

See `requirements.txt`/`environment.yaml`

### Repository structure

- `ca_body/` - python source
    * `models` - standalone models
    * `nn` - reusable modules (layers, blocks, learnable, modules, networks)
    * `utils` - reusable utils (functions, modules w/o learnable params)

- `notebooks/` - example notebooks
- `data/` - location of sample data and checkpoints


### Downloading data

TODO: 

```
python unpack_dataset
```

### Training

```
python ca_body/scripts/run_train.py <config.yml>
```

### Evaluation

TODO: 

```
python eval.py <config.yml>
```


### License

See LICENSE.
