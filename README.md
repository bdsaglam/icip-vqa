# Video Quality Assessment with deep learning



## Install

```sh
git clone https://github.com/bdsaglam/icip-vqa
cd icip-vqa
pip install .[dev]
```

## Run experiment

Download [the dataset from Kaggle](https://www.kaggle.com/datasets/bdsaglam/vsqad2022480x270).

Prepare a config file, see example config in `./scripts/` directory.

```sh
wandb login --relogin
python ./scripts/experiment_mtl.py --cfg ./scripts/configs/config-mtl.json
```
