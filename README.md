# Video Quality Assessment with deep learning
> Video Quality Assessment with deep learning.


## Install

```sh
git clone https://github.com/bdsaglam/icip-vqa
cd icip-vqa
pip install .[dev]
```

## Run experiment

Prepare a config file, see example config in `./scripts/` directory.

```sh
wandb login --relogin
python ./scripts/experiment.py --cfg /path/to/config.json
```
