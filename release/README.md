# When Does Full Fine-Tuning Fail? A Cross-Backbone Study of Visual Adaptation under Label Scarcity

Code, splits and results accompanying the manuscript submitted to *Neurocomputing*.

**Authors:** Jiahao Wu, Sihao Li (Hubei University)

## What this repository contains

```
src/            training and evaluation code
results/        416 per-configuration result files (JSON)
results_all.csv       one row per run
results_summary.csv   mean and standard deviation per cell
```

The datasets are public and deliberately **not** included:

- EuroSAT — 27,000 Sentinel-2 patches, 10 classes (<https://madm.dfki.de/downloads>)
- DTD — 5,640 texture images, 47 classes (<https://www.robots.ox.ac.uk/~vgg/data/dtd/>)

## Setup

```bash
conda create -n adapt python=3.10 -y
conda activate adapt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers timm numpy pandas matplotlib
```

Place the datasets so that the directory layout is:

```
data/eurosat/2750/<class>/*.jpg
data/dtd/<class>/*.jpg
```

EuroSAT is downloaded automatically on first use; DTD must be placed manually.

## Reproducing a single run

```bash
cd src
python -m src.train --dataset eurosat --backbone clip-vit-b16 --method paa \
    --ratio 0.01 --seed 0 --epochs 20
```

Supported backbones: `clip-vit-b16`, `vit-b16`, `vit-small`, `convnext-tiny`, `resnet50`.
Supported methods: `zeroshot`, `coop`, `linear`, `bitfit`, `lora`, `adapter`, `paa`, `full`.

CoOp has its own entry point:

```bash
python -m src.train_coop --dataset eurosat --ratio 0.01 --seed 0
```

## Reproducing the full grid

```bash
python run_grid.py --datasets eurosat dtd --backbone clip-vit-b16 \
    --methods linear bitfit lora adapter paa full \
    --ratios 0.01 0.1 1.0 --seeds 0 1 2
```

Finished configurations are skipped automatically, so the grid can be interrupted
and resumed.

## Results format

Each file in `results/` is one run:

```json
{
  "dataset": "eurosat", "backbone": "clip-vit-b16", "method": "paa",
  "ratio": 0.01, "seed": 0,
  "acc": 0.9412, "trainable_params": 5398785, "train_time_sec": 1208.7
}
```

`acc` is top-1 accuracy on the full test split. Splits are stratified 80/10/10 with
a fixed seed and are identical for every method; they are cached in `splits/`.

## Hardware

All runs were performed on a single NVIDIA RTX 5070 Ti laptop GPU.
Total compute for the 416 configurations was approximately 60 GPU-hours.

## License

Code is released under the MIT License. Results are released under CC BY 4.0.
