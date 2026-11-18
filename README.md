# When Does Full Fine-Tuning Fail?

### A Cross-Backbone Study of Visual Adaptation under Label Scarcity

Code, run records and manuscript source for a controlled study of *when* full
fine-tuning of a pretrained vision backbone collapses under label scarcity — and
whether that collapse is a property of the labels or of the backbone.

**Status:** manuscript under review (*Neurocomputing*). This repository is the
artifact released with it.

---

## Headline results

Eight adaptation paradigms, five backbones, two datasets, three label ratios,
three seeds, 1,428 runs. The grid is run twice: once with **one shared learning
rate for every method** (which is what most published comparisons do) and once
with **the rate chosen per method** on the validation split.

At the 1 % budget on the image–text backbone (CLIP ViT-B/16), EuroSAT:

| method | shared rate ($10^{-3}$) | rate tuned |
|---|---|---|
| zero-shot (no adaptation) | 32.1 | 32.1 |
| full fine-tuning | **25.3** | **91.8** |
| best parameter-efficient method (PAA) | 94.1 | 94.1 |

Full fine-tuning is below doing nothing at the shared rate, and 2.3 points behind
the best adapter once its rate is tuned. The same pattern holds across the grid:
under the shared rate the transformer backbones lose 8–69 points to the best
parameter-efficient method, while with the rate tuned all five backbones land
within 2.6 points of it on EuroSAT.

The mechanism is that the accuracy-maximising rate moves with the architecture by
one to two orders of magnitude ($10^{-5}$–$10^{-4}$ for full fine-tuning on a
ViT, $3\times10^{-4}$–$10^{-3}$ on a CNN), while the whole parameter-efficient
family spans only 1.6–8.4 points across the grid and full fine-tuning spans
45–80. A comparison at a single rate therefore manufactures an architecture
effect; that is a plausible account of why the literature disagrees.

What survives tuning is the number of examples per class. DTD at 1 % leaves one
image per class, and there doing nothing beats every trainable method, with full
fine-tuning still 3–10 points behind the parameter-efficient family.

## Repository layout

```
code/
  src/                 adapters, backbones, data pipeline, training engine
  run_grid.py          grid runner (resumable)
  run_lr_sweep.py      runner for the per-method learning-rate grid (resumable)
  make_tables.py       recomputes every table in the paper from the run records,
                       selecting each method's learning rate on the validation split
  prepare_dataset.py   dataset download and ImageFolder conversion
  check_env.py         environment self-check (torch, CUDA, suggested batch size)
  splits/              the exact label-index splits used by every run
  results_all.csv      one row per run: dataset, backbone, method, ratio, seed,
                       learning rate, acc, val, params, train time (1,440 rows)
paper/
  main_journal.tex     manuscript source (Elsevier elsarticle)
  refs.bib
  figures/             the two figures included in the manuscript
```

The per-run JSON records (one file per configuration, including the epoch-by-epoch
validation history) are attached to the [latest release](../../releases); the CSV
in `code/` is their flattened form and is enough to reproduce every table.

## Coverage

| backbones | methods | label ratios | seeds |
|---|---|---|---|
| CLIP ViT-B/16, supervised ViT-B/16, ViT-S/16, ConvNeXt-T, ResNet-50 | zero-shot, linear probing, full FT, BitFit, LoRA, residual adapters, PAA, CoOp | 1 %, 10 %, 100 % | 0, 1, 2 |

Not every combination is populated, and the gaps are deliberate rather than
missing data:

* CoOp is a prompt-learning method and only applies to the image–text backbone,
  so it was run on CLIP only (18 runs).
* Zero-shot has no hyperparameters to vary, so it was run once per dataset (2 runs).
* The four non-CLIP backbones were run at the 1 % and 10 % budgets only (12 runs
  per method); the paper's claim concerns the label-scarce regime, and the
  saturated end of the curve is well covered by prior work.

CLIP rows therefore have 18 runs per method (2 datasets × 3 ratios × 3 seeds);
the other backbones have 12 (2 datasets × 2 ratios × 3 seeds).

## Reproducing

```bash
git clone <this-repo-url>
cd ccfc-cross-backbone

pip install -r code/requirements.txt
# install torch / torchvision separately, matched to your CUDA version —
# letting pip resolve them from PyPI usually gets you a CPU-only build
python code/check_env.py
```

Get the data (EuroSAT downloads automatically on first use; DTD needs one step):

```bash
cd code
python prepare_dataset.py --name dtd        # ~600 MB, 47 texture classes
```

Run a single configuration:

```bash
python -m src.train --dataset eurosat --backbone clip-vit-b16 --method full \
    --ratio 0.01 --seed 0 --epochs 30 --out results
```

Smoke-test the whole pipeline in about two minutes, then run the grid:

```bash
python run_grid.py --quick                          # 1 epoch, 2 methods
python run_grid.py --datasets eurosat dtd \
    --backbone clip-vit-b16 --methods full paa adapter
```

The runner takes `--backbone`, `--datasets`, `--methods`, `--ratios` and
`--seeds`; its default dataset list is a legacy one, so pass the datasets
explicitly for the paper's grid. To rebuild the LaTeX tables and the figures
from the JSON records:

```bash
python -m src.aggregate --results ./results --out ./tables
python -m src.plot      --results ./results --out ./figures
```

## Result record format

One JSON file per run, named
`{dataset}_{backbone}_{method}_r{label ratio}_s{seed}.json`, e.g.
`dtd_clip-vit-b16_adapter_r0.01_s0.json`. Fields:

| field | meaning |
|---|---|
| `acc`, `macro_acc` | test accuracy, overall and macro-averaged over classes |
| `best_val_acc`, `best_epoch` | early-stopping selection |
| `history` | per-epoch `train_loss` and `val_acc` |
| `train_time_sec` | wall-clock training time |
| `trainable_params`, `total_params` | parameter counts |
| `method`, `dataset`, `backbone`, `ratio`, `seed` | the configuration |

`results_ablation/` and `results_lr/` append a suffix (`_anchor_noproto`,
`_lr0.0003`, …) to the same convention.

## Data

Not included, for size and licensing reasons:

* **EuroSAT** — `torchvision.datasets.EuroSAT`, downloaded automatically.
* **DTD** (Describable Textures) — fetched by `prepare_dataset.py --name dtd`.

Both are public research datasets; cite the original papers (Helber et al. 2019;
Cimpoi et al. 2014) if you use them.

## Citation

```bibtex
@article{wu2026finefull,
  title   = {When Does Full Fine-Tuning Fail? A Cross-Backbone Study of
             Visual Adaptation under Label Scarcity},
  author  = {Wu, Jiahao and Li, Sihao},
  journal = {Neurocomputing (under review)},
  year    = {2026}
}
```

## License

Code (`code/`) is released under the MIT License — see [LICENSE](LICENSE).
The manuscript, its figures and the submission material (`paper/`) remain
© the authors, all rights reserved.

## Contact

Jiahao Wu — 202531123002081@stu.hubu.edu.cn
Hubei University, Wuhan, Hubei, China
