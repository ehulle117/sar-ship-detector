# SAR Ship Detector

Fine-tuning a Faster R-CNN object detector (PyTorch/torchvision) to detect ships
in synthetic aperture radar (SAR) imagery, using the SSDD (SAR Ship Detection
Dataset).

## Motivation

Classical SAR analysis pipelines (thresholding, CFAR, manual annotation) are
the traditional approach to detecting objects in SAR imagery. This project
explores a learned alternative: fine-tuning a pretrained object detector on a
small, labeled SAR dataset, and honestly evaluating where it does and doesn't
outperform classical approaches.

## Dataset

[SSDD (SAR Ship Detection Dataset)](https://github.com/TianwenZhang0825/Official-SSDD)
— ~1,160 SAR image chips, ~2,540 labeled ship instances, VOC-style bounding
box annotations. Not included in this repo; see `data/README.md` for download
instructions.

## Project structure

```
src/
  dataset.py     - PyTorch Dataset for loading SSDD images + boxes
  model.py       - model construction (pretrained Faster R-CNN, 1-class head)
  train.py       - training loop
  evaluate.py    - mAP / precision / recall evaluation
  visualize.py   - draw predicted vs ground-truth boxes on sample images
notebooks/
  01_explore_data.ipynb   - sanity-check dataset and visualize labels
data/            - dataset lives here (gitignored)
outputs/         - checkpoints, metrics, sample prediction images (gitignored)
```

## Status

- [x] Phase 1: Data loading + visualization
- [ ] Phase 2: Baseline training run
- [ ] Phase 3: Evaluation (mAP@0.5, precision/recall, failure case review)
- [ ] Phase 4: Writeup + polish

## Results

_TBD — filled in after Phase 3._

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Designed to run on Google Colab's free GPU tier if you don't have local CUDA.
