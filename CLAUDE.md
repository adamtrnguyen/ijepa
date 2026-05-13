# I-JEPA (Image-based Joint-Embedding Predictive Architecture)

Meta's self-supervised visual representation learning method. Trains a ViT encoder by predicting masked patch representations in embedding space (no pixel reconstruction).

## Project Structure

```
main.py              # Single-node launcher (multiprocessing)
main_distributed.py  # Multi-node SLURM launcher (submitit)
src/
  train.py           # Training loop (the core entry point)
  helper.py          # Checkpoint loading, model/optimizer init
  transforms.py      # Data augmentation transforms
  models/
    vision_transformer.py  # ViT encoder + predictor architectures
  masks/
    multiblock.py    # Multi-block masking strategy (MaskCollator)
    utils.py         # apply_masks helper
    random.py, default.py  # Alternative mask strategies
  datasets/
    imagenet1k.py    # Original filesystem-based ImageNet loader
    imagenet1k_hf.py # HuggingFace Hub ImageNet loader (active)
  utils/
    distributed.py   # DDP init
    logging.py       # CSVLogger, timers, grad logging
    schedulers.py    # WarmupCosine LR + CosineWD schedulers
    tensors.py       # trunc_normal_, repeat_interleave_batch
configs/             # YAML configs (model, data, mask, optimization params)
```

## Key Architecture

- **Encoder**: Standard ViT (tiny/small/base/large/huge/giant) — processes visible (context) patches
- **Predictor**: Lightweight ViT that predicts target patch representations from context representations
- **Target Encoder**: EMA copy of encoder (no gradients) — generates prediction targets
- **Loss**: Smooth L1 between predictor output and target encoder output on masked regions

## Running

```bash
# Single GPU training
uv run python main.py --fname configs/test_small.yaml --devices cuda:0

# Multi-GPU (single node)
uv run python main.py --fname configs/test_small.yaml --devices cuda:0 cuda:1

# Distributed (SLURM)
uv run python main_distributed.py --fname configs/in1k_vith14_ep300.yaml --folder logs/ --partition <partition>
```

## Data

Currently uses HuggingFace `ILSVRC/imagenet-1k` via `src/datasets/imagenet1k_hf.py`. Requires HF authentication for gated dataset access. Config fields `root_path` and `image_folder` are unused with the HF loader.

## Available Model Sizes

| Name | embed_dim | depth | heads |
|------|-----------|-------|-------|
| vit_tiny | 192 | 12 | 3 |
| vit_small | 384 | 12 | 6 |
| vit_base | 768 | 12 | 12 |
| vit_large | 1024 | 24 | 16 |
| vit_huge | 1280 | 32 | 16 |
| vit_giant | 1408 | 40 | 16 |

## Checkpoint Format

Saved as `.pth.tar` with keys: `encoder`, `predictor`, `target_encoder`, `opt`, `scaler`, `epoch`, `loss`, `batch_size`, `world_size`, `lr`. State dicts include DDP `module.` prefix.

## Dependencies

Managed with `uv`. Core: torch, torchvision, numpy, pyyaml, pillow, datasets (HuggingFace).

## Status

- Training: functional
- Inference/feature extraction: not yet implemented
