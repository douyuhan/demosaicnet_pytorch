# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Minimal PyTorch implementation of "Deep Joint Demosaicking and Denoising" (Gharbi et al., SIGGRAPH Asia 2016). It provides two pretrained CNNs (`BayerDemosaick`, `XTransDemosaick`) that reconstruct full-color images from raw sensor mosaics, plus the dataset/mosaicking utilities and a bare-bones training loop used to reproduce or fine-tune the models. The noise-aware variant from the paper is NOT implemented here (see the Caffe implementation at github.com/mgharbi/demosaicnet_caffe for that).

**Original implementation**: this repo is a PyTorch port of the original Caffe implementation, which lives locally at `../demosaicnet_caffe` (sibling directory, i.e. `AIDMSC_research/demosaicnet_caffe`). That directory has more detailed documentation (including its own `CLAUDE.md` and the original paper PDF) and is a useful reference for algorithm details, pretrained model provenance, or behavior that is under-documented here.

## Commands

Install (editable, from repo root):
```shell
python setup.py install
```

Run the inference demo (writes bayer/xtrans mosaick + reconstruction TIFFs):
```shell
python scripts/demosaicnet_demo.py output
```

Train (note: `scripts/train.py` is unverified/experimental per the README — expect to adapt it):
```shell
python scripts/train.py --data demosaicnet/data/dummy_dataset --checkpoint_dir ckpt
```

Evaluate a checkpoint (positional args, not `--data`/`--checkpoint_dir`):
```shell
python scripts/eval.py <data_root> <checkpoint_dir>
```

Run the pretrained Bayer model on real hardware raw dumps (see Architecture below for the expected pipeline and required preprocessing). Recursively finds every `.raw` file under `input_dir` and mirrors the same relative layout under `output_dir`:
```shell
python scripts/test.py <input_dir> <output_dir> --height <H> --width <W> \
    --in_bitwidth 12 --out_bitwidth 8 --bayer_pattern <0=BGGR|1=GBRG|2=GRBG|3=RGGB>
```

Tests:
```shell
make test        # runs `py.test tests`
```
There is currently no `tests/` directory in the repo, so this target will fail until tests are added.

Build docs (Sphinx, in `docs/`):
```shell
make docs
```

Build and publish a release:
```shell
pip install wheel twine
make distribution         # python setup.py sdist bdist_wheel; twine check
make upload_distribution  # twine upload dist/*
```

## Architecture

The installable package is `demosaicnet/`; `scripts/` contains standalone entry points that import it.

- **`demosaicnet/mosaic.py`** — pure mosaicking functions (`bayer`, `xtrans`, `xtrans_cell`). Work on both `np.ndarray` and `th.Tensor` inputs with shape `[..., c, h, w]`; build a binary color-filter-array mask and multiply it with the image. `xtrans_cell` defines the repeating 6x6 X-Trans pattern by explicit pixel-position lists per channel.
- **`demosaicnet/modules.py`** — the two `nn.Module` networks:
  - `BayerDemosaick`: packs the 2x2 Bayer mosaic to 4 channels (`pack_mosaic`), runs a stack of `depth` conv+ReLU layers at 1/4 resolution, splits the final feature map into "filters" and "masks" (elementwise multiplied — a departure from the paper, kept for compatibility with the released weights), predicts a residual, upsamples back to full res with a grouped `ConvTranspose2d`, then concatenates with the (cropped) original mosaic as a skip connection before a small `fullres_processor` head.
  - `XTransDemosaick`: no downsampling; runs full-res convs directly on the 3-channel mosaic, then the same skip-connection + `fullres_processor` pattern.
  - Both load pretrained weights from `demosaicnet/data/{bayer,xtrans}.pth` by default (`pretrained=True`), and assert the fixed `depth`/`width` those weights were trained with (15/64 for Bayer, 11/64 for X-Trans). Set `pretrained=False` to train from scratch with different depth/width.
  - `_crop_like` centers-crops a tensor to match another's spatial size; used wherever `pad=False` (the default — "valid" convolutions shrink feature maps) so the skip-connection tensors line up.
- **`demosaicnet/dataset.py`** — `Dataset(TorchDataset)` reads a `filelist.txt` per subset (`train`/`val`/`test`) under `root/<subset>/`, loads each image, and applies `bayer()`/`xtrans()` on the fly in `__getitem__` to produce `(mosaic, groundtruth)` pairs. `download=True` fetches and reassembles the ~80GB multi-part zip dataset from MIT's servers (`_download`, with per-part MD5 checks against the hardcoded `CHECKSUMS` table) — this is expensive and should not be triggered incidentally.
- **`demosaicnet/utils.py`** — training scaffolding independent of the model: `BasicArgumentParser` (common CLI flags used by `scripts/train.py`), `ModelInterface` (abstract adapter a training script implements — `training_step`, `validation_step`, etc.), `Trainer` (epoch/batch loop that drives a `ModelInterface` and dispatches to `Callback` hooks, with SIGINT handling for graceful stop), `Checkpointer` (saves/loads model+optimizer+scheduler state and arbitrary metadata, tracks checkpoint history for resume), and callbacks (`CheckpointingCallback`, `ProgressBarCallback`, `KeyedCallback`) used by `scripts/train.py`.
- **`demosaicnet/__init__.py`** re-exports the public API: `BayerDemosaick`, `XTransDemosaick`, `bayer`, `xtrans`, `xtrans_cell`, dataset symbols, and `utils` as a submodule.
- **`scripts/train.py`** wires the above together: defines `DemosaicnetInterface(ModelInterface)` with an MSE loss and a PSNR metric, resumes model hyperparameters (`depth`/`width`/`mode`) from an existing checkpoint's metadata when present, otherwise takes them from CLI args.
- **`scripts/eval.py`** loads a checkpoint's saved metadata to reconstruct the right model architecture, runs it over the `test` subset, and reports average PSNR/MSE. Its CLI takes `data` and `checkpoint_dir` as positional arguments (different convention from `train.py`'s flags).
- **`scripts/demosaicnet_demo.py`** is a standalone smoke test: loads the packaged `data/test_input.png`, mosaics it both ways, runs both pretrained models, and writes mosaic/result TIFFs to an output directory.
- **`scripts/test.py`** runs `BayerDemosaick` (Bayer only, not X-Trans) on real raw sensor dumps instead of a synthetic mosaic. It walks `input_dir` recursively for `.raw` files and writes each result under `output_dir` at the same relative path (extension swapped to `.bin`/`.png`); the model is loaded once and reused across all files, which must all share the same `--height`/`--width`/`--in_bitwidth`/`--bayer_pattern`. The network was trained on synthetically re-mosaicked, already-ISP'd sRGB images, so it expects Gamma-encoded (~2.2) input and produces Gamma-encoded output — it never sees or produces a CCM, since CCM needs per-pixel RGB that doesn't exist yet in a Bayer mosaic (can't run before demosaicking) and isn't modeled after it either. Each input is a flat, headerless binary of a single-channel mosaic that must already have BLC/raw-denoise/LSC/WB applied upstream (still linear at this point). Per-file pipeline: normalize by `--in_bitwidth` → Gamma-encode (hardcoded `x**(1/2.2)`) → align the sensor's native CFA phase to the network's fixed GRBG assumption by flipping the array left-right and/or up-down as needed (a 2-periodic pattern flipped along an even axis swaps parity losslessly — no crop/pad approximation, and it's an error if height/width aren't even) → `BayerDemosaick(pretrained=True, pad=True)` (keeps output size == input size) → invGamma (`x**2.2`) back to a pseudo-linear domain → undo the same flips → quantize to `--out_bitwidth` and write an HWC binary, plus a same-named `.png` (still linear-domain, so it looks dark in a normal viewer — meant for pixel-level sanity checks, not display). Meant to be spliced into an existing hardware ISP in place of its demosaic stage, feeding the output onward into that pipeline's own CCM/Gamma/YUV stages.

Pretrained weights and the demo image ship inside the package (`demosaicnet/data/*.pth`, `test_input.png`) via `MANIFEST.in`/`include_package_data`, and are located at runtime with `importlib.resources.files("demosaicnet")` (requires Python 3.9+; previously used the now-removed `pkg_resources.resource_filename`).

## Environment setup notes

- After cloning, run `pip install -e .` (or `python setup.py install`) from the repo root before running anything in `scripts/` directly as a file — otherwise `import demosaicnet` fails, since a script's own directory (not the repo root) is what lands on `sys.path`.
- `demosaicnet/dataset.py` unconditionally imports `wget` (only used by the optional dataset auto-download) — install it even if you never call `Dataset(download=True)`.
