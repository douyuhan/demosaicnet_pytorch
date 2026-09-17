# Deep Joint Demosaicking and Denoising
SiGGRAPH Asia 2016

Michaël Gharbi gharbi@mit.edu Gaurav Chaurasia Sylvain Paris Frédo Durand

A minimal pytorch implementation of "Deep Joint Demosaicking and Denoising" [Gharbi2016]

# Installation

From this repo:

```shell
python setup.py install
```

Using pip:

```shell
pip install demosaicnet
```

Then run the demo script with:

```shell
python scripts/demosaicnet_demo.py output
```

To train a dummy model on the demo dataset provided, run:

```shell
python scripts/train.py --data demosaicnet/data/dummy_dataset --checkpoint_dir ckpt
```

To run the pretrained Bayer model on a real raw sensor dump (e.g. to slot it into an existing hardware ISP in place of its demosaic stage), use `scripts/test.py`. The input must be a flat, headerless binary of a single-channel Bayer mosaic that has already been through BLC, raw denoise, LSC and WB (still linear, not yet demosaicked or color-corrected):

```shell
python scripts/test.py <input.raw> <output.bin> --height <H> --width <W> \
    --in_bitwidth 12 --out_bitwidth 8 --bayer_pattern <0=BGGR|1=GBRG|2=GRBG|3=RGGB>
```

It Gamma-encodes the input, runs the network, inverse-Gammas the output back to a pseudo-linear domain, and writes an HWC binary (plus a `.png` preview) meant to be fed onward into your pipeline's own CCM/Gamma/YUV stages.

To build and update the whee:

```shell
pip install wheel twine
make distribution
make upload_distribution
```

# FAQ

- **How is noise handled? Where is the pretrained model?** The noise-aware model is not implementation, see the earlier Caffe implementation for that <https://github.com/mgharbi/demosaicnet_caffe>
- **How do I train this?** The script `scripts/train.py` is a good start to setup your training job, but I haven't tested it yet, I recommend rolling your own.
- **Does this model do BLC / WB / CCM for me?** No. It was trained on ordinary sRGB photos that were synthetically re-mosaicked, so it expects an already BLC/denoise/LSC/WB'd, Gamma-encoded (~2.2) Bayer mosaic as input and produces a Gamma-encoded RGB image as output. It never applies (or expects) a CCM: CCM needs per-pixel RGB, which doesn't exist yet in a Bayer mosaic, so it can't run before demosaicking, and the network doesn't model it afterwards either. See `scripts/test.py` for a pipeline that bridges a real raw sensor dump to this network and back into a hardware ISP's own CCM/Gamma stages.
