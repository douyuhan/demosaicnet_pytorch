#!/usr/bin/env python
"""Run the pretrained BayerDemosaick model on real hardware raw dumps.

Recursively finds every `.raw` file under --input_dir and writes its result
under --output_dir, mirroring the same relative directory layout.

Expected input: flat, headerless binary files each holding a single-channel
Bayer mosaic that has already gone through BLC, raw denoise, LSC and WB in
the hardware ISP (i.e. linear, black-level-corrected, white-balanced sensor
data -- just not demosaicked yet). All files must share the same --height,
--width, --in_bitwidth and --bayer_pattern. The network was trained on
synthetically re-mosaicked sRGB (gamma-encoded) images, so for each file
this script:

  1. normalizes the raw samples to [0, 1] using --in_bitwidth,
  2. Gamma-encodes them (fixed gamma = 2.2) to match the network's domain,
  3. runs BayerDemosaick,
  4. inverts the Gamma to bring the result back to a pseudo-linear domain,
  5. quantizes to --out_bitwidth and writes an HWC binary (plus a PNG next
     to it for a quick visual check).

No CCM is applied on either side: CCM needs per-pixel RGB, which doesn't
exist yet in a Bayer mosaic, so it cannot run before the network -- and the
network doesn't produce/expect one either. The output is meant to be fed to
the hardware pipeline's own downstream CCM/Gamma/YUV stages.
"""
import argparse
import os

import numpy as np
import torch as th
import imageio

import demosaicnet


_GAMMA = 2.2

_PATTERN_NAMES = {0: "BGGR", 1: "GBRG", 2: "GRBG", 3: "RGGB"}

# (flip_vertical, flip_horizontal) that turns each pattern into GRBG.
# The Bayer pattern is 2-periodic, so flipping an even-length axis swaps
# that axis's parity everywhere -- e.g. RGGB [[R,G],[G,B]] flipped
# left-right becomes [[G,R],[B,G]] == GRBG, using only real pixels (no crop,
# no padding). Flipping the network's output the same way undoes it exactly.
_FLIP_TO_GRBG = {
    "GRBG": (False, False),
    "RGGB": (False, True),
    "BGGR": (True, False),
    "GBRG": (True, True),
}


def _find_raw_files(input_dir):
    raw_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".raw"):
                raw_files.append(os.path.join(root, f))
    raw_files.sort()
    return raw_files


def _read_raw(path, height, width, bitwidth):
    dtype = np.uint8 if bitwidth <= 8 else np.uint16
    raw = np.fromfile(path, dtype=dtype)
    if raw.size != height * width:
        raise ValueError(
            "Expected {}x{}={} samples, got {} from {}".format(
                height, width, height * width, raw.size, path))
    return raw.reshape(height, width).astype(np.float32)


def _align_to_grbg(raw, pattern_name):
    h, w = raw.shape
    if h % 2 or w % 2:
        # pack_mosaic (demosaicnet/modules.py) needs even height/width, and
        # the flip trick below only preserves periodicity on even axes.
        raise ValueError(
            "Expected even height/width for a 2x2 Bayer pattern, got "
            "{}x{}".format(h, w))
    flip_v, flip_h = _FLIP_TO_GRBG[pattern_name]
    if flip_v:
        raw = raw[::-1, :]
    if flip_h:
        raw = raw[:, ::-1]
    return np.ascontiguousarray(raw), (flip_v, flip_h)


def _undo_flip(img_hwc, flips):
    flip_v, flip_h = flips
    if flip_h:
        img_hwc = img_hwc[:, ::-1, :]
    if flip_v:
        img_hwc = img_hwc[::-1, :, :]
    return np.ascontiguousarray(img_hwc)


def _process_one(model, device, raw_path, out_bin_path, args):
    raw = _read_raw(raw_path, args.height, args.width, args.in_bitwidth)

    pattern_name = _PATTERN_NAMES[args.bayer_pattern]
    raw, flips = _align_to_grbg(raw, pattern_name)

    in_max = 2 ** args.in_bitwidth - 1
    linear = np.clip(raw / in_max, 0.0, 1.0)
    gamma_encoded = linear ** (1.0 / _GAMMA)

    mosaic3 = np.stack([gamma_encoded] * 3, axis=0)  # [3, h, w]
    mosaic = demosaicnet.bayer(mosaic3)  # zeroes out non-sampled channels (GRBG)

    mosaic_t = th.from_numpy(mosaic).unsqueeze(0).to(device)
    with th.no_grad():
        out = model(mosaic_t).squeeze(0).cpu().numpy()  # [3, h, w]

    out = np.clip(out, 0.0, 1.0)
    linear_out = out ** _GAMMA  # invGamma back to pseudo-linear domain

    out_hwc = np.transpose(linear_out, [1, 2, 0])  # [h, w, 3]
    out_hwc = _undo_flip(out_hwc, flips)  # restore original orientation

    out_max = 2 ** args.out_bitwidth - 1
    out_dtype = np.uint8 if args.out_bitwidth <= 8 else np.uint16
    quantized = np.round(np.clip(out_hwc, 0.0, 1.0) * out_max).astype(out_dtype)

    os.makedirs(os.path.dirname(out_bin_path), exist_ok=True)
    quantized.tofile(out_bin_path)

    png_path = os.path.splitext(out_bin_path)[0] + ".png"
    png = np.round(np.clip(out_hwc, 0.0, 1.0) * 255.0).astype(np.uint8)
    imageio.imsave(png_path, png)

    return png_path


def main(args):
    raw_files = _find_raw_files(args.input_dir)
    if not raw_files:
        raise ValueError("No .raw files found under {}".format(args.input_dir))

    model = demosaicnet.BayerDemosaick(pretrained=True, pad=True).to(args.device)
    model.eval()

    out_dtype_name = "uint8" if args.out_bitwidth <= 8 else "uint16"
    for raw_path in raw_files:
        rel_path = os.path.relpath(raw_path, args.input_dir)
        out_bin_path = os.path.join(
            args.output_dir, os.path.splitext(rel_path)[0] + ".bin")

        png_path = _process_one(model, args.device, raw_path, out_bin_path, args)

        print("{} -> {} ({}x{}x3, {}) + {}".format(
            raw_path, out_bin_path, args.height, args.width,
            out_dtype_name, png_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", help="root directory to search recursively for .raw files.")
    parser.add_argument("output_dir", help="root directory to mirror results into.")
    parser.add_argument("--height", type=int, required=True, help="image height (imgH).")
    parser.add_argument("--width", type=int, required=True, help="image width (imgW).")
    parser.add_argument("--in_bitwidth", type=int, default=12,
                        help="bit depth of the input raw samples.")
    parser.add_argument("--out_bitwidth", type=int, default=8,
                        help="bit depth for the output binary.")
    parser.add_argument("--bayer_pattern", type=int, required=True, choices=[0, 1, 2, 3],
                        help="0=BGGR, 1=GBRG, 2=GRBG, 3=RGGB.")
    parser.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
