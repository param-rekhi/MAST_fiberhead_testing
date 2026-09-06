#!/usr/bin/env python3
"""Generate synthetic fiberhead frames with known truth for testing.

Produces a 1440x1080 TIFF containing a single Gaussian spot of known position,
FWHM and total counts on a flat pedestal, with Poisson and read noise applied and
the result clipped at the 10-bit saturation level of the test camera. Used to
check that `aperture_photometry_single.py` recovers the injected values.

Example
-------
    python tests/make_test_frame.py /tmp/frame.tif --peak-counts 600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

IMAGE_SHAPE = (1080, 1440)  # (ny, nx)
SATURATION_ADU = 1023
GAIN_E_PER_ADU = 1.0
READ_NOISE_E = 0.46


def make_frame(shape=IMAGE_SHAPE, x=520.4, y=610.7, fwhm=11.0, peak_counts=600.0,
               pedestal=12.0, gain=GAIN_E_PER_ADU, read_noise_e=READ_NOISE_E,
               saturate=True, seed=42):
    """Build a noisy frame containing one Gaussian spot.

    Parameters
    ----------
    shape : tuple of int, optional
        Image shape as (ny, nx).
    x, y : float, optional
        True centroid of the spot in pixel coordinates.
    fwhm : float, optional
        True FWHM of the spot in pixels.
    peak_counts : float, optional
        Peak amplitude of the noiseless spot above the pedestal, in counts.
    pedestal : float, optional
        Flat background level in counts.
    gain : float, optional
        Detector gain in e-/ADU, used to scale the Poisson noise.
    read_noise_e : float, optional
        RMS read noise in e-.
    saturate : bool, optional
        If True, clip the frame at `SATURATION_ADU` as the real sensor would.
    seed : int, optional
        Seed for the random number generator, so frames are reproducible.

    Returns
    -------
    tuple
        (image, truth) where `image` is a uint16 array and `truth` is a dict of
        the injected values, including the total counts in the noiseless spot.
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    yy, xx = np.mgrid[0:ny, 0:nx]
    spot = peak_counts * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
    total_counts = 2.0 * np.pi * sigma**2 * peak_counts

    ideal = spot + pedestal
    noisy = rng.poisson(ideal * gain) / gain
    noisy = noisy + rng.normal(0.0, read_noise_e / gain, size=shape)

    if saturate:
        noisy = np.clip(noisy, 0, SATURATION_ADU)
    else:
        noisy = np.clip(noisy, 0, None)

    image = np.rint(noisy).astype(np.uint16)
    n_saturated = int(np.count_nonzero(image >= SATURATION_ADU))

    truth = {
        "x": x,
        "y": y,
        "fwhm": fwhm,
        "sigma": sigma,
        "peak_counts": peak_counts,
        "total_counts": total_counts,
        "pedestal": pedestal,
        "n_saturated": n_saturated,
        "seed": seed,
    }
    return image, truth


def main(argv=None):
    """Write a synthetic frame and print its truth values as JSON.

    Parameters
    ----------
    argv : list of str, optional
        Argument list to parse. Defaults to `sys.argv[1:]`.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        description="Generate a synthetic fiberhead frame with known truth values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output", type=Path, help="destination TIFF path")
    parser.add_argument("--x", type=float, default=520.4)
    parser.add_argument("--y", type=float, default=610.7)
    parser.add_argument("--fwhm", type=float, default=11.0)
    parser.add_argument("--peak-counts", type=float, default=600.0,
                        help="peak amplitude above the pedestal, in counts")
    parser.add_argument("--pedestal", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    image, truth = make_frame(
        x=args.x, y=args.y, fwhm=args.fwhm, peak_counts=args.peak_counts,
        pedestal=args.pedestal, seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(args.output), image)

    truth["path"] = str(args.output)
    print(json.dumps(truth, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
