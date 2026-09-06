_Accepted: 2026-08-25 IDT_

# Robust source detection: segmentation, a noise floor, and photutils 3.0 fixes

## Context

Running the script on the two 2 s frames in `data/24aug26_MAST/` exposed three separate
problems. Investigating them showed the "hot pixels detected as sources" symptom and the
`run0_3_2s.tif` crash have *different* causes, and that segmentation detection alone would
not have fixed either.

### 1. The spurious detections are not hot pixels — the threshold has collapsed

DAOStarFinder reports 52 sources in `run0_0_2s.tif` and 25 in `run0_3_2s.tif`. Inspecting
the detection table, every spurious entry has `peak = 1.0`, i.e. one ADU above the
background. They are clumps of 2-ADU noise pixels, not hot pixels.

The cause is `frame_stats`. These frames sit at 1–2 ADU on a 10-bit sensor, so the pixel
distribution is ~93 % at value 1 and ~7 % at value 2. `SigmaClip(sigma=3)` throws away the
value-2 tail — the very noise it is meant to measure — and converges to

| frame | fraction at 2 ADU | clipped std | 5σ threshold |
|---|---|---|---|
| `run0_0_2s` | 15.6 % | 0.360 | 1.80 (works) |
| `run0_3_2s` | 6.7 % | **0.000** | **0.00** (detects everything) |

So the threshold is data-dependent and degenerates to zero. Note this is already visible in
`results/aperture_photometry_single.csv`, which has rows with `bkg_std = 0.0000`.

Segmentation detection with a zero threshold is *worse*, not better: it returns one segment
of 1 555 200 px covering the entire frame.

### 2. `run0_3_2s.tif` errors out in the plot, not the photometry

`ValueError: minvalue must be less than or equal to maxvalue`. The cutout is overwhelmingly
a single value (1.0), so `ZScaleInterval().get_limits()` returns
`vmin = 1.0000000000000024`, `vmax = 0.9999999999999977` — inverted by floating-point noise —
and `imshow` rejects it. The photometry itself completes; `--no-plot` runs fine.

### 3. photutils 3.0 renamed two keywords

`centroid_quadratic(xpeak=, ypeak=)` and `find_peaks(npeaks=)` are both deprecated in 3.0,
removed in 4.0. (`xcentroid` → `x_centroid` was already fixed in the working tree.)

## Plan

### Floor the noise estimate at the digitization scale

In `frame_stats`, floor the returned scatter at `DIGITIZATION_SIGMA = 1/sqrt(12) ≈ 0.289`
ADU — the smallest scatter a digitized signal can meaningfully have. Print a note when the
floor engages. This is the root-cause fix and it is what makes any detection method work.

Measured effect on the threshold and the true scatter (std below the 99.9th percentile):

| frame | clipped | floored | true |
|---|---|---|---|
| `run0_0_2s` | 0.360 | 0.360 | 0.417 |
| `run0_3_2s` | 0.000 | **0.289** | 0.320 |

Blast radius is small: `sigma_pix = max(bkg_std, read_noise/gain)` is already floored at
0.46, so this changes the detection threshold and the reported `bkg_std`, not the errors.

### Segmentation detection as the default

Add `detect_source_segment`: convolve the median-subtracted frame with a Gaussian kernel of
the guessed FWHM, `detect_sources` above the floored threshold with a minimum connected-pixel
count, `deblend_sources`, then pick the brightest segment by `segment_flux` from a
`SourceCatalog`. Keep DAOStarFinder behind `--detect-method dao` so the two can be compared
on real data.

`n_pixels` is the part that answers the original question: a fiber output is an extended blob,
so requiring ≥25 connected pixels rejects single-pixel spikes structurally, in a way
DAOStarFinder's sharpness cuts only do incidentally.

Deblending is required, not optional. On `run0_yy_1s` — a frame with 19 % of pixels above the
background from diffuse scattered light — detection without deblending merges the fiber into a
332 766 px glow blob and picks its centroid at (784, 982) instead of the fiber at (811, 439).
Deblending recovers (811.0, 439.0). It costs ~11 s on that frame and ~0 s on the clean ones, so
it is on by default with `--no-deblend` to skip it.

Detections after both fixes, brightest segment in bold:

| frame | DAO before | segments after | brightest |
|---|---|---|---|
| `run0_0_2s` | 52 | 2 | **(810.8, 438.9)** fiber |
| `run0_3_2s` | 25 | 3 | **(810.8, 438.9)** fiber |
| `run0_yy_1s` | 24 | 1282 | **(811.0, 439.0)** fiber |

The real bright defect in these frames — a compact blob at (943, 162) with peak 778/793 ADU
and area ~180 px — is correctly ranked second, below the fiber.

### Remaining fixes

- `_display_limits` helper in `make_figure`: fall back from ZScale to the finite min/max, then
  to a ±0.5 pad, whenever the interval is non-finite or inverted.
- `centroid_sources(..., centroid_func=centroid_quadratic)` replaces the deprecated
  `xpeak`/`ypeak`; `find_peaks(n_peaks=1)` replaces `npeaks`.
- New CLI: `--detect-method {segment,dao}`, `--npixels`, `--no-deblend`.
- New CSV column `detect_method`. `update_csv` already passes `restval=""`, so older tables
  gain an empty column rather than breaking.

## Out of scope

The `run0_yy_1s` / `run0_zz_1s` frames have a diffuse glow across a fifth of the detector and
20 pixels at 1021–1022 ADU. A single sigma-clipped scalar background is the wrong model there
and `bkg_median = 1.0` badly underestimates it. `Background2D` is the right tool, and
saturation is flagged but not handled. Both are separate pieces of work.
