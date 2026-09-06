_Accepted: 2026-08-16 19:35 IDT_

# Single-source aperture photometry for MAST fiberhead images

## Context

The repo has no analysis code yet — only [claude.md](claude.md) and empty `data/`/`results/`.
This is the first script of the planned toolkit: measure the total counts from a **single**
fiber output in a 1440×1080 TIFF frame, where the source occupies only a few tens of pixels.

Because the source is a tiny fraction of the frame, the script has to (1) find it, (2) estimate
the background *locally* rather than globally, and (3) report counts with a defensible 1σ error
bar. Decisions confirmed with the user:

- Source is a **compact / defocused spot** (roughly Gaussian), not a sharp-edged fiber face.
- Detector constants hard-coded: **gain = 1.0 e⁻/ADU, read noise = 0.46 e⁻ RMS** (CLI-overridable).
- Aperture radius from the **measured FWHM**, `r = k · FWHM`, default `k = 5`.
- Background is **global** (sigma-clipped median of the frame) by default; a local annulus is opt-in
  via a flag, with radii 9 × FWHM and 14 × FWHM.
- Saturation is any pixel at exactly **1023 counts** (10-bit sensor).
- Outputs: annotated cutout PNG in `results/`, interactive window, curve-of-growth panel, CSV row.
- **TIFF only** for now; FITS support comes later when FITS frames exist to test against.

Environment already has everything needed: photutils 2.3.0, astropy 7.2.0, numpy 2.4.1,
scipy 1.17.0, matplotlib 3.10.8, tifffile 2025.3.30.

## Deliverable

A single self-contained CLI script, `aperture_photometry_single.py`, at the repo root, written as
small independent functions so the generic pieces (image loading, source location, background
estimation, output formatting) can be lifted into a shared helpers module when the second script
(multi-source / PSF) arrives. Per [claude.md](claude.md): full docstrings (purpose, parameters,
returns) on every function, minimal inline commentary.

```bash
python aperture_photometry_single.py data/run01/frame_001.tif
```

## Implementation

Module constants at the top, so the detector assumptions are visible in one place:
`GAIN_E_PER_ADU = 1.0`, `READ_NOISE_E = 0.46`, `SATURATION_ADU = 1023`, `DEFAULT_K_FWHM = 5.0`,
`DEFAULT_FWHM_GUESS = 11.0`, `ANNULUS_IN_FWHM = 9.0`, `ANNULUS_OUT_FWHM = 14.0`.

**1. `load_image(path, plane=0) -> np.ndarray`**
`tifffile.imread`, cast to `float64`. If the array is 3D, select `plane` along the leading axis
(multi-page) or average the last axis when it looks like RGB(A) — either way return a 2D array and
say in the log which branch was taken.

**2. `locate_source(data, fwhm_guess, nsigma) -> (x, y)`**
- `astropy.stats.sigma_clipped_stats` for a global median/std (detection threshold only).
- `photutils.detection.DAOStarFinder(fwhm=fwhm_guess, threshold=nsigma*std)` on the
  median-subtracted frame; take the highest-flux detection.
- Fallback if nothing is found: `photutils.detection.find_peaks` on a Gaussian-smoothed copy,
  brightest peak.
- Refine with `photutils.centroids.centroid_quadratic` on a small box around the candidate.
- Raise a clear error naming the frame if both paths fail.

**3. `measure_fwhm(data, position, bkg) -> float`**
`photutils.profiles.RadialProfile` centred on the refined position → its `gaussian_fwhm`
attribute. Falls back to the `fwhm_guess` with a warning if the Gaussian fit does not converge.

**4. Background — two estimators behind one interface, each returning `(median, std, n_pix)`**
- `estimate_background_global(data)` — **default**. `sigma_clipped_stats(data, sigma=3.0)` over the
  whole frame. With the source only a few tens of pixels across, it contributes negligibly to the
  clipped statistics, and `n_pix` is ~10⁶ so the error on the background *level* is essentially zero.
- `estimate_background_annulus(data, position, r_in, r_out)` — opt-in via `--annulus-bkg`.
  `CircularAnnulus` + `ApertureStats` with `SigmaClip(sigma=3.0)`, giving the sigma-clipped median,
  per-pixel std and clipped pixel count. Sigma clipping matters here: a neighbouring reflection or
  hot pixel in the annulus would otherwise bias the subtraction.
  At the requested radii (9 × FWHM and 14 × FWHM ≈ 99 and 154 px for an 11 px FWHM) the annulus is
  large enough to run off the frame for a source near an edge — `ApertureStats` handles the partial
  overlap, but the script warns if more than a few percent of the annulus falls outside the image.

**5. `measure_counts(...) -> dict`**
- `CircularAperture(position, r = k·fwhm)`, `aperture_photometry(..., method='exact')`.
- Net counts = raw sum − `bkg_median · aperture_area`.
- Per-pixel background error floored at the read-noise equivalent:
  `sigma_pix = max(bkg_std, READ_NOISE_E / GAIN)`.
- Total error via `photutils.utils.calc_total_error(data - bkg_median, sigma_pix, GAIN)` fed back
  into `aperture_photometry(error=...)`, then the background-subtraction term added in quadrature:
  `sigma_total² = aperture_sum_err² + (A² · sigma_pix²) / n_pix_annulus`.
  This is the standard CCD equation; read noise and background Poisson noise are already inside the
  empirical `bkg_std`, so they are not added a second time.
- Count pixels inside the aperture at exactly `SATURATION_ADU` (1023) using the aperture mask.
  The count goes into the CSV as `n_saturated`. **Only if it is non-zero**, print to stdout:
  `WARNING: N saturated pixels inside the aperture — measured counts are a lower limit.`

**6. Outputs**
- **Figure, two panels.** Left: cutout via `astropy.nddata.Cutout2D` sized ~`3 × r_aper`, ZScale
  stretch, with `aperture.plot()` overlaid (plus `annulus.plot()` when `--annulus-bkg` is used) and
  the centroid marked. Right: `photutils.profiles.CurveOfGrowth` over radii `1 … 1.5 · r_aper`, with
  a vertical line at the chosen radius — this is the check that the aperture encloses the spot.
- Saved to `results/<stem>_aperture.png`; `plt.show()` unless `--no-plot`.
- **CSV**: append a row to `results/aperture_photometry_single.csv` (header written if absent):
  `filename, x, y, fwhm_pix, radius_pix, bkg_method, bkg_median, bkg_std, net_counts,
  counts_err, snr, n_saturated, gain, read_noise_e`.
- **stdout**: the headline numbers, formatted as `counts = 1.234e+06 ± 2.1e+03 (SNR 588)`, plus
  centroid, FWHM, radius and background level.

**CLI** (`argparse`): `image` (positional), `--fwhm-guess` (default 11 px), `--k-fwhm` (default 5),
`--radius` (explicit override in px, skips the FWHM scaling), `--nsigma` (default 5),
`--annulus-bkg` (switch to the local annulus estimator), `--r-in` / `--r-out` (in units of FWHM,
defaults 9 and 14), `--gain`, `--read-noise`, `--plane`, `--no-plot`, `--no-csv`,
`--outdir` (default `results/`).

## Verification

No real frames exist yet, so correctness is checked against synthetic data with known truth:

1. Write `tests/make_test_frame.py` — generates a 1440×1080 frame with a Gaussian spot of known
   total counts and FWHM (~11 px) at a known off-centre position, plus a constant pedestal and
   Poisson + read noise, clipped at 1023 counts to mimic the 10-bit sensor; writes a TIFF to the
   scratchpad (not into `data/`). Options for an unsaturated frame and a deliberately saturated one.
2. Run the script on the unsaturated frame and confirm: recovered centroid within ~0.2 px, recovered
   FWHM within ~10%, and recovered counts within 2σ of the injected truth (at `k = 5` the aperture
   encloses essentially all of a Gaussian, so the encircled-fraction correction is negligible).
3. Run on the saturated frame and confirm `n_saturated` matches the number of clipped pixels and the
   all-caps WARNING appears — and that no warning is printed for the clean frame.
4. Run both background modes on the same frame; the global and annulus medians should agree to well
   within their errors on synthetic data with a flat pedestal.
5. Eyeball the saved PNG: aperture centred on the spot, growth curve flat at the chosen radius.
6. Re-run twice to confirm the CSV appends rather than overwrites, and that `--no-plot` works
   headless.

Caveat to flag on delivery: all of this validates the maths and the plumbing, not the real
detector. The FWHM guess default and annulus radii will likely need one tweak once a genuine
fiberhead frame lands in `data/`.

## Commits

1. `aperture_photometry_single.py` — the script.
2. `tests/make_test_frame.py` — synthetic frame generator used for verification.
