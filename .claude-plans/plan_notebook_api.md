_Accepted: 2026-09-06 14:14 IDT_

# A notebook API for `aperture_photometry_single.py`

## Context

The module is currently a CLI-only script. Its one high-level entry point,
`measure_frame(image_path, args, show=False)` (`aperture_photometry_single.py:982`), cannot be
used from a notebook without fighting it:

- it takes a `pathlib.Path` and calls `load_image` itself, so **in-memory arrays cannot be
  measured at all**;
- it requires an `argparse.Namespace` carrying all 14 option attributes, not keyword arguments;
- it prints a six-line summary to stdout unconditionally;
- it returns a dict of **pre-formatted strings** (`"1.234560e+06"`, `x` to 3 decimals, `snr` to 2)
  keyed for `csv.DictWriter`, so `pd.DataFrame([...])` yields object-dtype columns and lossy values;
- batching and table assembly live in `main()` (`:1079`), wired to `gather_frames`/`update_csv`.

Everything *below* `measure_frame` is already array-native — `frame_stats`, `locate_source`,
`measure_fwhm`, `estimate_background_global` / `_annulus`, and `measure_counts` all take a 2D
`data` array. So the fix is a thin re-layering, not new photometry.

Goal: `measure_images(list_of_arrays_or_paths) -> pandas.DataFrame` with numeric columns, plus
opt-in inline diagnostic figures. The CLI keeps behaving exactly as it does today, driven by the
same code path.

## Design decisions (confirmed with the user)

- Batch return type: **pandas DataFrame** (pandas 3.0.5 installed).
- Accepted input: **a list of 2D arrays, or a list of paths / path-like strings**. No dicts.
- Extras: **inline figures, opt-in**. No CSV writing from the notebook API.

## Implementation

All changes are in `aperture_photometry_single.py`.

### 1. `measure_single_image(data, ...) -> dict` — the new core

Extract the pipeline body of `measure_frame` (`:1014`–`:1038`) into a public, array-native,
keyword-driven function. This becomes the single definition of the measurement; nothing else
duplicates it.

```python
def measure_single_image(data, name=None, fwhm_guess=DEFAULT_FWHM_GUESS, k_fwhm=DEFAULT_K_FWHM,
                  radius=None, nsigma=DEFAULT_NSIGMA, detect_method=DEFAULT_DETECT_METHOD,
                  npixels=DEFAULT_NPIXELS, deblend=True, annulus_bkg=False,
                  r_in=ANNULUS_IN_FWHM, r_out=ANNULUS_OUT_FWHM,
                  gain=GAIN_E_PER_ADU, read_noise_e=READ_NOISE_E,
                  plot=False, verbose=False):
```

Same call order as today: `frame_stats` → `locate_source` → `measure_fwhm` →
`radius or k_fwhm * fwhm` → background branch → `measure_counts`.

Returns a dict of **numbers**, not strings:

`name, x, y, fwhm_pix, radius_pix, detect_method, bkg_method, bkg_median, bkg_std, n_bkg_pix,
net_counts, counts_err, snr, n_saturated, area, sigma_pix, gain, read_noise_e`

Notes:
- `n_bkg_pix`, `area`, `sigma_pix` are new relative to `CSV_COLUMNS`; `update_csv` already uses
  `extrasaction="ignore"` (`:814`), so they are harmless if a row is ever routed to CSV.
- The `CircularAperture` object stays internal — it is not DataFrame material.
- `verbose=True` prints the same six-line summary and saturation warning `measure_frame` prints
  today; default `False` keeps batches quiet.
- `plot=True` adds a `"figure"` key holding the live `matplotlib.figure.Figure`.
- The `[load]` and `[stats]` notices from `load_image`/`frame_stats` are *not* suppressed by
  `verbose=False` — the digitization-floor message is a genuine data-quality warning and only
  fires on frames where the sigma clip lost the noise tail.

### 2. `measure_images(images, ...) -> pandas.DataFrame` — the batch wrapper

```python
def measure_images(images, plane=0, plot=False, verbose=False, on_error="raise", **params):
```

- `images` is a list whose elements are each either a 2D `numpy.ndarray` or a path
  (`str` / `os.PathLike`). A bare single array or single path is accepted as a one-element list.
  Paths go through the existing `load_image(path, plane=plane)`.
- `name` column: `str(path)` for paths, `f"image_{i}"` for arrays.
- `**params` forwards to `measure_single_image`, so every CLI knob is available by keyword.
- `on_error="raise"` (default) surfaces failures immediately — a silent gap in a notebook
  DataFrame is worse than a traceback. `on_error="skip"` mirrors `main()`'s per-frame
  `except Exception`, printing the failure and omitting the row, for long batches.
- With `plot=True`, figures are left open so the inline backend renders them; the `"figure"` key
  is popped before the DataFrame is built.
- **Import pandas lazily inside this function**, following the existing lazy-`pyplot` pattern in
  `make_figure` (`:754`) — the CLI must not pay pandas' import cost for a feature it never uses.

### 3. `make_figure`: allow no-save, no-close

`make_figure` (`:721`) currently always `savefig`s and always `plt.close(fig)` (`:804`–`:809`),
so it cannot hand back a live figure.

- `out_path=None` → skip the `mkdir`/`savefig`.
- New `close=True` parameter → when `False`, do not close the figure.
- Return `fig` instead of `out_path`. Safe: `measure_frame` calls it at `:1056` without using the
  return value, and already holds `figure_path` for its own print.

### 4. `measure_frame`: reduce to a thin CLI adapter

Rewrite as `load_image` → `measure_single_image(..., verbose=True, **vars-from-args)` → figure to
`PLOT_DIR` → format the CSV row. Add a small `_csv_row(result)` helper holding the existing
f-string formatting (`:1060`–`:1076`) so the string formatting lives in exactly one place and
`main()` is untouched.

### 5. Docs

Add a short "Use from a notebook" section to the module docstring and one line to `claude.md`,
showing the import shim the repo requires (no package, no `__init__.py`):

```python
import sys; sys.path.insert(0, "/home/paramre/Param/scripts/MAST_fiberhead_testing")
from aperture_photometry_single import measure_images, measure_single_image

df = measure_images(my_arrays)                       # -> DataFrame, numeric columns
df = measure_images(paths, annulus_bkg=True, k_fwhm=3.0)
res = measure_single_image(arr, plot=True, verbose=True)     # single image + inline figure
```

## Files

- `aperture_photometry_single.py` — all code changes above.
- `claude.md` — one line noting the module is importable as a library.
- `.claude-plans/plan_notebook_api.md` — this plan, saved with an accepted-timestamp header, per
  the `claude.md` rule.

## Verification

1. **Truth recovery from arrays.** `tests/make_test_frame.py:29` `make_frame(...)` returns
   `(image, truth)` in memory. In a scratchpad script, build three frames (clean, saturated,
   different FWHM), call `measure_images([a0, a1, a2])`, and confirm the DataFrame has 3 rows,
   float dtypes on `x`/`y`/`fwhm_pix`/`net_counts`/`counts_err`/`snr`, recovered counts within 2σ
   of the injected truth, and `n_saturated` non-zero only for the saturated frame.
2. **Parity with the CLI.** Run `python aperture_photometry_single.py data/24aug26_MAST/run0_0_2s.tif
   --no-plot --no-csv` and `measure_images(["data/24aug26_MAST/run0_0_2s.tif"])`; the centroid,
   FWHM, radius, background and counts must match to full precision.
3. **CLI regression.** Run the CLI over `data/24aug26_MAST/` before and after the change and diff
   the resulting CSV and stdout — both must be identical, and the PNGs must still land in `plots/`.
4. **Figures.** `measure_single_image(arr, plot=True)` returns an unclosed `Figure` with two axes and
   writes nothing to disk; `measure_frame` via the CLI still writes
   `plots/<stem>_aperture.png`.
5. **No pandas at import time.** `python -X importtime -c "import aperture_photometry_single"`
   must not show pandas.
6. **Error policy.** `measure_images([good, np.zeros((100, 100))], on_error="skip")` returns one
   row and prints the failure; the same call with the default raises.
