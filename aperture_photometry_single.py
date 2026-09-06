#!/usr/bin/env python3
"""Aperture photometry for a single fiber output in MAST fiberhead images.

Locates the one bright spot in a TIFF or FITS frame, subtracts a background
estimated either globally or from a local annulus, and integrates the counts
inside a circular aperture of fixed radius.

The radius is fixed (--radius) rather than scaled to the source width. The camera
and optics are the same for every frame, so the fraction of the light enclosed at
a given radius is the same too, and a fixed aperture is what makes the fluxes
directly comparable. A measured width, by contrast, drifts with source brightness
- the profile's wings clear the noise floor on a bright frame and sink into it on
a faint one - so scaling the radius by one would fold that drift into the flux and
make it look like a throughput trend. --k-fwhm restores the scaled behaviour for
comparison.

The spot is found by segmentation: the brightest group of connected pixels above
the detection threshold that covers at least --npixels pixels, discarding any
group whose counts sit almost entirely in one pixel. Requiring an area is what
keeps small noise clumps from being taken for the fiber on a faint frame, and the
concentration cut is what rejects hot pixels and cosmic rays, which the area cut
cannot because detection runs on a convolved frame. --detect-method dao selects
PSF-correlation detection instead.

Outputs a row in a CSV table of measurements (see --outfile), a summary printed
to stdout, and a two-panel diagnostic figure (annotated cutout + curve of growth)
written to `plots/`. The figures are throwaway sanity checks, so their location
is fixed and untracked; --no-plot skips them entirely.

Printing is controlled by two independent switches. --verbose adds the running
notes the measurement makes about itself - how a stack was reduced, how many
detections were found, which were discarded as spikes - and is off by default.
--quiet drops the summary and the batch progress lines. Neither silences a
warning about a degraded or unreliable result, nor an error. The library
functions take the same two as `verbose` and `print_output`, both defaulting to
False, so a notebook call is silent unless it has something to warn about.

Any number of frames can be measured in one invocation, given as files, as
directories (whose top-level images are measured), or as a text file listing
either of those one per line. Batching this way is worth it: the astropy and
photutils imports cost ~0.6 s and are paid once per invocation rather than once
per frame.

Examples
--------
    python aperture_photometry_single.py data/run01/frame_001.tif
    python aperture_photometry_single.py data/run01/frame_001.fits
    python aperture_photometry_single.py data/run01/
    python aperture_photometry_single.py frames_to_measure.txt

Use from a notebook
-------------------
The module is importable as a library, and the measurement is array-native, so
frames already in memory can be measured without going through disk or the CLI.
There is no package, so the repo root has to be on `sys.path`:

    import sys; sys.path.insert(0, "/path/to/MAST_fiberhead_testing")
    from aperture_photometry_single import measure_images, measure_single_image

    df = measure_images(my_arrays)              # list of 2D arrays
    df = measure_images(paths, radius=30.0)     # or a list of paths
    res = measure_single_image(arr, plot=True)  # one frame, with a figure

`measure_images` returns a pandas DataFrame of numbers, not the CSV's formatted
strings, and writes nothing to disk. Every CLI option is available to both as a
keyword argument.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)
from photutils.detection import DAOStarFinder, find_peaks
from photutils.profiles import CurveOfGrowth
from photutils.segmentation import (
    SourceCatalog,
    deblend_sources,
    detect_sources,
    make_2dgaussian_kernel,
)
from photutils.utils import calc_total_error
from scipy import ndimage

# Detector properties of the fiberhead test camera.
GAIN_E_PER_ADU = 1.0
READ_NOISE_E = 0.46
# Digital saturation level. The 10-bit sensor's nominal maximum code is 1023,
# but its output rails one count below that: saturated frames in
# data/real_data_sample have flat tops of 100+ pixels at exactly 1022 and no
# pixel anywhere at 1023. Counting saturation from 1023 therefore reports none
# at all on a frame whose core is fully clipped.
SATURATION_ADU = 1022  # 10-bit sensor, observed rail

# Measurement defaults.
DEFAULT_FWHM_GUESS = 11.0  # pixels
# Fixed extraction radius, in pixels. Set from the curve of growth of a stack of
# real frames, which is flat from well inside this radius out past it: far enough
# to enclose the wings, near enough that the background noise the aperture admits
# (going as its area) stays small. Being on the flat part also makes the flux
# insensitive to where the aperture sits, which is what lets the centroid be
# measured once and reused across a run.
DEFAULT_RADIUS = 36.0
ANNULUS_IN_FWHM = 4.0
ANNULUS_OUT_FWHM = 7.0
DEFAULT_NSIGMA = 5.0
SIGMA_CLIP = 3.0
DEFAULT_NPIXELS = 25  # smallest connected area a segmentation detection may have

# Smallest per-pixel scatter a digitized signal can meaningfully have: the
# standard deviation of the rounding error of a quantizer with a 1-count step.
# Faint frames on this 10-bit sensor sit at 1-2 counts, where sigma clipping can
# discard the whole noise tail and return a scatter of exactly zero; flooring it
# here keeps the detection threshold from collapsing to the background level.
DIGITIZATION_SIGMA = 1.0 / np.sqrt(12.0)

# Largest share of a detection's counts that may sit in its single brightest
# pixel. Detection runs on a convolved frame, where one hot pixel is smeared into
# a blob hundreds of pixels wide, so the --npixels area cut cannot reject it; this
# does, by asking how concentrated the detection is in the unsmoothed data. A
# Gaussian source of FWHM f puts a fraction of about 0.88 / f^2 of its counts in
# the peak pixel - 0.007 for the ~11 px fiber output - while an isolated spike
# puts all of them there. Set tight enough to also reject a source as narrow as
# 3 px FWHM (fraction ~0.1), so a source has to be several pixels across to pass.
SPIKE_PEAK_FRACTION = 0.1

DETECT_METHODS = ("segment", "dao")
DEFAULT_DETECT_METHOD = "segment"

TIFF_SUFFIXES = {".tif", ".tiff"}
FITS_SUFFIXES = {".fits", ".fit", ".fts"}
IMAGE_SUFFIXES = TIFF_SUFFIXES | FITS_SUFFIXES

# Output locations. Diagnostic figures are throwaway sanity checks, so they
# always land in PLOT_DIR rather than anywhere the user chooses.
RESULTS_DIR = Path("results")
PLOT_DIR = Path("plots")
DEFAULT_OUTFILE = RESULTS_DIR / "aperture_photometry_single.csv"
CSV_COLUMNS = [
    "filename",
    "x",
    "y",
    "fwhm_pix",
    "radius_pix",
    "detect_method",
    "bkg_method",
    "bkg_level",
    "bkg_std",
    "net_counts",
    "counts_err",
    "snr",
    "n_saturated",
    "gain",
    "read_noise_e",
]

# Columns of the DataFrame returned by measure_images. A superset of
# CSV_COLUMNS: the notebook table carries the background pixel count, the
# aperture area and the per-pixel sigma as well, which the CSV has no room for.
RESULT_COLUMNS = [
    "name",
    "x",
    "y",
    "fwhm_pix",
    "radius_pix",
    "detect_method",
    "bkg_method",
    "bkg_level",
    "bkg_std",
    "n_bkg_pix",
    "net_counts",
    "counts_err",
    "snr",
    "n_saturated",
    "area",
    "sigma_pix",
    "gain",
    "read_noise_e",
]


def expand_input(entry, allow_list=True):
    """Expand one command-line entry into the frames it refers to.

    An entry may be an image file (TIFF or FITS), a directory (whose top-level
    images are taken, sorted by name, without recursing), or a text file listing
    files and directories one per line. Blank lines and lines starting with `#` are ignored
    in a list file, and relative paths in it are resolved against the directory
    the list itself lives in.

    Parameters
    ----------
    entry : str or pathlib.Path
        The path to expand.
    allow_list : bool, optional
        Whether a non-image file may be read as a list of paths. Set False when
        expanding the contents of a list file, so lists cannot nest.

    Returns
    -------
    list of pathlib.Path
        The frames to measure, in the order they were given.

    Raises
    ------
    FileNotFoundError
        If the entry does not exist.
    ValueError
        If a directory holds no images, or a list file is referenced from inside
        another list file.
    """
    path = Path(entry)

    if not path.exists():
        msg = f"no such file or directory: {path}"
        raise FileNotFoundError(msg)

    if path.is_dir():
        frames = sorted(p for p in path.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not frames:
            msg = (f"directory {path} contains no image files (looked for "
                   f"{', '.join(sorted(IMAGE_SUFFIXES))} at the top level only)")
            raise ValueError(msg)
        return frames

    if path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]

    if not allow_list:
        msg = (f"{path} is not a TIFF, a FITS file or a directory, and list "
               f"files cannot nest")
        raise ValueError(msg)

    frames = []
    with path.open() as handle:
        for lineno, line in enumerate(handle, start=1):
            entry = line.split("#", 1)[0].strip()
            if not entry:
                continue
            listed = Path(entry)
            if not listed.is_absolute():
                listed = path.parent / listed
            try:
                frames.extend(expand_input(listed, allow_list=False))
            except (FileNotFoundError, ValueError) as exc:
                msg = f"{path}:{lineno}: {exc}"
                raise ValueError(msg) from None

    if not frames:
        msg = f"list file {path} names no frames"
        raise ValueError(msg)

    return frames


def gather_frames(entries):
    """Expand all command-line entries into a de-duplicated list of frames.

    Parameters
    ----------
    entries : list of str
        The positional arguments as given on the command line.

    Returns
    -------
    list of pathlib.Path
        Frames to measure, in the order first seen, with repeats removed.
    """
    frames = []
    seen = set()

    for entry in entries:
        for frame in expand_input(entry):
            try:
                key = frame.resolve()
            except OSError:
                key = frame
            if key not in seen:
                seen.add(key)
                frames.append(frame)

    return frames


def load_image(path, plane=0, verbose=False):
    """Read a TIFF or FITS frame as a 2D float array.

    The format is chosen from the file suffix: `.fits`, `.fit` and `.fts` are
    read with astropy from the primary HDU, which is where the fiberhead camera
    writes its image, and everything else is read with tifffile. Multi-page or
    multi-plane stacks are reduced by selecting a single plane; RGB(A) images are
    reduced by averaging the colour channels.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the image file.
    plane : int, optional
        Plane to select from a stack. Ignored for 2D and RGB(A) input.
    verbose : bool, optional
        If True, report how a stack or an RGB(A) image was reduced to 2D.

    Returns
    -------
    numpy.ndarray
        Two-dimensional image in float64, in the original count units.

    Raises
    ------
    ValueError
        If the file does not reduce to a 2D image, or `plane` is out of range.
    """
    path = Path(path)

    if path.suffix.lower() in FITS_SUFFIXES:
        raw = np.asarray(fits.getdata(str(path), 0))
        kind = "FITS"
    else:
        raw = tifffile.imread(str(path))
        kind = "TIFF"

    if raw.ndim == 2:
        image = raw
    elif raw.ndim == 3 and kind == "TIFF" and raw.shape[-1] in (3, 4):
        image = raw[..., :3].mean(axis=-1)
        if verbose:
            print(f"[load] RGB(A) TIFF {raw.shape} -> averaged colour channels")
    elif raw.ndim == 3:
        if not 0 <= plane < raw.shape[0]:
            msg = f"plane {plane} out of range for a stack of {raw.shape[0]} planes"
            raise ValueError(msg)
        image = raw[plane]
        if verbose:
            print(f"[load] multi-plane {kind} {raw.shape} -> using plane {plane}")
    else:
        msg = f"cannot reduce a {kind} image of shape {raw.shape} to a 2D image"
        raise ValueError(msg)

    return np.asarray(image, dtype=np.float64)


def frame_stats(data, verbose=False):
    """Return sigma-clipped statistics for the whole frame.

    Computed once per frame and reused for the detection threshold, the profile
    background and (unless a local annulus is requested) the background
    subtraction itself, since clipping a 1440x1080 frame is not free.

    The level is the clipped *mean*, not the median. The background of a 10-bit
    frame occupies only a couple of adjacent ADU codes, and the median of so
    coarsely quantized a distribution snaps to whichever code holds the majority
    - it can only ever return an integer. On the sample frames that rounds 2.73
    counts up to exactly 3.00, and over-subtracting 0.27 counts from every pixel
    drains the curve of growth by several percent at the extraction radius, an
    error that grows as the aperture's area. The mean of the same clipped pixels
    carries the fractional part and does not.

    The scatter is floored at `DIGITIZATION_SIGMA`. On a faint frame sitting at
    one or two counts, the pixel distribution is a spike at the background with a
    sparse tail one count above it, and sigma clipping discards that tail - the
    only noise there is - leaving a scatter of exactly zero. Anything scaled by
    the scatter then degenerates: a 5-sigma detection threshold becomes a 0-sigma
    one that flags every pixel above the background. The floor is the standard
    deviation of a 1-count quantizer's rounding error, so it says no more than
    that a digitized frame cannot be quieter than its own digitization.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image.
    verbose : bool, optional
        If True, report when the scatter is raised to the digitization floor.

    Returns
    -------
    tuple
        (level, std, n_pixels) of the sigma-clipped frame, where `level` is the
        clipped mean, `std` is not less than `DIGITIZATION_SIGMA` and `n_pixels`
        is the number of pixels surviving the clip.
    """
    clipped = SigmaClip(sigma=SIGMA_CLIP)(data.ravel(), masked=True)
    level, std = float(np.ma.mean(clipped)), float(np.ma.std(clipped))

    if std < DIGITIZATION_SIGMA:
        # Only worth reporting when the clip has actually lost the noise tail. A
        # frame whose scatter merely rounds down to just under the floor is not
        # telling the reader anything.
        if verbose and std < 0.9 * DIGITIZATION_SIGMA:
            print(f"[stats] sigma-clipped scatter is {std:.3f} counts/px, below the "
                  f"digitization floor; using {DIGITIZATION_SIGMA:.3f} instead")
        std = DIGITIZATION_SIGMA

    return level, std, int(clipped.count())


def detect_segment(data, level, std, fwhm_guess=DEFAULT_FWHM_GUESS,
                   nsigma=DEFAULT_NSIGMA, npixels=DEFAULT_NPIXELS, deblend=True,
                   reject_spikes=True, verbose=False):
    """Find the brightest source by image segmentation.

    The frame is background-subtracted and convolved with a Gaussian kernel
    matched to the expected source size, then pixels above `nsigma` times the
    background scatter are grouped into connected segments of at least `npixels`
    pixels. The brightest segment by total flux is taken as the source.

    Requiring a minimum connected area is what makes this robust on faint frames:
    a fiber output is an extended blob, so noise clumps of a few pixels are
    rejected structurally rather than incidentally, as they are by DAOStarFinder's
    PSF-correlation and sharpness cuts. The area cut does not reject a single hot
    pixel, though. Segmentation runs on the convolved frame, and convolution
    smears one bright spike into a blob as wide as the kernel: an 877-count hot
    pixel in data/real_data_sample survives an npixels of 25 as a ~100-pixel
    segment. Such detections are removed instead by `SPIKE_PEAK_FRACTION`, which
    asks how concentrated each detection is in the unsmoothed data.

    Deblending splits segments that merge several peaks. It matters whenever the
    frame carries diffuse scattered light: without it, a bright fiber sitting
    inside a large low-level glow is absorbed into the glow's segment, and the
    reported position is the centroid of the glow rather than of the fiber.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image.
    level : float
        Background level to subtract before detection.
    std : float
        Per-pixel background scatter that `nsigma` is measured in.
    fwhm_guess : float, optional
        Approximate source FWHM in pixels, used to size the convolution kernel.
    nsigma : float, optional
        Detection threshold in units of `std`.
    npixels : int, optional
        Smallest number of connected pixels a detection may have.
    deblend : bool, optional
        Whether to deblend merged segments. Costs seconds on a frame with
        hundreds of segments and nothing on a clean one.
    reject_spikes : bool, optional
        Whether to discard detections whose counts are concentrated in a single
        pixel, i.e. hot pixels and cosmic rays. See `SPIKE_PEAK_FRACTION`.
    verbose : bool, optional
        If True, report which detections were discarded as spikes.

    Returns
    -------
    tuple or None
        ((x, y), n_segments) for the brightest segment, where `n_segments` counts
        only the segments kept, or None if nothing was detected.
    """
    kernel_size = 2 * int(np.ceil(fwhm_guess)) + 1
    convolved = convolve(data - level, make_2dgaussian_kernel(fwhm_guess, kernel_size))

    segments = detect_sources(convolved, nsigma * std, n_pixels=npixels)
    if segments is None:
        return None

    if deblend:
        segments = deblend_sources(convolved, segments, npixels, progress_bar=False)

    catalog = SourceCatalog(data - level, segments, convolved_data=convolved)
    flux = np.asarray(catalog.segment_flux, dtype=float)
    peak = np.asarray(catalog.max_value, dtype=float)
    keep = np.isfinite(flux)

    if reject_spikes:
        with np.errstate(divide="ignore", invalid="ignore"):
            peak_fraction = np.where(flux > 0, peak / flux, np.inf)
        spikes = keep & (peak_fraction > SPIKE_PEAK_FRACTION)
        if spikes.any():
            keep = keep & ~spikes
            if verbose:
                where = ", ".join(f"({float(catalog.x_centroid[i]):.0f}, "
                                  f"{float(catalog.y_centroid[i]):.0f})"
                                  for i in np.flatnonzero(spikes))
                print(f"[detect] ignored {int(spikes.sum())} single-pixel spike(s) "
                      f"at {where}; a hot pixel or cosmic ray survives the --npixels "
                      "cut because detection runs on the convolved frame")

    if not keep.any():
        return None

    brightest = int(np.flatnonzero(keep)[np.argmax(flux[keep])])
    position = (float(catalog.x_centroid[brightest]),
                float(catalog.y_centroid[brightest]))
    if not np.isfinite(position).all():
        return None

    return position, int(keep.sum())


def detect_dao(data, level, std, fwhm_guess=DEFAULT_FWHM_GUESS,
               nsigma=DEFAULT_NSIGMA):
    """Find the brightest source with DAOStarFinder.

    Correlates the background-subtracted frame against a Gaussian PSF of the
    guessed FWHM and takes the highest-flux detection. Better suited to
    star-like sources than to the extended fiber outputs this script measures,
    and kept mainly so the two detection methods can be compared on real frames.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image.
    level : float
        Background level to subtract before detection.
    std : float
        Per-pixel background scatter that `nsigma` is measured in.
    fwhm_guess : float, optional
        Approximate source FWHM in pixels, used to size the detection kernel.
    nsigma : float, optional
        Detection threshold in units of `std`.

    Returns
    -------
    tuple or None
        ((x, y), n_sources) for the brightest detection, or None if nothing was
        found.
    """
    sources = DAOStarFinder(threshold=nsigma * std, fwhm=fwhm_guess)(data - level)
    if sources is None or len(sources) == 0:
        return None

    brightest = sources[np.argmax(sources["flux"])]
    position = (float(brightest["x_centroid"]), float(brightest["y_centroid"]))

    return position, len(sources)


def detect_peak(data, level, std, fwhm_guess=DEFAULT_FWHM_GUESS,
                nsigma=DEFAULT_NSIGMA):
    """Find the brightest peak of a smoothed frame.

    Last-resort fallback for when the chosen detection method finds nothing: it
    imposes no shape or area requirement, so it returns a position as long as
    anything at all rises above the threshold.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image.
    level : float
        Background level to subtract before the peak search.
    std : float
        Per-pixel background scatter that `nsigma` is measured in.
    fwhm_guess : float, optional
        Approximate source FWHM in pixels, used to size the smoothing kernel.
    nsigma : float, optional
        Detection threshold in units of `std`.

    Returns
    -------
    tuple or None
        The (x, y) position of the brightest peak, or None if nothing rose above
        the threshold.
    """
    kernel = Gaussian2DKernel(x_stddev=fwhm_guess / 2.355)
    smoothed = convolve(data - level, kernel)

    peaks = find_peaks(smoothed, threshold=nsigma * std, n_peaks=1)
    if peaks is None or len(peaks) == 0:
        return None

    return float(peaks["x_peak"][0]), float(peaks["y_peak"][0])


def locate_source(data, fwhm_guess=DEFAULT_FWHM_GUESS, nsigma=DEFAULT_NSIGMA,
                  stats=None, method=DEFAULT_DETECT_METHOD,
                  npixels=DEFAULT_NPIXELS, deblend=True, reject_spikes=True,
                  verbose=False):
    """Find the brightest source in the frame and return its position.

    Detection uses image segmentation or DAOStarFinder as selected by `method`,
    falling back to the brightest peak of a smoothed copy if that finds nothing.
    The detected position is returned as it stands.

    There is deliberately no sub-pixel refinement on top of it. The aperture
    radius is several times the source FWHM, so the enclosed counts are
    insensitive to where the aperture sits to well within a pixel, and the
    detection position - a flux-weighted segment centroid, for the default
    method - is already good to about 0.2 px. A quadratic centroid fit used to
    run here and bought nothing, while failing outright on a saturated core,
    whose flat top gives the fit no single peak to sit on.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image.
    fwhm_guess : float, optional
        Approximate source FWHM in pixels, used to size the detection kernel.
    nsigma : float, optional
        Detection threshold in units of the background standard deviation.
    stats : tuple, optional
        Precomputed `frame_stats` output, to avoid clipping the frame twice.
    method : {'segment', 'dao'}, optional
        Detection method to use.
    npixels : int, optional
        Smallest connected area a segmentation detection may have. Ignored by
        the `dao` method.
    deblend : bool, optional
        Whether to deblend merged segments. Ignored by the `dao` method.
    reject_spikes : bool, optional
        Whether to discard hot pixels and cosmic rays. Ignored by the `dao`
        method.
    verbose : bool, optional
        If True, report how many detections were found. A message about a
        degraded result - the detection fallback - is printed either way.

    Returns
    -------
    tuple of float
        The (x, y) pixel position of the source, from detection alone.

    Raises
    ------
    ValueError
        If `method` is not one of `DETECT_METHODS`.
    RuntimeError
        If neither the chosen method nor the peak-search fallback finds a source.
    """
    if method not in DETECT_METHODS:
        msg = f"unknown detection method {method!r}; expected one of {DETECT_METHODS}"
        raise ValueError(msg)

    level, std, _ = stats if stats is not None else frame_stats(data)

    if method == "segment":
        found = detect_segment(data, level, std, fwhm_guess=fwhm_guess,
                               nsigma=nsigma, npixels=npixels, deblend=deblend,
                               reject_spikes=reject_spikes, verbose=verbose)
        noun = "segments"
    else:
        found = detect_dao(data, level, std, fwhm_guess=fwhm_guess, nsigma=nsigma)
        noun = "sources"

    if found is not None:
        (xpeak, ypeak), n_found = found
        if verbose and n_found > 1:
            print(f"[detect] {n_found} {noun} found; using the brightest at "
                  f"({xpeak:.1f}, {ypeak:.1f})")
    else:
        fallback = detect_peak(data, level, std, fwhm_guess=fwhm_guess, nsigma=nsigma)
        if fallback is None:
            msg = (
                f"no source detected: the {method} method and the peak search both "
                f"came up empty at a {nsigma:g}-sigma threshold. Try lowering "
                "--nsigma or adjusting --fwhm-guess."
            )
            raise RuntimeError(msg)
        xpeak, ypeak = fallback
        print(f"[detect] the {method} method found nothing; fell back to a smoothed "
              "peak search")

    return xpeak, ypeak


def measure_fwhm(data, position, background, fwhm_guess=DEFAULT_FWHM_GUESS):
    """Measure the source FWHM from the area of its half-maximum footprint.

    The width is read off the number of pixels standing above half the peak, as
    `2 * sqrt(N / pi)` - the diameter of the circle of that area - rather than
    from a fitted model.

    Fitting a Gaussian, the obvious alternative, measures the wrong thing here. A
    fiber output is a near-field disc convolved with the PSF: flat-topped,
    steep-shouldered, and carrying more light in its wings than a Gaussian has.
    On a stack of real frames the profile stands 0.12 above the best-fit Gaussian
    at r = 4 px, dips 0.05 below it at r = 6, and runs above it again from r = 10
    outwards - structure, not scatter. A fit to a shape it cannot represent
    settles on a compromise between core and wings, and the balance of that
    compromise moves with how much of the wing clears the noise floor, so the
    fitted width drifts with source brightness even though the optics never
    change. Counting area assumes nothing about the shape, is steadier frame to
    frame (0.09 px against 0.11 px on frames of equal brightness), and costs a
    threshold and a sum instead of a least-squares fit that can fail to converge.

    Only the connected footprint containing the centroid is counted, and the
    reference peak is taken from a 3x3 median of the cutout. Every real frame from
    this camera has hot pixels; they stand far above half maximum, and each one
    would otherwise both inflate the count and, if bright enough, set the
    half-maximum level itself. The source core is flat over far more than 3 px, so
    the median filter leaves it alone.

    This is a diagnostic, not an input to the photometry. The extraction radius is
    fixed precisely so that the flux does not depend on a measured width.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image.
    position : tuple of float
        The (x, y) source centroid.
    background : float
        Background level to subtract before thresholding.
    fwhm_guess : float, optional
        Approximate FWHM in pixels; sets the size of the search box and is
        returned as a fallback if no footprint can be measured.

    Returns
    -------
    float
        Measured FWHM in pixels.
    """
    half_size = int(np.ceil(2.0 * fwhm_guess))
    xc, yc = int(round(position[0])), int(round(position[1]))
    y0, y1 = max(yc - half_size, 0), min(yc + half_size + 1, data.shape[0])
    x0, x1 = max(xc - half_size, 0), min(xc + half_size + 1, data.shape[1])
    cutout = data[y0:y1, x0:x1] - background

    peak = float(ndimage.median_filter(cutout, size=3).max())
    if not np.isfinite(peak) or peak <= 0.0:
        print("[fwhm] no positive peak above the background; using --fwhm-guess instead")
        return fwhm_guess

    labels, _ = ndimage.label(cutout >= 0.5 * peak)
    at_source = int(labels[yc - y0, xc - x0])
    if at_source == 0:
        print("[fwhm] the centroid does not sit inside the half-maximum footprint; "
              "using --fwhm-guess instead")
        return fwhm_guess

    footprint = labels == at_source
    if (footprint[0, :].any() or footprint[-1, :].any()
            or footprint[:, 0].any() or footprint[:, -1].any()):
        print("[fwhm] the half-maximum footprint reaches the edge of the search box; "
              "the width is a lower limit - raise --fwhm-guess")

    return 2.0 * np.sqrt(int(np.count_nonzero(footprint)) / np.pi)


def estimate_background_global(data, stats=None):
    """Estimate the background from sigma-clipped statistics of the whole frame.

    Appropriate when the source occupies a negligible fraction of the frame, as
    is the case for a single fiber output in a 1440x1080 image.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image.
    stats : tuple, optional
        Precomputed `frame_stats` output, returned unchanged when given.

    Returns
    -------
    tuple
        (level, std, n_pixels) of the sigma-clipped background, where `level` is
        the clipped mean and `n_pixels` is the number of pixels surviving the
        clip.
    """
    return stats if stats is not None else frame_stats(data)


def estimate_background_annulus(data, position, r_in, r_out):
    """Estimate the background from a sigma-clipped annulus around the source.

    Sigma clipping guards against a neighbouring reflection or hot pixel inside
    the annulus biasing the subtraction.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image.
    position : tuple of float
        The (x, y) source centroid.
    r_in, r_out : float
        Inner and outer annulus radii in pixels.

    Returns
    -------
    tuple
        (level, std, n_pixels) of the sigma-clipped annulus pixels, where
        `level` is the clipped mean.
    """
    annulus = CircularAnnulus(position, r_in=r_in, r_out=r_out)

    outside = 1.0 - annulus.area_overlap(data) / annulus.area
    if outside > 0.02:
        print(
            f"WARNING: {100 * outside:.0f}% of the background annulus falls outside "
            "the frame; the background estimate uses only the overlapping pixels."
        )

    stats = ApertureStats(data, annulus, sigma_clip=SigmaClip(sigma=SIGMA_CLIP))
    n_pix = int(np.ma.count(stats.data_cutout))
    return float(stats.mean), float(stats.std), n_pix


def measure_counts(data, position, radius, bkg_level, bkg_std, n_bkg_pix,
                   gain=GAIN_E_PER_ADU, read_noise_e=READ_NOISE_E):
    """Integrate background-subtracted counts in a circular aperture.

    The uncertainty follows the CCD equation: Poisson noise on the source, the
    per-pixel background scatter over the aperture, and the uncertainty on the
    background level itself. Read noise and background Poisson noise are already
    contained in the empirical `bkg_std`, so they are not added again; the scatter
    is only floored at the read-noise level in case the measured value is smaller.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image, not background subtracted.
    position : tuple of float
        The (x, y) source centroid.
    radius : float
        Aperture radius in pixels.
    bkg_level : float
        Background level per pixel, in counts.
    bkg_std : float
        Per-pixel background standard deviation, in counts.
    n_bkg_pix : int
        Number of pixels the background estimate was drawn from.
    gain : float, optional
        Detector gain in e-/ADU.
    read_noise_e : float, optional
        RMS read noise in e-.

    Returns
    -------
    dict
        Keys: `aperture`, `net_counts`, `counts_err`, `snr`, `n_saturated`,
        `sigma_pix`, `area`.
    """
    aperture = CircularAperture(position, r=radius)
    subtracted = data - bkg_level

    sigma_pix = max(bkg_std, read_noise_e / gain)
    error = calc_total_error(subtracted, np.full_like(subtracted, sigma_pix), gain)

    phot = aperture_photometry(subtracted, aperture, error=error, method="exact")
    net_counts = float(phot["aperture_sum"][0])
    aperture_err = float(phot["aperture_sum_err"][0])

    area = float(aperture.area_overlap(data))
    bkg_level_var = area**2 * sigma_pix**2 / max(n_bkg_pix, 1)
    counts_err = float(np.sqrt(aperture_err**2 + bkg_level_var))

    mask = aperture.to_mask(method="center")
    cutout = mask.cutout(data)
    inside = mask.data.astype(bool)
    n_saturated = int(np.count_nonzero(cutout[inside] >= SATURATION_ADU))

    return {
        "aperture": aperture,
        "net_counts": net_counts,
        "counts_err": counts_err,
        "snr": net_counts / counts_err if counts_err > 0 else np.nan,
        "n_saturated": n_saturated,
        "sigma_pix": sigma_pix,
        "area": area,
    }


def display_limits(values):
    """Return a usable (vmin, vmax) pair for displaying an image.

    ZScale is the right stretch for a frame with structure, but it is derived
    from a fit to the sorted pixel values and returns an unusable interval when
    there is nothing to fit. A cutout that is almost entirely one value - a faint
    frame whose background sits at a single count - can come back with vmin
    exceeding vmax by a floating-point rounding error, which `imshow` rejects
    outright with "minvalue must be less than or equal to maxvalue".

    Falls back to the full data range, then to a unit-wide interval around the
    single value present, so a degenerate cutout still plots.

    Parameters
    ----------
    values : numpy.ndarray
        Finite pixel values to derive the stretch from.

    Returns
    -------
    tuple of float
        (vmin, vmax) with vmax strictly greater than vmin.
    """
    from astropy.visualization import ZScaleInterval

    vmin, vmax = ZScaleInterval().get_limits(values)

    if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
        vmin, vmax = float(np.min(values)), float(np.max(values))

    if vmax <= vmin:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    return float(vmin), float(vmax)


def make_figure(data, position, aperture, fwhm, bkg_level, out_path=None, annulus=None,
                show=True, close=True):
    """Build the cutout and curve-of-growth diagnostic figure.

    The cutout axes are shifted so that the source centroid sits at (0, 0), i.e.
    the tick labels give the offset from the source in pixels. pyplot is imported
    here rather than at module scope so that a --no-plot run, which never calls
    this function, does not import matplotlib at all.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image, not background subtracted.
    position : tuple of float
        The (x, y) source centroid.
    aperture : photutils.aperture.CircularAperture
        The measurement aperture.
    fwhm : float
        Measured source FWHM in pixels; drawn as a circle of radius FWHM / 2.
    bkg_level : float
        Background level per pixel, subtracted before building the growth curve.
    out_path : pathlib.Path, optional
        Destination for the PNG. If None the figure is not written to disk, which
        is what a notebook caller wants.
    annulus : photutils.aperture.CircularAnnulus, optional
        Background annulus to overlay, when local background estimation was used.
    show : bool, optional
        If True, open an interactive figure window.
    close : bool, optional
        If True, close the figure before returning. Pass False to keep it alive
        for a notebook, whose inline backend renders the figures still open when
        the cell finishes.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, already closed unless `close` is False.
    """
    import matplotlib.pyplot as plt

    radius = aperture.r
    hwhm = fwhm / 2.0

    half_size = 3.0 * radius
    if annulus is not None:
        half_size = max(half_size, 1.1 * annulus.r_out)
    size = int(np.ceil(2 * half_size))

    cutout = Cutout2D(data, position, size, mode="partial", fill_value=np.nan)
    xcut, ycut = cutout.to_cutout_position(position)

    ny, nx = cutout.data.shape
    extent = [-xcut - 0.5, nx - xcut - 0.5, -ycut - 0.5, ny - ycut - 0.5]

    fig, (ax_img, ax_cog) = plt.subplots(1, 2, figsize=(11, 4.6))

    vmin, vmax = display_limits(cutout.data[np.isfinite(cutout.data)])
    ax_img.imshow(cutout.data, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                  extent=extent)
    CircularAperture((0.0, 0.0), r=radius).plot(
        ax=ax_img, color="white", lw=1.5, label=f"aperture r = {radius:.1f} px"
    )
    CircularAperture((0.0, 0.0), r=hwhm).plot(
        ax=ax_img, color="deepskyblue", lw=1.2, label=f"HWHM r = {hwhm:.1f} px"
    )
    if annulus is not None:
        CircularAnnulus((0.0, 0.0), r_in=annulus.r_in, r_out=annulus.r_out).plot(
            ax=ax_img, color="orange", lw=1.0, ls="--", label="background annulus"
        )
    ax_img.plot(0.0, 0.0, "r+", ms=8)
    ax_img.set_xlim(extent[0], extent[1])
    ax_img.set_ylim(extent[2], extent[3])
    ax_img.set_title(f"source at ({position[0]:.1f}, {position[1]:.1f})")
    ax_img.set_xlabel("x offset from source [px]")
    ax_img.set_ylabel("y offset from source [px]")
    ax_img.legend(loc="upper right", fontsize=8, framealpha=0.6)

    radii = np.linspace(1.0, 1.5 * radius, 60)
    cog = CurveOfGrowth(data - bkg_level, position, radii)
    ax_cog.plot(cog.radius, cog.profile, color="k", lw=1.2)
    ax_cog.axvline(radius, color="crimson", ls="--", lw=1.2, label=f"r = {radius:.1f} px")
    ax_cog.axvline(hwhm, color="deepskyblue", ls=":", lw=1.2, label=f"HWHM = {hwhm:.1f} px")
    ax_cog.set_xlabel("aperture radius [px]")
    ax_cog.set_ylabel("enclosed counts")
    ax_cog.set_title("curve of growth")
    ax_cog.legend(loc="lower right", fontsize=8)

    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)

    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig


def update_csv(csv_path, rows):
    """Write measurements to the results CSV, keeping one row per image.

    The existing table and the new measurements are concatenated, then only the
    last row for each image is kept, so a re-measured frame moves to the bottom
    and any duplicates already sitting in the table collapse too, whether or not
    this run touched them. Filenames are compared as resolved paths, so different
    spellings of the same file still match. Deduplication keeps the last
    occurrence by index rather than assigning into a dict keyed by filename,
    because reassigning a dict key preserves its original position and would
    leave re-measured frames where they were.

    The table is written once, via a temporary file and an atomic replace, so an
    interrupted run cannot leave it truncated. Called once per invocation rather
    than once per frame, both to avoid rewriting the table N times and so that a
    failed batch does not leave it half updated.

    Parameters
    ----------
    csv_path : pathlib.Path
        Destination CSV. Created along with its parent directory if absent.
    rows : list of dict
        Measurement values keyed by the names in `CSV_COLUMNS`.

    Returns
    -------
    tuple
        (path, n_removed) where `n_removed` is the number of superseded rows
        dropped from the table.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    def key(name):
        """Return a comparable form of a filename, resolving it where possible."""
        try:
            return str(Path(name).resolve())
        except OSError:
            return str(name)

    existing = []
    if csv_path.exists():
        with csv_path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))

    combined = existing + list(rows)
    last_index = {key(row.get("filename", "")): i for i, row in enumerate(combined)}
    kept = [row for i, row in enumerate(combined)
            if last_index[key(row.get("filename", ""))] == i]

    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, restval="",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    tmp_path.replace(csv_path)

    return csv_path, len(combined) - len(kept)


def resolve_outfile(outfile):
    """Resolve the --outfile value to the CSV path to write.

    A value carrying a directory component is used exactly as given, absolute or
    relative; a bare filename is placed inside `RESULTS_DIR`, so `run01.csv`
    means `results/run01.csv`.

    Note that pathlib normalises `./run01.csv` to `run01.csv`, which therefore
    still lands in `results/`. Writing to the current working directory needs an
    explicit absolute path.

    Parameters
    ----------
    outfile : str or pathlib.Path
        The value given to --outfile.

    Returns
    -------
    pathlib.Path
        The CSV path to write to.
    """
    path = Path(outfile)
    return path if path.parent != Path(".") else RESULTS_DIR / path.name


def parse_args(argv=None):
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Argument list to parse. Defaults to `sys.argv[1:]`.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Aperture photometry of a single fiber output in TIFF or FITS frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", metavar="INPUT",
                        help="TIFF or FITS frames, directories whose top-level "
                             "images are measured, or a text file listing either one "
                             "per line")
    parser.add_argument("--fwhm-guess", type=float, default=DEFAULT_FWHM_GUESS,
                        help="approximate source FWHM in pixels, used for detection")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS,
                        help="fixed aperture radius in pixels")
    parser.add_argument("--k-fwhm", type=float, default=None,
                        help="scale the aperture radius to the measured FWHM by this "
                             "factor instead of using --radius. Off by default, and "
                             "best left off: the measured width drifts with source "
                             "brightness, so an aperture scaled to it folds that "
                             "drift into the flux. Kept for comparison")
    parser.add_argument("--no-reuse-centroid", action="store_true",
                        help="detect the source on every frame. By default it is "
                             "detected once, on the first frame that succeeds, and "
                             "every later frame is measured at that centroid: the "
                             "fiberhead does not move, detection is ~470 ms of a "
                             "~520 ms frame, and the aperture is large enough that a "
                             "fraction of a pixel of misplacement does not move the "
                             "flux. Frames measured at the inherited centroid record "
                             "'reused' as their detect_method")
    parser.add_argument("--nsigma", type=float, default=DEFAULT_NSIGMA,
                        help="detection threshold in background sigma")
    parser.add_argument("--detect-method", choices=DETECT_METHODS,
                        default=DEFAULT_DETECT_METHOD,
                        help="how to find the source. 'segment' groups connected "
                             "pixels above the threshold into segments and takes the "
                             "brightest, which suits an extended fiber output and "
                             "rejects hot pixels and small noise clumps by requiring "
                             "at least --npixels connected pixels. 'dao' correlates "
                             "against a Gaussian PSF instead, and is kept for "
                             "comparison")
    parser.add_argument("--npixels", type=int, default=DEFAULT_NPIXELS,
                        help="smallest connected area in pixels a segmentation "
                             "detection may have (--detect-method segment only)")
    parser.add_argument("--no-deblend", action="store_true",
                        help="do not deblend merged segments (--detect-method segment "
                             "only). Deblending is what keeps a fiber sitting inside "
                             "diffuse scattered light from being absorbed into the "
                             "glow's segment; it is free on a clean frame but costs "
                             "seconds on one with hundreds of segments")
    parser.add_argument("--keep-spikes", action="store_true",
                        help="do not discard detections whose counts sit almost "
                             "entirely in one pixel (--detect-method segment only). "
                             "Such detections are hot pixels or cosmic rays, which "
                             "the --npixels area cut cannot reject because detection "
                             "runs on the convolved frame; keep them to see the raw "
                             "segment count")
    parser.add_argument("--annulus-bkg", action="store_true",
                        help="estimate the background from a local annulus instead of "
                             "the sigma-clipped whole frame")
    parser.add_argument("--r-in", type=float, default=ANNULUS_IN_FWHM,
                        help="inner annulus radius in units of the measured FWHM")
    parser.add_argument("--r-out", type=float, default=ANNULUS_OUT_FWHM,
                        help="outer annulus radius in units of the measured FWHM")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print the diagnostic notes the measurement makes along "
                             "the way: how a stack was reduced, how many detections "
                             "were found, which were discarded as spikes. Notes about "
                             "a degraded or unreliable result are printed either way")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print the per-frame summary or the batch "
                             "progress lines. Warnings and errors are still printed")
    parser.add_argument("--gain", type=float, default=GAIN_E_PER_ADU,
                        help="detector gain in e-/ADU")
    parser.add_argument("--read-noise", type=float, default=READ_NOISE_E,
                        help="RMS read noise in e-")
    parser.add_argument("--plane", type=int, default=0,
                        help="plane to use from a multi-plane TIFF or FITS stack")
    parser.add_argument("--outfile", type=Path, default=DEFAULT_OUTFILE,
                        help=f"CSV file to write. A value with a directory component "
                             f"is used as given (--outfile runs/run01.csv, --outfile "
                             f"/data/x.csv); a bare filename goes inside "
                             f"{RESULTS_DIR}/, so --outfile run01.csv writes "
                             f"{RESULTS_DIR}/run01.csv. Note that ./run01.csv "
                             f"normalises to run01.csv and so also lands in "
                             f"{RESULTS_DIR}/ - use an absolute path to write to the "
                             f"current directory. Diagnostic figures are not affected: "
                             f"they always go to {PLOT_DIR}/")
    parser.add_argument("--no-plot", action="store_true",
                        help=f"skip the diagnostic figure entirely - nothing displayed "
                             f"and nothing written to {PLOT_DIR}/. Saves ~0.15 s per "
                             f"frame plus a one-off ~0.3 s of pyplot import, or ~1.4 s "
                             f"for a single frame, where displaying the window also "
                             f"starts an interactive backend. Windows are suppressed "
                             f"automatically when measuring more than one frame, but "
                             f"the PNGs are still written")
    parser.add_argument("--no-csv", action="store_true",
                        help="do not write results to the CSV table")

    return parser.parse_args(argv)


def measure_single_image(data, name=None, fwhm_guess=DEFAULT_FWHM_GUESS,
                         radius=DEFAULT_RADIUS, k_fwhm=None, position=None,
                         nsigma=DEFAULT_NSIGMA,
                         detect_method=DEFAULT_DETECT_METHOD, npixels=DEFAULT_NPIXELS,
                         deblend=True, reject_spikes=True, annulus_bkg=False, r_in=ANNULUS_IN_FWHM,
                         r_out=ANNULUS_OUT_FWHM, gain=GAIN_E_PER_ADU,
                         read_noise_e=READ_NOISE_E, plot=False, verbose=False,
                         print_output=False):
    """Measure one already-loaded frame and return the result as numbers.

    This is the whole measurement in one call, and the only definition of it:
    the command line reaches it through `measure_frame`, a notebook through
    `measure_images` or directly. It takes an array rather than a path so that
    frames already in memory - synthetic frames, a page pulled out of a stack,
    anything preprocessed - can be measured without a round trip through disk.

    Unlike the CSV row built by `csv_row`, every value returned here is a number
    at full precision, ready to be computed with.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image, not background subtracted.
    name : str, optional
        Label for this frame, passed through to the `name` field of the result.
    fwhm_guess : float, optional
        Expected source FWHM in pixels, used to size the detection kernel and the
        FWHM search box, and returned as the fallback if the half-maximum
        footprint cannot be measured.
    radius : float, optional
        Fixed aperture radius in pixels, used unless `k_fwhm` is given.
    k_fwhm : float, optional
        Aperture radius in units of the measured FWHM, overriding `radius`. Off by
        default, and best left off: a width measured from the frame drifts with
        source brightness, so scaling the aperture by one makes the flux depend on
        how bright the source happened to be. Kept for comparison.
    position : tuple of float, optional
        Centroid (x, y) to measure at, skipping detection entirely. The fiberhead
        does not move, so one centroid can serve a whole run; `measure_images` and
        the command line do that for themselves, see `reuse_centroid`. A frame
        measured this way reports `reused` as its `detect_method`.
    nsigma : float, optional
        Detection threshold in units of the background scatter.
    detect_method : {'segment', 'dao'}, optional
        Detection method, as for --detect-method.
    npixels : int, optional
        Smallest connected area a segmentation detection may have.
    deblend : bool, optional
        Whether to deblend segments before picking the brightest.
    reject_spikes : bool, optional
        Whether to discard hot pixels and cosmic rays before picking the
        brightest detection.
    annulus_bkg : bool, optional
        If True, estimate the background from a local annulus rather than from
        the whole frame.
    r_in, r_out : float, optional
        Annulus radii in units of the measured FWHM. Used only when `annulus_bkg`.
    gain : float, optional
        Detector gain in e-/ADU.
    read_noise_e : float, optional
        RMS read noise in e-.
    plot : bool, optional
        If True, build the diagnostic figure and return it under the `figure`
        key. Nothing is written to disk and the figure is left open, so a
        notebook's inline backend renders it when the cell finishes.
    verbose : bool, optional
        If True, print the diagnostic notes the measurement makes along the way:
        how a stack was reduced, how many detections were found, which were
        discarded as spikes. Messages about a degraded or unreliable result are
        printed either way.
    print_output : bool, optional
        If True, print the same summary of the headline numbers the command line
        prints. Off by default, since a notebook has the returned dict.

    Returns
    -------
    dict
        The measurement. The `RESULT_COLUMNS` keys - `name`, `x`, `y`,
        `fwhm_pix`, `radius_pix`, `detect_method` (or `reused` when `position`
        was supplied), `bkg_method`, `bkg_level`,
        `bkg_std`, `n_bkg_pix`, `net_counts`, `counts_err`, `snr`,
        `n_saturated`, `area`, `sigma_pix`, `gain`, `read_noise_e` - are all
        numbers or strings. Alongside them are the photutils objects the
        measurement used, `aperture` and `annulus` (None unless `annulus_bkg`),
        and `figure` when `plot` is True.

    Raises
    ------
    ValueError
        If `data` is not two-dimensional.
    RuntimeError
        If no source is detected.

    See Also
    --------
    measure_images : the same measurement over a list of frames, as a DataFrame.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        msg = f"expected a 2D image, got an array of shape {data.shape}"
        raise ValueError(msg)

    stats = frame_stats(data, verbose=verbose)

    if position is None:
        position = locate_source(data, fwhm_guess=fwhm_guess, nsigma=nsigma,
                                 stats=stats, method=detect_method, npixels=npixels,
                                 deblend=deblend, reject_spikes=reject_spikes,
                                 verbose=verbose)
        position_from = detect_method
    else:
        position = (float(position[0]), float(position[1]))
        position_from = "reused"
        if verbose:
            print(f"[detect] measuring at the supplied centroid ({position[0]:.1f}, "
                  f"{position[1]:.1f}); detection skipped")

    fwhm = measure_fwhm(data, position, stats[0], fwhm_guess=fwhm_guess)
    aperture_radius = k_fwhm * fwhm if k_fwhm is not None else radius

    annulus = None
    if annulus_bkg:
        ann_in, ann_out = r_in * fwhm, r_out * fwhm
        bkg_level, bkg_std, n_bkg_pix = estimate_background_annulus(
            data, position, ann_in, ann_out
        )
        annulus = CircularAnnulus(position, r_in=ann_in, r_out=ann_out)
        bkg_method = f"annulus[{ann_in:.1f}-{ann_out:.1f}]"
    else:
        bkg_level, bkg_std, n_bkg_pix = estimate_background_global(data, stats=stats)
        bkg_method = "global"

    counts = measure_counts(
        data, position, aperture_radius, bkg_level, bkg_std, n_bkg_pix,
        gain=gain, read_noise_e=read_noise_e,
    )

    result = {
        "name": name,
        "x": float(position[0]),
        "y": float(position[1]),
        "fwhm_pix": float(fwhm),
        "radius_pix": float(aperture_radius),
        "detect_method": position_from,
        "bkg_method": bkg_method,
        "bkg_level": float(bkg_level),
        "bkg_std": float(bkg_std),
        "n_bkg_pix": int(n_bkg_pix),
        "net_counts": counts["net_counts"],
        "counts_err": counts["counts_err"],
        "snr": counts["snr"],
        "n_saturated": counts["n_saturated"],
        "area": counts["area"],
        "sigma_pix": counts["sigma_pix"],
        "gain": gain,
        "read_noise_e": read_noise_e,
        "aperture": counts["aperture"],
        "annulus": annulus,
    }

    if print_output:
        report_measurement(result)

    # Saturation is a property of the data, not a running commentary on the
    # measurement, so it is reported whatever the printing flags say: the counts
    # it qualifies are wrong in a way no other line of output reveals.
    if result["n_saturated"] > 0:
        print(f"WARNING: {result['n_saturated']} saturated pixels "
              f"(>= {SATURATION_ADU} counts) inside the aperture - the measured "
              "counts are a lower limit.")

    if plot:
        result["figure"] = make_figure(
            data, position, counts["aperture"], fwhm, bkg_level,
            annulus=annulus, show=False, close=False,
        )

    return result


def report_measurement(result):
    """Print the headline numbers of a measurement to stdout.

    Kept out of `measure_single_image` so that the command line and a notebook call
    print the same thing from one definition. The saturation warning is not part of
    it: that is a data-quality warning, printed by `measure_single_image` whether or
    not this summary is.

    Parameters
    ----------
    result : dict
        A result from `measure_single_image`. The leading `file` line is omitted when
        its `name` is None.

    Returns
    -------
    None
    """
    if result["name"] is not None:
        print(f"\nfile        : {result['name']}")
    print(f"centroid    : ({result['x']:.2f}, {result['y']:.2f}) px "
          f"[{result['detect_method']}]")
    print(f"FWHM        : {result['fwhm_pix']:.2f} px")
    print(f"aperture    : r = {result['radius_pix']:.2f} px  "
          f"(area {result['area']:.1f} px^2)")
    print(f"background  : {result['bkg_level']:.3f} +/- {result['bkg_std']:.3f} "
          f"counts/px [{result['bkg_method']}, {result['n_bkg_pix']} px]")
    print(f"counts      : {result['net_counts']:.4e} +/- {result['counts_err']:.2e} "
          f"(SNR {result['snr']:.1f})")


def measure_images(images, plane=0, plot=False, verbose=False, print_output=False,
                   on_error="skip", reuse_centroid=True, **params):
    """Measure a list of frames and return the results as a DataFrame.

    The notebook counterpart of the command line: the same measurement, no files
    written, and the results as numeric columns rather than a CSV of formatted
    strings.

    Parameters
    ----------
    images : sequence
        The frames to measure, as a list of 2D arrays or a list of paths (`str`
        or `pathlib.Path`); the two may be mixed, since each element is
        dispatched on its own type. A single array or a single path is accepted
        as a one-element list. Paths are read with `load_image`.
    plane : int, optional
        Plane to select from a multi-plane stack. Applies to path inputs only.
    plot : bool, optional
        If True, build a diagnostic figure per frame. The figures are left open
        rather than saved, so a notebook's inline backend renders them when the
        cell finishes; they are not part of the returned table.
    verbose : bool, optional
        If True, print the diagnostic notes each measurement makes along the way.
        Messages about a degraded or unreliable result are printed either way.
    print_output : bool, optional
        If True, print the summary of the headline numbers for each frame. Off by
        default, since the returned table has them.
    reuse_centroid : bool, optional
        If True, the default, detect the source on the first frame that succeeds
        and measure every later frame at that same centroid. The fiberhead is
        fixed, so the spot does not move, and detection is by far the most
        expensive step of a measurement - around 470 ms of a ~520 ms frame. The
        extraction radius sits on the flat part of the curve of growth, where the
        enclosed flux barely responds to a fraction of a pixel of misplacement, so
        nothing is given up by not re-finding it. Frames measured at the inherited
        centroid report `reused` as their `detect_method`, so the table shows
        which row the position came from. Pass False to detect on every frame; an
        explicit `position` overrides this either way.
    on_error : {'raise', 'skip'}, optional
        What to do when a frame fails. 'skip', the default, reports the failure
        on stderr and still appends a row for it, with every column NaN except
        `name`, so a bad frame does not shift later rows out of correspondence
        with their inputs. 'raise' propagates the exception instead.
    **params
        Forwarded to `measure_single_image`, so every measurement option is available
        by keyword: `fwhm_guess`, `radius`, `k_fwhm`, `position`, `nsigma`,
        `detect_method`, `npixels`, `deblend`, `annulus_bkg`, `r_in`, `r_out`,
        `gain`, `read_noise_e`.

    Returns
    -------
    pandas.DataFrame
        One row per input frame, with the columns named in `RESULT_COLUMNS`. The
        `name` column holds the path for path inputs and `image_<index>` for
        array inputs. A frame that failed under `on_error='skip'` is present
        with `name` set and every other column NaN. Empty input gives an empty
        DataFrame with those columns.

    Raises
    ------
    ValueError
        If `on_error` is not 'raise' or 'skip'.

    Examples
    --------
    >>> df = measure_images(my_arrays)                             # doctest: +SKIP
    >>> df = measure_images(paths, radius=30.0, annulus_bkg=True)  # doctest: +SKIP
    """
    # pandas is as expensive to import as astropy and the command line never
    # needs it, so it is imported here rather than at module scope.
    import pandas as pd

    if on_error not in ("raise", "skip"):
        msg = f"on_error must be 'raise' or 'skip', not {on_error!r}"
        raise ValueError(msg)

    if isinstance(images, (str, Path)) or (
        isinstance(images, np.ndarray) and images.ndim == 2
    ):
        images = [images]

    rows = []
    # Held across the loop so the source is detected once and inherited from
    # there. A frame that fails leaves it unset, so the next frame detects again
    # rather than the whole run going undetected because the first frame was bad.
    inherited = None

    for index, image in enumerate(images):
        is_path = isinstance(image, (str, Path))
        name = str(image) if is_path else f"image_{index}"

        try:
            data = (load_image(image, plane=plane, verbose=verbose)
                    if is_path else image)
            frame_params = dict(params)
            if reuse_centroid and inherited is not None:
                frame_params.setdefault("position", inherited)
            result = measure_single_image(data, name=name, plot=plot,
                                          verbose=verbose,
                                          print_output=print_output, **frame_params)
            rows.append(result)
            if reuse_centroid and inherited is None:
                inherited = (result["x"], result["y"])
        except Exception as exc:  # noqa: BLE001 - the policy is the caller's
            if on_error == "raise":
                raise
            print(f"ERROR: {name}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
            rows.append({col: (name if col == "name" else np.nan)
                        for col in RESULT_COLUMNS})

    # Selecting the columns explicitly drops the photutils objects and the
    # figure, which have no place in a table.
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def csv_row(result):
    """Format a measurement as a row of the results CSV.

    Parameters
    ----------
    result : dict
        A result from `measure_single_image`. Its `name` becomes the `filename` column.

    Returns
    -------
    dict
        A row keyed by the names in `CSV_COLUMNS`.
    """
    return {
        "filename": result["name"],
        "x": f"{result['x']:.3f}",
        "y": f"{result['y']:.3f}",
        "fwhm_pix": f"{result['fwhm_pix']:.3f}",
        "radius_pix": f"{result['radius_pix']:.3f}",
        "detect_method": result["detect_method"],
        "bkg_method": result["bkg_method"],
        "bkg_level": f"{result['bkg_level']:.4f}",
        "bkg_std": f"{result['bkg_std']:.4f}",
        "net_counts": f"{result['net_counts']:.6e}",
        "counts_err": f"{result['counts_err']:.6e}",
        "snr": f"{result['snr']:.2f}",
        "n_saturated": result["n_saturated"],
        "gain": result["gain"],
        "read_noise_e": result["read_noise_e"],
    }


def measure_frame(image_path, args, show=False, position=None):
    """Measure one frame from disk and report the result to stdout.

    The command-line adapter around `measure_single_image`: it reads the file, unpacks
    the parsed options into keyword arguments, and writes the diagnostic figure
    to `PLOT_DIR/<stem>_aperture.png` unless --no-plot was given.

    Parameters
    ----------
    image_path : pathlib.Path
        The TIFF or FITS frame to measure.
    args : argparse.Namespace
        Parsed command-line options.
    show : bool, optional
        If True, open an interactive figure window for this frame.
    position : tuple of float, optional
        Centroid (x, y) to measure at, skipping detection. `main` passes the
        centroid found on the first frame of a batch unless --no-reuse-centroid.

    Returns
    -------
    tuple
        The CSV row of the measurement, keyed by the names in `CSV_COLUMNS`, and
        the (x, y) centroid it was measured at, as numbers. The centroid is
        returned separately because the row holds it as a formatted string, and
        `main` needs it back as a number to pass to the rest of the batch.

    Raises
    ------
    Exception
        Propagates whatever the underlying readers and photometry raise, for
        example `tifffile.TiffFileError` for an unreadable frame, `ValueError`
        for one that is not 2D, or `RuntimeError` if no source is detected. The
        caller decides whether that ends the run.

    See Also
    --------
    measure_single_image : the measurement itself, for callers that are not the CLI.
    locate_source : detection, selected by --detect-method.
    """
    data = load_image(image_path, plane=args.plane, verbose=args.verbose)

    result = measure_single_image(
        data, name=str(image_path), fwhm_guess=args.fwhm_guess,
        radius=args.radius, k_fwhm=args.k_fwhm, position=position,
        nsigma=args.nsigma, detect_method=args.detect_method,
        npixels=args.npixels, deblend=not args.no_deblend,
        reject_spikes=not args.keep_spikes,
        annulus_bkg=args.annulus_bkg, r_in=args.r_in, r_out=args.r_out,
        gain=args.gain, read_noise_e=args.read_noise, verbose=args.verbose,
        print_output=not args.quiet,
    )

    if not args.no_plot:
        figure_path = PLOT_DIR / f"{image_path.stem}_aperture.png"
        make_figure(data, (result["x"], result["y"]), result["aperture"],
                    result["fwhm_pix"], result["bkg_level"], figure_path,
                    annulus=result["annulus"], show=show)
        if not args.quiet:
            print(f"figure      : {figure_path}")

    return csv_row(result), (result["x"], result["y"])


def main(argv=None):
    """Measure every requested frame and write the results.

    A frame that cannot be read or has no detectable source is reported on
    stderr and given a placeholder CSV row - every column NaN except `filename`
    - so one bad file does not abandon the batch or shift later rows out of
    correspondence with their inputs. The CSV named by --outfile is written once,
    after all frames are measured; diagnostic figures go to `PLOT_DIR`.

    Parameters
    ----------
    argv : list of str, optional
        Argument list to parse. Defaults to `sys.argv[1:]`.

    Returns
    -------
    int
        Process exit status: 0 if every frame succeeded, 1 otherwise.
    """
    args = parse_args(argv)

    try:
        frames = gather_frames(args.inputs)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # An interactive window per frame is unusable in a batch, so figures are only
    # displayed for a single frame; a headless backend is needed when they are
    # saved without being shown. With --no-plot matplotlib is never imported.
    show = not args.no_plot and len(frames) == 1
    if not args.no_plot and not show:
        import matplotlib
        matplotlib.use("Agg")

    if len(frames) > 1 and not args.quiet:
        print(f"measuring {len(frames)} frames")

    rows = []
    failures = []
    # Detected on the first frame that succeeds and inherited by the rest of the
    # batch, unless --no-reuse-centroid. A frame that fails leaves this unset, so
    # the next one detects rather than the batch inheriting nothing.
    inherited = None

    for frame in frames:
        try:
            row, position = measure_frame(frame, args, show=show, position=inherited)
            rows.append(row)
            if not args.no_reuse_centroid and inherited is None:
                inherited = position
        except Exception as exc:  # noqa: BLE001 - one bad frame must not end the batch
            failures.append(frame)
            print(f"ERROR: {frame}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
            rows.append({col: (str(frame) if col == "filename" else "nan")
                        for col in CSV_COLUMNS})

    if rows and not args.no_csv:
        csv_path, n_removed = update_csv(resolve_outfile(args.outfile), rows)
        removed = f" (removed {n_removed} superseded row"
        removed += "s)" if n_removed > 1 else ")"
        if not args.quiet:
            print(f"\ncsv         : {csv_path}{removed if n_removed else ''}")

    if (len(frames) > 1 or failures) and not args.quiet:
        print(f"measured {len(rows) - len(failures)}/{len(frames)} frames"
              + (f", {len(failures)} failed" if failures else ""))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
