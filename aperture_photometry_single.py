#!/usr/bin/env python3
"""Aperture photometry for a single fiber output in MAST fiberhead images.

Locates the one bright spot in a TIFF frame, measures its FWHM, subtracts a
background estimated either globally or from a local annulus, and integrates the
counts inside a circular aperture whose radius is scaled to the measured FWHM.

The spot is found by segmentation: the brightest group of connected pixels above
the detection threshold that covers at least --npixels pixels. Requiring an area
is what keeps hot pixels and small noise clumps from being taken for the fiber on
a faint frame. --detect-method dao selects PSF-correlation detection instead.

Outputs a row in a CSV table of measurements (see --outfile), a summary printed
to stdout, and a two-panel diagnostic figure (annotated cutout + curve of growth)
written to `plots/`. The figures are throwaway sanity checks, so their location
is fixed and untracked; --no-plot skips them entirely.

Any number of frames can be measured in one invocation, given as files, as
directories (whose top-level TIFFs are measured), or as a text file listing
either of those one per line. Batching this way is worth it: the astropy and
photutils imports cost ~0.6 s and are paid once per invocation rather than once
per frame.

Examples
--------
    python aperture_photometry_single.py data/run01/frame_001.tif
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
    df = measure_images(paths, k_fwhm=3.0)      # or a list of paths
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
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)
from photutils.centroids import centroid_quadratic, centroid_sources
from photutils.detection import DAOStarFinder, find_peaks
from photutils.profiles import CurveOfGrowth, RadialProfile
from photutils.segmentation import (
    SourceCatalog,
    deblend_sources,
    detect_sources,
    make_2dgaussian_kernel,
)
from photutils.utils import calc_total_error

# Detector properties of the fiberhead test camera.
GAIN_E_PER_ADU = 1.0
READ_NOISE_E = 0.46
SATURATION_ADU = 1023  # 10-bit sensor

# Measurement defaults.
DEFAULT_FWHM_GUESS = 11.0  # pixels
DEFAULT_K_FWHM = 2.5  # aperture radius in units of the measured FWHM
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

DETECT_METHODS = ("segment", "dao")
DEFAULT_DETECT_METHOD = "segment"

TIFF_SUFFIXES = {".tif", ".tiff"}

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
    "bkg_median",
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
    "bkg_median",
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

    An entry may be a TIFF file, a directory (whose top-level TIFFs are taken,
    sorted by name, without recursing), or a text file listing files and
    directories one per line. Blank lines and lines starting with `#` are ignored
    in a list file, and relative paths in it are resolved against the directory
    the list itself lives in.

    Parameters
    ----------
    entry : str or pathlib.Path
        The path to expand.
    allow_list : bool, optional
        Whether a non-TIFF file may be read as a list of paths. Set False when
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
        If a directory holds no TIFFs, or a list file is referenced from inside
        another list file.
    """
    path = Path(entry)

    if not path.exists():
        msg = f"no such file or directory: {path}"
        raise FileNotFoundError(msg)

    if path.is_dir():
        frames = sorted(p for p in path.iterdir()
                        if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES)
        if not frames:
            msg = (f"directory {path} contains no TIFF files (looked for "
                   f"{', '.join(sorted(TIFF_SUFFIXES))} at the top level only)")
            raise ValueError(msg)
        return frames

    if path.suffix.lower() in TIFF_SUFFIXES:
        return [path]

    if not allow_list:
        msg = f"{path} is not a TIFF or a directory, and list files cannot nest"
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


def load_image(path, plane=0):
    """Read a TIFF frame as a 2D float array.

    Multi-page stacks are reduced by selecting a single page; RGB(A) images are
    reduced by averaging the colour channels.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the TIFF file.
    plane : int, optional
        Page to select from a multi-page stack. Ignored for 2D and RGB(A) input.

    Returns
    -------
    numpy.ndarray
        Two-dimensional image in float64, in the original count units.

    Raises
    ------
    ValueError
        If the file does not reduce to a 2D image, or `plane` is out of range.
    """
    raw = tifffile.imread(str(path))

    if raw.ndim == 2:
        image = raw
    elif raw.ndim == 3 and raw.shape[-1] in (3, 4):
        image = raw[..., :3].mean(axis=-1)
        print(f"[load] RGB(A) TIFF {raw.shape} -> averaged colour channels")
    elif raw.ndim == 3:
        if not 0 <= plane < raw.shape[0]:
            msg = f"plane {plane} out of range for a stack of {raw.shape[0]} pages"
            raise ValueError(msg)
        image = raw[plane]
        print(f"[load] multi-page TIFF {raw.shape} -> using page {plane}")
    else:
        msg = f"cannot reduce a TIFF of shape {raw.shape} to a 2D image"
        raise ValueError(msg)

    return np.asarray(image, dtype=np.float64)


def frame_stats(data):
    """Return sigma-clipped statistics for the whole frame.

    Computed once per frame and reused for the detection threshold, the profile
    background and (unless a local annulus is requested) the background
    subtraction itself, since clipping a 1440x1080 frame is not free.

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

    Returns
    -------
    tuple
        (median, std, n_pixels) of the sigma-clipped frame, where `std` is not
        less than `DIGITIZATION_SIGMA` and `n_pixels` is the number of pixels
        surviving the clip.
    """
    clipped = SigmaClip(sigma=SIGMA_CLIP)(data.ravel(), masked=True)
    median, std = float(np.ma.median(clipped)), float(np.ma.std(clipped))

    if std < DIGITIZATION_SIGMA:
        # Only worth reporting when the clip has actually lost the noise tail. A
        # frame whose scatter merely rounds down to just under the floor is not
        # telling the reader anything.
        if std < 0.9 * DIGITIZATION_SIGMA:
            print(f"[stats] sigma-clipped scatter is {std:.3f} counts/px, below the "
                  f"digitization floor; using {DIGITIZATION_SIGMA:.3f} instead")
        std = DIGITIZATION_SIGMA

    return median, std, int(clipped.count())


def detect_segment(data, median, std, fwhm_guess=DEFAULT_FWHM_GUESS,
                   nsigma=DEFAULT_NSIGMA, npixels=DEFAULT_NPIXELS, deblend=True):
    """Find the brightest source by image segmentation.

    The frame is background-subtracted and convolved with a Gaussian kernel
    matched to the expected source size, then pixels above `nsigma` times the
    background scatter are grouped into connected segments of at least `npixels`
    pixels. The brightest segment by total flux is taken as the source.

    Requiring a minimum connected area is what makes this robust on faint frames:
    a fiber output is an extended blob, so single-pixel spikes and two- or
    three-pixel noise clumps are rejected structurally rather than incidentally,
    as they are by DAOStarFinder's PSF-correlation and sharpness cuts.

    Deblending splits segments that merge several peaks. It matters whenever the
    frame carries diffuse scattered light: without it, a bright fiber sitting
    inside a large low-level glow is absorbed into the glow's segment, and the
    reported position is the centroid of the glow rather than of the fiber.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image.
    median : float
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

    Returns
    -------
    tuple or None
        ((x, y), n_segments) for the brightest segment, or None if nothing was
        detected.
    """
    kernel_size = 2 * int(np.ceil(fwhm_guess)) + 1
    convolved = convolve(data - median, make_2dgaussian_kernel(fwhm_guess, kernel_size))

    segments = detect_sources(convolved, nsigma * std, n_pixels=npixels)
    if segments is None:
        return None

    if deblend:
        segments = deblend_sources(convolved, segments, npixels, progress_bar=False)

    catalog = SourceCatalog(data - median, segments, convolved_data=convolved)
    flux = np.asarray(catalog.segment_flux, dtype=float)
    if not np.isfinite(flux).any():
        return None

    brightest = int(np.nanargmax(flux))
    position = (float(catalog.x_centroid[brightest]),
                float(catalog.y_centroid[brightest]))
    if not np.isfinite(position).all():
        return None

    return position, segments.n_labels


def detect_dao(data, median, std, fwhm_guess=DEFAULT_FWHM_GUESS,
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
    median : float
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
    sources = DAOStarFinder(threshold=nsigma * std, fwhm=fwhm_guess)(data - median)
    if sources is None or len(sources) == 0:
        return None

    brightest = sources[np.argmax(sources["flux"])]
    position = (float(brightest["x_centroid"]), float(brightest["y_centroid"]))

    return position, len(sources)


def detect_peak(data, median, std, fwhm_guess=DEFAULT_FWHM_GUESS,
                nsigma=DEFAULT_NSIGMA):
    """Find the brightest peak of a smoothed frame.

    Last-resort fallback for when the chosen detection method finds nothing: it
    imposes no shape or area requirement, so it returns a position as long as
    anything at all rises above the threshold.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image.
    median : float
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
    smoothed = convolve(data - median, kernel)

    peaks = find_peaks(smoothed, threshold=nsigma * std, n_peaks=1)
    if peaks is None or len(peaks) == 0:
        return None

    return float(peaks["x_peak"][0]), float(peaks["y_peak"][0])


def locate_source(data, fwhm_guess=DEFAULT_FWHM_GUESS, nsigma=DEFAULT_NSIGMA,
                  stats=None, method=DEFAULT_DETECT_METHOD,
                  npixels=DEFAULT_NPIXELS, deblend=True):
    """Find the brightest source in the frame and return a refined centroid.

    Detection uses image segmentation or DAOStarFinder as selected by `method`,
    falling back to the brightest peak of a smoothed copy if that finds nothing.
    The chosen position is then refined with a quadratic centroid fit.

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

    Returns
    -------
    tuple of float
        The (x, y) pixel position of the source.

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

    median, std, _ = stats if stats is not None else frame_stats(data)

    if method == "segment":
        found = detect_segment(data, median, std, fwhm_guess=fwhm_guess,
                               nsigma=nsigma, npixels=npixels, deblend=deblend)
        noun = "segments"
    else:
        found = detect_dao(data, median, std, fwhm_guess=fwhm_guess, nsigma=nsigma)
        noun = "sources"

    if found is not None:
        (xpeak, ypeak), n_found = found
        if n_found > 1:
            print(f"[detect] {n_found} {noun} found; using the brightest at "
                  f"({xpeak:.1f}, {ypeak:.1f})")
    else:
        fallback = detect_peak(data, median, std, fwhm_guess=fwhm_guess, nsigma=nsigma)
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

    # `fit_boxsize` has to be passed explicitly: centroid_sources cuts out a
    # box_size region and then calls centroid_quadratic on it with that keyword
    # left at its default of 5, which fits a 5x5 quadratic to a spot several
    # times wider and lands about a pixel off.
    box = max(5, int(2 * round(fwhm_guess / 2)) + 1)
    xcen, ycen = centroid_sources(data, xpeak, ypeak, box_size=box,
                                  centroid_func=centroid_quadratic,
                                  fit_boxsize=box)

    if not np.isfinite([xcen, ycen]).all():
        print("[detect] quadratic centroid failed; using the detection position")
        return xpeak, ypeak

    return float(xcen[0]), float(ycen[0])


def measure_fwhm(data, position, background, fwhm_guess=DEFAULT_FWHM_GUESS):
    """Measure the source FWHM from a Gaussian fit to its radial profile.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image.
    position : tuple of float
        The (x, y) source centroid.
    background : float
        Background level to subtract before profiling.
    fwhm_guess : float, optional
        Approximate FWHM in pixels; sets the profile extent and is returned as a
        fallback if the fit fails.

    Returns
    -------
    float
        Measured FWHM in pixels.
    """
    radii = np.arange(0.0, 4.0 * fwhm_guess, 1.0)

    try:
        profile = RadialProfile(data - background, position, radii)
        fwhm = float(profile.gaussian_fwhm)
    except Exception as exc:  # noqa: BLE001 - any fit failure falls back to the guess
        print(f"[fwhm] radial profile fit failed ({exc}); using --fwhm-guess instead")
        return fwhm_guess

    if not np.isfinite(fwhm) or fwhm <= 0 or fwhm > 4.0 * fwhm_guess:
        print(f"[fwhm] implausible fitted FWHM ({fwhm:.2f} px); using --fwhm-guess instead")
        return fwhm_guess

    return fwhm


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
        (median, std, n_pixels) of the sigma-clipped background, where
        `n_pixels` is the number of pixels surviving the clip.
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
        (median, std, n_pixels) of the sigma-clipped annulus pixels.
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
    return float(stats.median), float(stats.std), n_pix


def measure_counts(data, position, radius, bkg_median, bkg_std, n_bkg_pix,
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
    bkg_median : float
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
    subtracted = data - bkg_median

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


def make_figure(data, position, aperture, fwhm, bkg_median, out_path=None, annulus=None,
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
    bkg_median : float
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
    cog = CurveOfGrowth(data - bkg_median, position, radii)
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
        description="Aperture photometry of a single fiber output in TIFF frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", metavar="INPUT",
                        help="TIFF frames, directories whose top-level TIFFs are "
                             "measured, or a text file listing either one per line")
    parser.add_argument("--fwhm-guess", type=float, default=DEFAULT_FWHM_GUESS,
                        help="approximate source FWHM in pixels, used for detection")
    parser.add_argument("--k-fwhm", type=float, default=DEFAULT_K_FWHM,
                        help="aperture radius in units of the measured FWHM")
    parser.add_argument("--radius", type=float, default=None,
                        help="explicit aperture radius in pixels (overrides --k-fwhm)")
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
    parser.add_argument("--annulus-bkg", action="store_true",
                        help="estimate the background from a local annulus instead of "
                             "the sigma-clipped whole frame")
    parser.add_argument("--r-in", type=float, default=ANNULUS_IN_FWHM,
                        help="inner annulus radius in units of the measured FWHM")
    parser.add_argument("--r-out", type=float, default=ANNULUS_OUT_FWHM,
                        help="outer annulus radius in units of the measured FWHM")
    parser.add_argument("--gain", type=float, default=GAIN_E_PER_ADU,
                        help="detector gain in e-/ADU")
    parser.add_argument("--read-noise", type=float, default=READ_NOISE_E,
                        help="RMS read noise in e-")
    parser.add_argument("--plane", type=int, default=0,
                        help="page to use from a multi-page TIFF")
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
                         k_fwhm=DEFAULT_K_FWHM, radius=None, nsigma=DEFAULT_NSIGMA,
                         detect_method=DEFAULT_DETECT_METHOD, npixels=DEFAULT_NPIXELS,
                         deblend=True, annulus_bkg=False, r_in=ANNULUS_IN_FWHM,
                         r_out=ANNULUS_OUT_FWHM, gain=GAIN_E_PER_ADU,
                         read_noise_e=READ_NOISE_E, plot=False, verbose=False):
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
        Expected source FWHM in pixels, used to size the detection kernel and as
        the fallback if the profile fit does not converge.
    k_fwhm : float, optional
        Aperture radius in units of the measured FWHM. Ignored if `radius` is given.
    radius : float, optional
        Explicit aperture radius in pixels, overriding the FWHM scaling.
    nsigma : float, optional
        Detection threshold in units of the background scatter.
    detect_method : {'segment', 'dao'}, optional
        Detection method, as for --detect-method.
    npixels : int, optional
        Smallest connected area a segmentation detection may have.
    deblend : bool, optional
        Whether to deblend segments before picking the brightest.
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
        If True, print the same summary the command line prints.

    Returns
    -------
    dict
        The measurement. The `RESULT_COLUMNS` keys - `name`, `x`, `y`,
        `fwhm_pix`, `radius_pix`, `detect_method`, `bkg_method`, `bkg_median`,
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

    stats = frame_stats(data)

    position = locate_source(data, fwhm_guess=fwhm_guess, nsigma=nsigma, stats=stats,
                             method=detect_method, npixels=npixels, deblend=deblend)
    fwhm = measure_fwhm(data, position, stats[0], fwhm_guess=fwhm_guess)
    aperture_radius = radius if radius is not None else k_fwhm * fwhm

    annulus = None
    if annulus_bkg:
        ann_in, ann_out = r_in * fwhm, r_out * fwhm
        bkg_median, bkg_std, n_bkg_pix = estimate_background_annulus(
            data, position, ann_in, ann_out
        )
        annulus = CircularAnnulus(position, r_in=ann_in, r_out=ann_out)
        bkg_method = f"annulus[{ann_in:.1f}-{ann_out:.1f}]"
    else:
        bkg_median, bkg_std, n_bkg_pix = estimate_background_global(data, stats=stats)
        bkg_method = "global"

    counts = measure_counts(
        data, position, aperture_radius, bkg_median, bkg_std, n_bkg_pix,
        gain=gain, read_noise_e=read_noise_e,
    )

    result = {
        "name": name,
        "x": float(position[0]),
        "y": float(position[1]),
        "fwhm_pix": float(fwhm),
        "radius_pix": float(aperture_radius),
        "detect_method": detect_method,
        "bkg_method": bkg_method,
        "bkg_median": float(bkg_median),
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

    if verbose:
        report_measurement(result)

    if plot:
        result["figure"] = make_figure(
            data, position, counts["aperture"], fwhm, bkg_median,
            annulus=annulus, show=False, close=False,
        )

    return result


def report_measurement(result):
    """Print the headline numbers of a measurement to stdout.

    Kept out of `measure_single_image` so that the command line and a verbose notebook
    call print the same thing from one definition.

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
    print(f"background  : {result['bkg_median']:.3f} +/- {result['bkg_std']:.3f} "
          f"counts/px [{result['bkg_method']}, {result['n_bkg_pix']} px]")
    print(f"counts      : {result['net_counts']:.4e} +/- {result['counts_err']:.2e} "
          f"(SNR {result['snr']:.1f})")

    if result["n_saturated"] > 0:
        print(f"WARNING: {result['n_saturated']} saturated pixels "
              f"(>= {SATURATION_ADU} counts) inside the aperture - the measured "
              "counts are a lower limit.")


def measure_images(images, plane=0, plot=False, verbose=False, on_error="raise",
                   **params):
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
        Page to select from multi-page TIFFs. Applies to path inputs only.
    plot : bool, optional
        If True, build a diagnostic figure per frame. The figures are left open
        rather than saved, so a notebook's inline backend renders them when the
        cell finishes; they are not part of the returned table.
    verbose : bool, optional
        If True, print the summary for each frame.
    on_error : {'raise', 'skip'}, optional
        What to do when a frame fails. 'raise', the default, propagates the
        exception, on the grounds that a row silently missing from a notebook
        table is worse than a traceback. 'skip' reports the failure on stderr
        and omits the row, which is usually what a long batch wants.
    **params
        Forwarded to `measure_single_image`, so every measurement option is available
        by keyword: `fwhm_guess`, `k_fwhm`, `radius`, `nsigma`, `detect_method`,
        `npixels`, `deblend`, `annulus_bkg`, `r_in`, `r_out`, `gain`,
        `read_noise_e`.

    Returns
    -------
    pandas.DataFrame
        One row per successfully measured frame, with the columns named in
        `RESULT_COLUMNS`. The `name` column holds the path for path inputs and
        `image_<index>` for array inputs. Empty input, or a batch in which every
        frame failed under `on_error='skip'`, gives an empty DataFrame with
        those columns.

    Raises
    ------
    ValueError
        If `on_error` is not 'raise' or 'skip'.

    Examples
    --------
    >>> df = measure_images(my_arrays)                             # doctest: +SKIP
    >>> df = measure_images(paths, k_fwhm=3.0, annulus_bkg=True)   # doctest: +SKIP
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
    for index, image in enumerate(images):
        is_path = isinstance(image, (str, Path))
        name = str(image) if is_path else f"image_{index}"

        try:
            data = load_image(image, plane=plane) if is_path else image
            rows.append(measure_single_image(data, name=name, plot=plot,
                                             verbose=verbose, **params))
        except Exception as exc:  # noqa: BLE001 - the policy is the caller's
            if on_error == "raise":
                raise
            print(f"ERROR: {name}: {exc.__class__.__name__}: {exc}", file=sys.stderr)

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
        "bkg_median": f"{result['bkg_median']:.4f}",
        "bkg_std": f"{result['bkg_std']:.4f}",
        "net_counts": f"{result['net_counts']:.6e}",
        "counts_err": f"{result['counts_err']:.6e}",
        "snr": f"{result['snr']:.2f}",
        "n_saturated": result["n_saturated"],
        "gain": result["gain"],
        "read_noise_e": result["read_noise_e"],
    }


def measure_frame(image_path, args, show=False):
    """Measure one frame from disk and report the result to stdout.

    The command-line adapter around `measure_single_image`: it reads the file, unpacks
    the parsed options into keyword arguments, and writes the diagnostic figure
    to `PLOT_DIR/<stem>_aperture.png` unless --no-plot was given.

    Parameters
    ----------
    image_path : pathlib.Path
        The TIFF frame to measure.
    args : argparse.Namespace
        Parsed command-line options.
    show : bool, optional
        If True, open an interactive figure window for this frame.

    Returns
    -------
    dict
        A CSV row of the measurement, keyed by the names in `CSV_COLUMNS`.

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
    data = load_image(image_path, plane=args.plane)

    result = measure_single_image(
        data, name=str(image_path), fwhm_guess=args.fwhm_guess, k_fwhm=args.k_fwhm,
        radius=args.radius, nsigma=args.nsigma, detect_method=args.detect_method,
        npixels=args.npixels, deblend=not args.no_deblend,
        annulus_bkg=args.annulus_bkg, r_in=args.r_in, r_out=args.r_out,
        gain=args.gain, read_noise_e=args.read_noise, verbose=True,
    )

    if not args.no_plot:
        figure_path = PLOT_DIR / f"{image_path.stem}_aperture.png"
        make_figure(data, (result["x"], result["y"]), result["aperture"],
                    result["fwhm_pix"], result["bkg_median"], figure_path,
                    annulus=result["annulus"], show=show)
        print(f"figure      : {figure_path}")

    return csv_row(result)


def main(argv=None):
    """Measure every requested frame and write the results.

    A frame that cannot be read or has no detectable source is reported and
    skipped, so one bad file does not abandon a batch. The CSV named by --outfile
    is written once, after all frames are measured; diagnostic figures go to
    `PLOT_DIR`.

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

    if len(frames) > 1:
        print(f"measuring {len(frames)} frames")

    rows = []
    failures = []

    for frame in frames:
        try:
            rows.append(measure_frame(frame, args, show=show))
        except Exception as exc:  # noqa: BLE001 - one bad frame must not end the batch
            failures.append(frame)
            print(f"ERROR: {frame}: {exc.__class__.__name__}: {exc}", file=sys.stderr)

    if rows and not args.no_csv:
        csv_path, n_removed = update_csv(resolve_outfile(args.outfile), rows)
        removed = f" (removed {n_removed} superseded row"
        removed += "s)" if n_removed > 1 else ")"
        print(f"\ncsv         : {csv_path}{removed if n_removed else ''}")

    if len(frames) > 1 or failures:
        print(f"measured {len(rows)}/{len(frames)} frames"
              + (f", {len(failures)} failed" if failures else ""))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
