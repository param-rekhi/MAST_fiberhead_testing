#!/usr/bin/env python3
"""Aperture photometry for a single fiber output in MAST fiberhead images.

Locates the one bright spot in a TIFF frame, measures its FWHM, subtracts a
background estimated either globally or from a local annulus, and integrates the
counts inside a circular aperture whose radius is scaled to the measured FWHM.

Outputs a two-panel diagnostic figure (annotated cutout + curve of growth), a row
in a CSV table of measurements, and a summary printed to stdout.

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
from astropy.visualization import ZScaleInterval
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)
from photutils.centroids import centroid_quadratic
from photutils.detection import DAOStarFinder, find_peaks
from photutils.profiles import CurveOfGrowth, RadialProfile
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

TIFF_SUFFIXES = {".tif", ".tiff"}

CSV_NAME = "aperture_photometry_single.csv"
CSV_COLUMNS = [
    "filename",
    "x",
    "y",
    "fwhm_pix",
    "radius_pix",
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

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image.

    Returns
    -------
    tuple
        (median, std, n_pixels) of the sigma-clipped frame, where `n_pixels` is
        the number of pixels surviving the clip.
    """
    clipped = SigmaClip(sigma=SIGMA_CLIP)(data.ravel(), masked=True)
    return float(np.ma.median(clipped)), float(np.ma.std(clipped)), int(clipped.count())


def locate_source(data, fwhm_guess=DEFAULT_FWHM_GUESS, nsigma=DEFAULT_NSIGMA,
                  stats=None):
    """Find the brightest source in the frame and return a refined centroid.

    Detection uses DAOStarFinder on the median-subtracted frame, falling back to
    the brightest peak of a smoothed copy if nothing is detected. The chosen
    candidate is refined with a quadratic centroid fit.

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

    Returns
    -------
    tuple of float
        The (x, y) pixel position of the source.

    Raises
    ------
    RuntimeError
        If no source is found by either method.
    """
    median, std, _ = stats if stats is not None else frame_stats(data)
    subtracted = data - median

    finder = DAOStarFinder(threshold=nsigma * std, fwhm=fwhm_guess)
    sources = finder(subtracted)

    if sources is not None and len(sources) > 0:
        brightest = sources[np.argmax(sources["flux"])]
        xpeak, ypeak = float(brightest["xcentroid"]), float(brightest["ycentroid"])
        if len(sources) > 1:
            print(
                f"[detect] {len(sources)} sources found; using the brightest "
                f"at ({xpeak:.1f}, {ypeak:.1f})"
            )
    else:
        kernel = Gaussian2DKernel(x_stddev=fwhm_guess / 2.355)
        smoothed = convolve(subtracted, kernel)
        peaks = find_peaks(smoothed, threshold=nsigma * std, npeaks=1)
        if peaks is None or len(peaks) == 0:
            msg = (
                "no source detected: DAOStarFinder and peak search both came up "
                f"empty at a {nsigma:g}-sigma threshold. Try lowering --nsigma or "
                "adjusting --fwhm-guess."
            )
            raise RuntimeError(msg)
        xpeak, ypeak = float(peaks["x_peak"][0]), float(peaks["y_peak"][0])
        print("[detect] DAOStarFinder found nothing; fell back to a smoothed peak search")

    box = max(5, int(2 * round(fwhm_guess / 2)) + 1)
    xcen, ycen = centroid_quadratic(data, xpeak=xpeak, ypeak=ypeak, fit_boxsize=box)

    if not np.isfinite([xcen, ycen]).all():
        print("[detect] quadratic centroid failed; using the detection position")
        return xpeak, ypeak

    return float(xcen), float(ycen)


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


def make_figure(data, position, aperture, fwhm, bkg_median, out_path, annulus=None,
                show=True):
    """Save (and optionally display) the cutout and curve-of-growth diagnostic.

    The cutout axes are shifted so that the source centroid sits at (0, 0), i.e.
    the tick labels give the offset from the source in pixels.

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
    out_path : pathlib.Path
        Destination for the PNG.
    annulus : photutils.aperture.CircularAnnulus, optional
        Background annulus to overlay, when local background estimation was used.
    show : bool, optional
        If True, open an interactive figure window.

    Returns
    -------
    pathlib.Path
        The path the figure was written to.
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

    vmin, vmax = ZScaleInterval().get_limits(cutout.data[np.isfinite(cutout.data)])
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)

    if show:
        plt.show()
    plt.close(fig)

    return out_path


def update_csv(csv_path, rows):
    """Write measurements to the results CSV, superseding any earlier ones.

    Existing rows referring to the same images are dropped and the new
    measurements appended at the bottom, so the table holds exactly one current
    result per file in the order it was last measured. Filenames are compared as
    resolved paths, so different spellings of the same file still match. The
    table is rewritten via a temporary file and an atomic replace, so an
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
        (path, n_replaced) where `n_replaced` is the number of superseded rows.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    def key(name):
        """Return a comparable form of a filename, resolving it where possible."""
        try:
            return str(Path(name).resolve())
        except OSError:
            return str(name)

    new_keys = {key(row["filename"]) for row in rows}
    kept = []
    n_replaced = 0

    if csv_path.exists():
        with csv_path.open(newline="") as handle:
            for existing in csv.DictReader(handle):
                if key(existing.get("filename", "")) in new_keys:
                    n_replaced += 1
                else:
                    kept.append(existing)

    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, restval="",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows(rows)
    tmp_path.replace(csv_path)

    return csv_path, n_replaced


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
    parser.add_argument("--outdir", type=Path, default=Path("results"),
                        help="directory for the figure and CSV")
    parser.add_argument("--no-plot", action="store_true",
                        help="do not open the interactive figure window (the PNG is "
                             "still written to --outdir; windows are suppressed "
                             "automatically when measuring more than one frame)")
    parser.add_argument("--no-figure", action="store_true",
                        help="skip the diagnostic figure entirely, saving ~0.3 s "
                             "per frame")
    parser.add_argument("--no-csv", action="store_true",
                        help="do not write results to the CSV table")

    return parser.parse_args(argv)


def measure_frame(image_path, args, show=False):
    """Measure one frame and report the result to stdout.

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
    """
    data = load_image(image_path, plane=args.plane)
    stats = frame_stats(data)

    position = locate_source(data, fwhm_guess=args.fwhm_guess, nsigma=args.nsigma,
                             stats=stats)
    fwhm = measure_fwhm(data, position, stats[0], fwhm_guess=args.fwhm_guess)
    radius = args.radius if args.radius is not None else args.k_fwhm * fwhm

    annulus = None
    if args.annulus_bkg:
        r_in, r_out = args.r_in * fwhm, args.r_out * fwhm
        bkg_median, bkg_std, n_bkg_pix = estimate_background_annulus(
            data, position, r_in, r_out
        )
        annulus = CircularAnnulus(position, r_in=r_in, r_out=r_out)
        bkg_method = f"annulus[{r_in:.1f}-{r_out:.1f}]"
    else:
        bkg_median, bkg_std, n_bkg_pix = estimate_background_global(data, stats=stats)
        bkg_method = "global"

    result = measure_counts(
        data, position, radius, bkg_median, bkg_std, n_bkg_pix,
        gain=args.gain, read_noise_e=args.read_noise,
    )

    print(f"\nfile        : {image_path}")
    print(f"centroid    : ({position[0]:.2f}, {position[1]:.2f}) px")
    print(f"FWHM        : {fwhm:.2f} px")
    print(f"aperture    : r = {radius:.2f} px  (area {result['area']:.1f} px^2)")
    print(f"background  : {bkg_median:.3f} +/- {bkg_std:.3f} counts/px "
          f"[{bkg_method}, {n_bkg_pix} px]")
    print(f"counts      : {result['net_counts']:.4e} +/- {result['counts_err']:.2e} "
          f"(SNR {result['snr']:.1f})")

    if result["n_saturated"] > 0:
        print(f"WARNING: {result['n_saturated']} saturated pixels (>= {SATURATION_ADU} "
              "counts) inside the aperture - the measured counts are a lower limit.")

    if not args.no_figure:
        figure_path = args.outdir / f"{image_path.stem}_aperture.png"
        make_figure(data, position, result["aperture"], fwhm, bkg_median, figure_path,
                    annulus=annulus, show=show)
        print(f"figure      : {figure_path}")

    return {
        "filename": str(image_path),
        "x": f"{position[0]:.3f}",
        "y": f"{position[1]:.3f}",
        "fwhm_pix": f"{fwhm:.3f}",
        "radius_pix": f"{radius:.3f}",
        "bkg_method": bkg_method,
        "bkg_median": f"{bkg_median:.4f}",
        "bkg_std": f"{bkg_std:.4f}",
        "net_counts": f"{result['net_counts']:.6e}",
        "counts_err": f"{result['counts_err']:.6e}",
        "snr": f"{result['snr']:.2f}",
        "n_saturated": result["n_saturated"],
        "gain": args.gain,
        "read_noise_e": args.read_noise,
    }


def main(argv=None):
    """Measure every requested frame and write the results.

    A frame that cannot be read or has no detectable source is reported and
    skipped, so one bad file does not abandon a batch. The CSV is written once,
    after all frames are measured.

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

    # An interactive window per frame is unusable in a batch, and a headless
    # backend is needed as soon as nothing is displayed at all.
    show = not args.no_plot and not args.no_figure and len(frames) == 1
    if not show:
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
        csv_path, n_replaced = update_csv(args.outdir / CSV_NAME, rows)
        superseded = f" (replaced {n_replaced} earlier row"
        superseded += "s)" if n_replaced > 1 else ")"
        print(f"\ncsv         : {csv_path}{superseded if n_replaced else ''}")

    if len(frames) > 1 or failures:
        print(f"measured {len(rows)}/{len(frames)} frames"
              + (f", {len(failures)} failed" if failures else ""))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
