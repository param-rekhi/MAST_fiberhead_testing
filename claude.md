# Project: MAST fiberhead test suite

## What this does
Analyzes TIFF or FITS images of the fiber output(s) from the MAST fiber coupled telescopes to measure the light output from each fiber.

## Project vision
This will grow into a small toolkit to conduct photometry on images of the fiberhead.
- There will be separate scripts for aperture and psf photometry, as well as for single and multiple sources (i.e. fiber outputs) per image.
- Eventually these may share common helper code (e.g. reading FITS files, source detection, output formatting)

## Using the code
- The scripts are also importable as libraries. `aperture_photometry_single.py`
  exposes `measure_single_image` (one in-memory frame -> dict of numbers) and
  `measure_images` (a list of arrays or paths -> pandas DataFrame) for notebook
  use; see the "Use from a notebook" section of its module docstring.

## Data and results
- Raw data lives in subdirectories in `data/`
- Any saved outputs go in `results/` when applicable
- Plots are currently only for sanity check and are saved temporarily in `plots/`. This folder is not tracked by git.

## Tools to use
- Python, with photutils, astropy, numpy, matplotlib, tifffile

## Rules
- Ask before overwriting or deleting any files in `data/`
- Save every accepted plan to `.claude-plans/plan_<relevant_title>.md`, with a
  timestamp at the top of the document
- Prioritize proper documentation over inline comments — e.g. clear docstrings 
  (purpose, parameters, returns) for every function/script, rather than 
  line-by-line comments
