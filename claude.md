# Project: MAST fiberhead test suite

## What this does
Analyzes TIFF or FITS images of the fiber output(s) from the MAST fiber coupled telescopes to measure the light output from each fiber.

## Project vision
This will grow into a small toolkit to conduct photometry on images of the fiberhead.
- There will be separate scripts for aperture and psf photometry, as well as for single and multiple sources (i.e. fiber outputs) per image.
- Eventually these may share common helper code (e.g. reading FITS files, source detection, output formatting)

## Data and results
- Raw data lives in subdirectories in `data/`
- Any saved outputs go in `results/` when applicable

## Tools to use
- Python, with photutils, astropy, numpy, matplotlib, tifffile

## Rules
- Ask before overwriting or deleting any files in `data/`
- Prioritize proper documentation over inline comments — e.g. clear docstrings 
  (purpose, parameters, returns) for every function/script, rather than 
  line-by-line comments
