r"""°°°
This is a quick rundown of how to run LESSPayne by writing python code.

The script LESSPayne/LESSPayne/cli/run.py allows you to do this from the command line
°°°"""
# |%%--%%| <zHesrNqysT|PZ4BzZT3de>

import yaml

# Imports for automatic running
from LESSPayne.PayneEchelle.run_payne_echelle import run_payne_echelle
from LESSPayne.autosmh.run_normalization import run_normalization
from LESSPayne.autosmh.run_eqw_fit import run_eqw_fit
from LESSPayne.autosmh.run_stellar_parameters import run_stellar_parameters
from LESSPayne.autosmh.run_synth_fit import run_synth_fit

## Restart and Run All to get the total runtime to fit this spectrum
import time

startall = time.time()

# |%%--%%| <PZ4BzZT3de|GZsbqx8cBx>

## This is an example configuration file for LESSPayne, parsed with yaml

## What is generally recommended is to put this in a file and load it
## But for this tutorial we'll put it in a string
cfg_string = """
# These directories are automatically created
output_directory: ./outputs
figure_directory: ./figs
# This is used to set filenames, make it unique for each object
output_name: tutorial-hd122563
# TODO Specify the spectrum filenames
# If there's only one file, just do one row, but don't remove the dash (-) or it may break
# Make sure the first one contains your autovelocity wavelength range
spectrum_fnames:
- ./hd122563red_multi.fits
- ./hd122563blue_multi.fits
# Note: right now most code will ignore this flag and always overwrite
# To be updated later
overwrite_files: True
payne_fname: tutorial-hd122563_paynefit.npz
smh_fname: tutorial-hd122563_lesspayne.smh
# TODO update this to point to the actual file
NN_file: ./NN_normalized_spectra_float16_fixwave.npz
# parameters for automatic velocity finding
autovelocity:
  # TODO update this to point to the actual file
  template_spectrum_fname: hd122563.fits
  wavelength_region_min: 5150
  wavelength_region_max: 5200
# parameters for automatic payne fit
payne:
  # automatically find the velocity from xcor
  initial_velocity: null
  # restrict the wavelength range
  # the full allowed range is 3500-10000A
  wmin: 4800
  wmax: 7000
  mask_list:
  - - 6276
    - 6320
  - - 6866
    - 6881
  - - 6883
    - 6962
  - - 6985
    - 7070
  rv_target_wavelengths:
  - 5183
  - 4861
  initial_parameters:
    Teff: 4500
    logg: 1.5
    MH:   -2.0
    aFe:  0.4
  save_figures: False
run_normalization:
  # threshold for deciding whether to mask a line in the Payne model
  mask_sigma: 4.0
  # parameters for growing the mask size a bit
  mask_smooth: 2
  mask_thresh: 0.15
  # maximum fraction of masked pixels per order; increases mask_sigma by 0.5 until reached
  max_mask_frac: 0.8
  # minimum fraction of unmasked pixels/knot; increases knot_spacing by 2 until reached
  min_frac_per_knot: 0.05
  # blue and red trim in pixels
  blue_trim: 30
  red_trim: 30
  # normalization kwargs
  continuum_spline_order: 3
  continuum_max_iterations: 5
  # Flag to output a normalization figure
  save_figure: True
run_eqw_fit:
  max_fwhm: 1.0
  # TODO update this to point to the right filename
  eqw_linelist_fname: rpa2k.eqwshort-moog.txt
  extra_eqw_linelist_fname: null
  # threshold for deciding whether to mask a line in the Payne model
  mask_sigma: 2.0
  # parameters for growing the mask size a bit
  mask_smooth: 2
  mask_thresh: 0.15
  clear_all_existing_fits: True
  save_figure: False
run_stellar_parameters:
  method: rpa_calibration
  manual_Teff: null
  manual_logg: null
  manual_vt: null
  manual_MH: null
  manual_aFe: 0.4
  solve_vt: True
  solve_vt_settings:
      num_acceptable_iter: 8
      
  measure_eqw_abundances: True
  save_figure_eqw: True
run_synth_fit:
  max_fwhm: 1.0
  # Note: check the paths in the master list to make sure they're right!
  # TODO update this list
  synthesis_linelist_fname: quick.synth-master-v1.txt
  #synthesis_linelist_fname: null
  extra_synthesis_linelist_fname: null
  # number of times to iterate through fitting all syntheses
  # the actual number of iterations is 2x this: the first set puts a smoothing prior,
  # while the second set uses known abundances
  # the runtime is essentially proportional to this x the number of synthesis lines
  num_iter_all: 2
  # each optimization takes a few times to minimize the chi2
  max_iter_each: 3
  # parameters for initial prior on smoothing in Angstroms
  smooth_approx: 0.1
  smooth_scale : 0.3
  # Set O, Si, Ca, Ti to [Mg/Fe]: currently forced to be true, capped at +/- 0.4
  # use_MgFe_for_aFe: True
  # Set neutron-capture elements to r-process value: currently forced to be true
  # use_EuFe_for_rproc: True
  clear_all_existing_syntheses: False
  save_figure: True
# Not used yet
run_errors:
  calculate_stellar_params_spectroscopic_uncertainties: False
  e_Teff: 50
  e_logg: 0.1
  e_vt: 0.1
  e_MH: 0.1
  save_figure: True
# Not used yet
run_plot:
"""

# |%%--%%| <GZsbqx8cBx|NhvbnIqvSt>

cfg = yaml.load(cfg_string, yaml.FullLoader)

# |%%--%%| <NhvbnIqvSt|lk0Gkg348T>

## This parses into a dictionary as you can see
cfg

# |%%--%%| <lk0Gkg348T|cL2MeuNXbd>

# run PayneEchelle. Creates outputs/tutorial-hd122563_paynefit.npz and some figures
run_payne_echelle(cfg)

# |%%--%%| <cL2MeuNXbd|QbupN8oVZo>

# create SMHR file with normalizations.
# Uses PayneEchelle model to find pixels to mask
# Creates ./outputs/tutorial-hd122563_lesspayne.smh, overwrites existing files
# This file can be opened in a GUI:
#  go into LESSPayne/LESSPayne/smh/gui and run `pythonw __main__.py`
# Also creates a figure in ./figs/ showing all the normalizations
run_normalization(cfg)

# |%%--%%| <QbupN8oVZo|xVBBR9UEzr>

# fit EQWs
# Reads ./outputs/tutorial-hd122563_lesspayne.smh and overwrites this file when it completes
run_eqw_fit(cfg)

# |%%--%%| <xVBBR9UEzr|3QAHvraXOt>

# put in stellar parameters and (optionally) measure EQW abundances
# overwrites the SMH file
run_stellar_parameters(cfg)

# |%%--%%| <3QAHvraXOt|YTH6vljCBh>

# import and iteratively fit syntheses
# overwrites the SMH file
run_synth_fit(cfg)

# |%%--%%| <YTH6vljCBh|Nb15nLxvLl>

print(f"If restart/runall, this took {time.time()-startall:.1f} seconds")

# |%%--%%| <Nb15nLxvLl|y16gvgfm4j>
r"""°°°
At the end of this is an SMHR file in outputs/tutorial-hd122563_lesspayne.smh.

In general it is wise to do this in stages:
* run PayneEchelle (2-5 minutes), and optionally inspect the automatic fit to make sure it's OK
* run normalization. Once this is done, inspect the SMH file and make any manual modifications to normalization.
* run eqw fits. When this is done, inspect the SMH file and make any manual changes to the EQW fits.
* run stellar parameters. Right now the only options are to manually specify the parameters, or to use a calibrated version of the PayneEchelle parameters (calibrated to RPA Duplicates).
* once you are happy with EQW fits and stellar parameters, run synthesis fits. If running a lot of syntheses, this is the most time consuming part as it needs to iterate multiple times.

After the normalizations, you can always edit the SMH file manually to fix anything that didn't automatically work.
°°°"""

