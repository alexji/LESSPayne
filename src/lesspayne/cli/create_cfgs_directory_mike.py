"""
Creates scripts to run LESSPayne on all spectra in a directory

Assumes the data is MIKE data
"""

import os, sys, glob, time
import numpy as np
import yaml
from copy import deepcopy

# Need to specify output_name, spectrum_fnames, payne_fname, smh_fname
def add_to_dict(name, kws, reduced_path):
    rf = name+"red_multi.fits"
    bf = name+"blue_multi.fits"
    pf = name+"_paynefit.npz"
    sf = name+"_lesspayne.smh"
    kws["output_name"] = name
    kws["spectrum_fnames"] = [f"{reduced_path}/{x}" for x in [rf, bf]]
    kws["payne_fname"] = pf
    kws["smh_fname"] = sf

default_kws = dict(
    output_directory = "./outputs",
    figure_directory = "./figs",
    overwrite_files = True,
    NN_file = "/Users/alexji/lib/LESSPayne/LESSPayne/data/NN_normalized_spectra_float16_fixwave.npz",
    autovelocity = dict(
        template_spectrum_fname = "/Users/alexji/lib/LESSPayne/LESSPayne/data/template_spectra/hd122563.fits",
        wavelength_region_min = 5150,
        wavelength_region_max = 5200,
    ),
    payne = dict(
        initial_velocity = None,
        wmin = 5000,
        wmax = 7000,
        mask_list = [
            [6276,6320],
            [6866,6881],
            [6883,6962],
            [6985,7070],
        ],
        rv_target_wavelengths = [5183,4861],
        initial_parameters = dict(
            Teff = 4500,
            logg = 1.5,
            MH =   -2.0,
            aFe =  0.4,
        ),
        save_figures = False,
    ),
    run_normalization = dict(
        mask_sigma = 0.5,
        mask_smooth = 2,
        mask_thresh = 0.15,
        max_mask_frac = 0.8,
        min_frac_per_knot = 0.05,
        blue_trim = 30,
        red_trim = 30,
        continuum_spline_order = 3,
        continuum_max_iterations = 5,
        save_figure = True,
    ),
    run_eqw_fit = dict(
        max_fwhm = 1.0,
        eqw_linelist_fname = "/Users/alexji/Dropbox/Ant2Cra2/linelists/master_merged_eqw_short.moog",
        extra_eqw_linelist_fname = None,
        mask_sigma = 0.5,
        mask_smooth = 2,
        mask_thresh = 0.15,
        clear_all_existing_fits = True,
        save_figure = False,
        output_suffix = None,
    ),
    run_stellar_parameters = dict(
        method = "rpa_calibration",
        measure_eqw_abundances = True,
        save_figure_eqw = True,
        output_suffix = None,
        manual_Teff = None,
        manual_logg = None,
        manual_vt = None,
        manual_MH = None,
        manual_aFe = 0.4,
    ),
    run_synth_fit = dict(
        max_fwhm = 1.0,
        synthesis_linelist_fname = None,
        extra_synthesis_linelist_fname = None,
        num_iter_all = 2,
        max_iter_each = 3,
        smooth_approx = 0.1,
        smooth_scale = 0.3,
        clear_all_existing_syntheses = False,
        save_figure = True,
        output_suffix = None,
    ),
    run_errors = dict(
        calculate_stellar_params_spectroscopic_uncertainties = False,
        e_Teff = 100,
        e_logg = 0.2,
        e_vt = 0.2,
        e_MH = 0.2,
        save_figure = True,
        output_suffix = None
    )
)

if __name__ == "__main__":
    dirname = sys.argv[1]
    cfgdir = sys.argv[2] or "."
    print("Processing directory", dirname)
    print("Writing cfgs to", cfgdir)
    
    fnames = np.sort(glob.glob(dirname+"/*.fits"))
    blue_fnames = [x for x in fnames if "blue_multi" in x]
    red_fnames = [x for x in fnames if "red_multi" in x]
    if len(blue_fnames) != len(red_fnames):
        print("ERROR", len(blue_fnames), "!=", len(red_fnames))
        print(blue_fnames)
        print(red_fnames)
        exit(1)
    
    for bf, rf in zip(blue_fnames, red_fnames):
        assert bf.replace("blue_multi.fits","") == rf.replace('red_multi.fits',"")
        name = os.path.basename(bf).replace("blue_multi.fits","")
        
        cfgfile = os.path.join(cfgdir, "cfg_"+name+".yaml")
        
        kwds = deepcopy(default_kws)
        add_to_dict(name, kwds, dirname)
        with open(cfgfile, "w") as fp:
            fp.write(yaml.dump(kwds))
