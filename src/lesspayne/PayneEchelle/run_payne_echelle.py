import sys, os
import yaml
from astropy.stats import biweight_scale

from lesspayne.specutils import Spectrum1D
from lesspayne.specutils.utils import fast_find_continuum
from lesspayne.specutils.rv import quick_measure_mike_velocities
from lesspayne.PayneEchelle.spectral_model import DefaultPayneModel, YYLiPayneModel
from lesspayne.PayneEchelle import plotting
from lesspayne.PayneEchelle import utils
from lesspayne.PayneEchelle import fitting

import pickle
import numpy as np

from scipy import interpolate
from scipy import signal
from scipy.stats import norm
import time, os, glob

def read_spectrum(fname):
    specs = Spectrum1D.read(fname)
    waves = np.array([np.median(spec.dispersion) for spec in specs])
    iisort = np.argsort(waves)
    specs = [specs[ix] for ix in iisort]

    Npix = len(specs[0].dispersion)
    Nord = len(specs)

    wavelength = np.zeros((Nord, Npix))
    spectrum = np.zeros((Nord, Npix))
    spectrum_err = np.zeros((Nord, Npix))
    for i,spec in enumerate(specs):
        assert len(spec.dispersion) == Npix
        wavelength[i] = spec.dispersion
        spectrum[i] = spec.flux
        spectrum_err[i] = spec.ivar**-0.5
    return wavelength, spectrum, spectrum_err
def get_quick_continuum(wavelength, spectrum):
    cont = np.zeros_like(wavelength)
    for i in range(cont.shape[0]):
        cont[i] = fast_find_continuum(spectrum[i])
    return cont

def run_payne_echelle(cfg):
    start = time.time()
    
    name = cfg["output_name"]
    NNpath = cfg["NN_file"]
    NNtype = cfg["NN_type"]
    assert NNtype in ["default", "yyli"], NNtype
    outdir = cfg["output_directory"]
    figdir = cfg["figure_directory"]
    if not os.path.exists(outdir): os.makedirs(outdir)
    if not os.path.exists(figdir): os.makedirs(figdir)
    outfname_bestfit = os.path.join(outdir, cfg["payne_fname"])
    print("Saving to output directory:",outdir,outfname_bestfit)
    print("Saving figures to output directory:",figdir)
    
    fnames = cfg["spectrum_fnames"]
    specfname = fnames[0]
    
    pcfg = cfg["payne"]
    print(pcfg)
    wmin, wmax = pcfg["wmin"], pcfg["wmax"]
    mask_list = pcfg["mask_list"]
    rv_target_wavelengths = pcfg["rv_target_wavelengths"]
    
    if NNtype == "default":
        Teff0, logg0, MH0, aFe0 = pcfg["initial_parameters"]["Teff"], pcfg["initial_parameters"]["logg"], \
            pcfg["initial_parameters"]["MH"], pcfg["initial_parameters"]["aFe"]
        initial_stellar_labels = [Teff0, logg0, MH0, aFe0]
    elif NNtype == "yyli":
        Teff0, logg0, MH0, aFe0 = pcfg["initial_parameters"]["Teff"], pcfg["initial_parameters"]["logg"], \
            pcfg["initial_parameters"]["MH"], pcfg["initial_parameters"]["aFe"]
        vt0 = pcfg["initial_parameters"].get("vt", 1.5)
        CFe0 = pcfg["initial_parameters"].get("CFe", 0.0)
        MgFe0 = pcfg["initial_parameters"].get("MgFe", aFe0)
        CaFe0 = pcfg["initial_parameters"].get("CaFe", aFe0)
        TiFe0 = pcfg["initial_parameters"].get("TiFe", aFe0)
        initial_stellar_labels = [Teff0, logg0, vt0, MH0, CFe0, MgFe0, CaFe0, TiFe0]
    
    poly_coeff_min = pcfg.get("poly_coeff_min",-1000)
    poly_coeff_max = pcfg.get("poly_coeff_max",1000)
    polynomial_order = pcfg.get("polynomial_order",6)

    ## Get initial RV
    if pcfg["initial_velocity"] is None:
        vcfg = cfg["autovelocity"]
        print(f"Automatic velocity with template {vcfg['template_spectrum_fname']}")
        template = Spectrum1D.read(vcfg["template_spectrum_fname"])
        rv_wave1, rv_wave2 = float(vcfg["wavelength_region_min"]), float(vcfg["wavelength_region_max"])
        rv0, vhelcorr = quick_measure_mike_velocities(specfname, template_spectrum=template,
                                                      wmin=rv_wave1, wmax=rv_wave2)
        print(f"Automatic velocity from {rv_wave1:.0f}-{rv_wave2:.0f}: {rv0:.1f} (+ {vhelcorr:.1f} for vhel)")
    else:
        rv0 = float(pcfg["initial_velocity"])
        print(f"Initial RV {rv0:.1f}")
    
    ## Preprocess the spectrum
    wavelength, spectrum, spectrum_err = read_spectrum(specfname)
    wavelength_blaze = wavelength.copy() # blaze and spec have same
    spectrum_blaze = get_quick_continuum(wavelength, spectrum)
    wavelength, spectrum, spectrum_err = utils.cut_wavelength(wavelength, spectrum, spectrum_err, wmin, wmax)
    wavelength_blaze, spectrum_blaze, spectrum_blaze_err = utils.cut_wavelength(
        wavelength_blaze, spectrum_blaze, spectrum_blaze.copy(), wmin, wmax)
    num_order, num_pixel = wavelength.shape
    spectrum = np.abs(spectrum)
    spectrum_err[(spectrum_err==0) | np.isnan(spectrum_err)] = 999.
    spectrum_blaze = np.abs(spectrum_blaze)    
    spectrum_blaze[spectrum_blaze == 0] = 1.
    
    # rescale the spectra by its median so it has a more reasonable y-range
    spectrum, spectrum_err = utils.scale_spectrum_by_median(spectrum, spectrum_err.copy())
    spectrum_blaze = spectrum_blaze/np.nanmedian(spectrum_blaze, axis=1)[:,np.newaxis]
    # some orders are all zeros, remove these
    bad_orders = np.all(np.isnan(spectrum), axis=1)
    
    if bad_orders.sum() > 0:
        print("Removing {} bad orders".format(bad_orders.sum()))
    wavelength, spectrum, spectrum_err, spectrum_blaze = \
        wavelength[~bad_orders], spectrum[~bad_orders], spectrum_err[~bad_orders], spectrum_blaze[~bad_orders]
    
    # eliminate zero values in the blaze function to avoid dividing with zeros
    # the truncation is quite aggresive, can be improved if needed
    ind_valid = np.min(np.abs(spectrum_blaze), axis=0) != 0
    spectrum_blaze = spectrum_blaze[:,ind_valid]
    wavelength_blaze = wavelength_blaze[:,ind_valid]
    
    # match the wavelength (blaze -> spectrum)
    spectrum_blaze, wavelength_blaze = utils.match_blaze_to_spectrum(wavelength, spectrum, wavelength_blaze, spectrum_blaze)
    
    norder, npix = wavelength.shape
    
    spectrum_err = utils.mask_wavelength_regions(wavelength, spectrum_err, mask_list)
    
    ## Sigma clip away outliers in the noise, these are bad pixels. We will increase their spectrum_err to very large
    mask = (~np.isfinite(spectrum)) | (~np.isfinite(spectrum_err)) | (~np.isfinite(spectrum_blaze))
    for j in range(spectrum_err.shape[0]):
        err_cont = fast_find_continuum(spectrum_err[j])
        err_norm = spectrum_err[j]/err_cont - 1
        err_errs = biweight_scale(err_norm)
        mask[j, np.abs(err_norm/err_errs) > 10] = True
    
    # Normalize the errors to cap out at 999
    spectrum_err[spectrum_err > 999] = 999
    

    ## Set up RV Array
    RV_array = np.array([rv0/100.])
    found = False
    for target_wavelength in rv_target_wavelengths:
        for iorder in range(num_order):
            if (wavelength[iorder,0] < target_wavelength) and (wavelength[iorder,-1] > target_wavelength):
                found = True
                break
        if found: break
    else:
        raise RuntimeError(f"{specfname} does not have any of the target wavelengths: {rv_target_wavelengths}")
    print(f"Median RV order wavelength: {np.median(wavelength[iorder]):.1f}")

    start2 = time.time()
    if NNtype == "default":
        print(f"Running with DefaultPayneModel ({NNpath})")
        model = DefaultPayneModel.load(NNpath, num_order=norder)
        errors_payne = utils.read_default_model_mask(wavelength_payne=model.wavelength_payne)
        model = DefaultPayneModel.load(NNpath, num_order=norder, errors_payne=errors_payne,
                                       polynomial_order=polynomial_order)
    elif NNtype == "yyli":
        print(f"Running with YYLiPayneModel ({NNpath})")
        model = YYLiPayneModel.load(NNpath, num_order=norder)
        errors_payne = utils.read_default_model_mask(wavelength_payne=model.wavelength_payne)
        model = YYLiPayneModel.load(NNpath, num_order=norder, errors_payne=errors_payne,
                                    polynomial_order=polynomial_order)

    print("starting fit")
    out = fitting.fit_global(spectrum, spectrum_err, spectrum_blaze, wavelength,
                             model, initial_stellar_parameters=initial_stellar_labels,
                             RV_array = RV_array, order_choice=[iorder],
                             poly_coeff_min=poly_coeff_min, poly_coeff_max=poly_coeff_max,
                             get_perr=True)
    popt_best, model_spec_best, chi_square, perr_best = out
    print(f"PayneEchelle Fit Took {time.time()-start2:.1f}")
    popt_print = model.transform_coefficients(popt_best)
    perr_print = np.abs(model.transform_coefficients(popt_best + perr_best) - popt_print)
    if NNtype == "default":
        print("[Teff [K], logg, Fe/H, Alpha/Fe] = ",\
              int(popt_print[0]*1.)/1.,\
              int(popt_print[1]*100.)/100.,\
              int(popt_print[2]*100.)/100.,\
              int(popt_print[3]*100.)/100.,\
              )
        print("vbroad [km/s] = ", int(popt_print[-2]*10.)/10.)
        print("RV [km/s] = ", int(popt_print[-1]*10.)/10.)
    elif NNtype == "yyli":
        print("[Teff [K], logg, vt, Fe/H, CFe, MgFe, CaFe, TiFe] = ",\
              int(popt_print[0]*1.)/1.,\
              int(popt_print[1]*100.)/100.,\
              int(popt_print[2]*100.)/100.,\
              int(popt_print[3]*100.)/100.,\
              int(popt_print[4]*100.)/100.,\
              int(popt_print[5]*100.)/100.,\
              int(popt_print[6]*100.)/100.,\
              int(popt_print[7]*100.)/100.,\
              )
        print("vbroad [km/s] = ", int(popt_print[-2]*10.)/10.)
        print("RV [km/s] = ", int(popt_print[-1]*10.)/10.)
    print("Chi square = ", chi_square)
    
    np.savez(outfname_bestfit,
             popt_best=popt_best,
             perr_best=perr_best,
             popt_print=popt_print,
             perr_print=perr_print,
             model_spec_best=model_spec_best,
             chi_square=chi_square,
             errors_payne=errors_payne,
             wavelength=wavelength,
             spectrum=spectrum,
             spectrum_err=spectrum_err,
             initial_stellar_labels=initial_stellar_labels,
             NNtype=NNtype, NNpath=NNpath)
    
    if pcfg["save_figures"]:
        plotting.save_figures_multipdf(name, wavelength, spectrum, spectrum_err, model_spec_best,
                                       errors_payne=errors_payne, popt_best=popt_best, model=model,
                                       outdir=figdir)
    
    print(f"TOTAL TIME: {time.time()-start:.1f}")
    
