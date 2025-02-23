# a few low-level functions that are used throughout
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import os
from scipy import interpolate
from .read_spectrum import read_carpy_fits
from . import spectral_model
import pickle

import pkg_resources
#DATA_PATH = pkg_resources.resource_filename('LESSPayne', 'data/')

#=======================================================================================================================

def vac2air(lamvac):
    """
    http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    Morton 2000
    """
    s2 = (1e4/lamvac)**2
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    return lamvac/n

def read_in_neural_network():

    '''
    read in the weights and biases parameterizing a particular neural network.
    '''

    path = pkg_resources.resource_filename('LESSPayne', 'data/NN_normalized_spectra_float16_fixwave.npz')
    NN_coeffs, wavelength_payne = read_in_nn_path(path)
    #wavelength_payne = vac2air(wavelength_payne)
    return NN_coeffs, wavelength_payne

def read_in_nn_path(path):
    """
    Read in NN from a specified path
    """
    tmp = np.load(path)
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    wavelength_payne = tmp["wavelength_payne"]
    NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    tmp.close()
    return NN_coeffs, wavelength_payne

def read_default_model_mask(wavelength_payne=None):
    if wavelength_payne is None:
        NN_coeffs, wavelength_payne = read_in_neural_network()
    
    errors_payne = np.zeros_like(wavelength_payne)
    theory_mask_path = pkg_resources.resource_filename('LESSPayne', 'PayneEchelle/other_data/theory_mask.txt')
    theory_mask = np.loadtxt(theory_mask_path)
    for wmin, wmax in theory_mask:
        assert wmin < wmax, (wmin, wmax)
        errors_payne[(wavelength_payne >= wmin) & (wavelength_payne <= wmax)] = 999.
    return errors_payne

def read_gaia_eso_benchmark_mask(Nmask=11, thresh=0.8):
    """
    Constructed by Allen Marquez in Summer 2020
    
    Reads other_data/gaia_eso_benchmark_mask_info.pkl, which is an
    Nthresh x Npixel array containing integers, which is the number of stars out of 22 where a pixel is masked.
    
    There are 22 stars from GES, and different threshold cuts for pixels to mask.
    
    Nmask is the number of stars needed to have a pixel masked to mask it
    thresh is selected from 
    thresholds = [0.05, .10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    Larger thresholds mask fewer pixels.
    
    The default values performed okay on the RPA (metal-poor giants)
    """
    thresholds = [0.05, .10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]    
    assert thresh in thresholds, thresholds
    thresh_ix = thresholds.index(thresh)
    
    ## NOTE: this doesn't exist right now
    path = pkg_resources.resource_filename('LESSPayne', 'data/gaia_eso_benchmark_mask_info.pkl')
    with open(path, "rb") as fp:
        all_masks = pickle.load(fp)
    assert len(thresholds) == all_masks.shape[0]
    
    NN_coeffs, wavelength_payne = read_in_neural_network()
    errors_payne = np.zeros_like(wavelength_payne)
    assert len(errors_payne) == all_masks.shape[1]
    
    mask = all_masks[thresh_ix,:] >= Nmask
    errors_payne[mask] = 999.
    return errors_payne

#--------------------------------------------------------------------------------------------------------------------------

def read_in_example():

    '''
    read in a default spectrum to be fitted.
    '''

    path = pkg_resources.resource_filename('LESSPayne', 'data/spectra/2M06062375-0639010_red_multi.fits')
    wavelength, spectrum, spectrum_err = read_carpy_fits(path)
    return wavelength, spectrum, spectrum_err


#--------------------------------------------------------------------------------------------------------------------------

def read_in_blaze_spectrum():

    '''
    read in a default hot star spectrum to determine telluric features and blaze function.
    '''

    path = pkg_resources.resource_filename('LESSPayne', 'data/spectra/Hot_Star_HR718.fits')
    wavelength_blaze, spectrum_blaze, spectrum_err_blaze = read_carpy_fits(path)
    return wavelength_blaze, spectrum_blaze, spectrum_err_blaze


#--------------------------------------------------------------------------------------------------------------------------

def doppler_shift(wavelength, flux, dv):

    '''
    dv is in km/s
    positive dv means the object is moving away.
    '''

    c = 2.99792458e5 # km/s
    doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c))
    new_wavelength = wavelength * doppler_factor
    new_flux = np.interp(new_wavelength, wavelength, flux)
    return new_flux


#--------------------------------------------------------------------------------------------------------------------------

def match_blaze_to_spectrum(wavelength, spectrum, wavelength_blaze, spectrum_blaze):

    '''
    match wavelength of the blaze spectrum to the wavelength of the fitting spectrum
    '''

    for i in range(wavelength.shape[0]):
        if wavelength_blaze[i,0] > wavelength[i,0]:
            wavelength_blaze[i,0] = wavelength[i,0]
        if wavelength_blaze[i,-1] < wavelength[i,-1]:
            wavelength_blaze[i,-1] = wavelength[i,-1]

    spectrum_interp = np.zeros(wavelength.shape)
    for i in range(wavelength.shape[0]):
        f_blaze = interpolate.interp1d(wavelength_blaze[i,:], spectrum_blaze[i,:])
        spectrum_interp[i,:] = f_blaze(wavelength[i,:])

    return spectrum_interp, wavelength


#------------------------------------------------------------------------------------------

def mask_telluric_region(spectrum_err, spectrum_blaze,
                         smooth_length=30, threshold=0.9):

    '''
    mask out the telluric region by setting infinite errors
    '''

    for j in range(spectrum_blaze.shape[0]):
        for i in range(spectrum_blaze[j,:].size-smooth_length):
            if np.min(spectrum_blaze[j,i:i+smooth_length]) \
                    < threshold*np.max(spectrum_blaze[j,i:i+smooth_length]):
                spectrum_err[j,i:i+smooth_length] = 999.
    return spectrum_err

#------------------------------------------------------------------------------------------

def cut_wavelength(wavelength, spectrum, spectrum_err, wavelength_min = 3500, wavelength_max = 10000):

    '''
    remove orders not in wavelength range
    '''

    ii_good = np.sum((wavelength > wavelength_min) & (wavelength < wavelength_max), axis=1) == wavelength.shape[1]
    print("Keeping {}/{} orders between {}-{}".format(ii_good.sum(), len(ii_good), wavelength_min, wavelength_max))
    return wavelength[ii_good,:], spectrum[ii_good,:], spectrum_err[ii_good,:]

#------------------------------------------------------------------------------------------

def mask_wavelength_regions(wavelength, spectrum_err, mask_list):

    '''
    mask out a mask_list by setting infinite errors
    '''

    assert wavelength.shape == spectrum_err.shape
    for wmin, wmax in mask_list:
        assert wmin < wmax
        spectrum_err[(wavelength > wmin) & (wavelength < wmax)] = 999.
    return spectrum_err

#------------------------------------------------------------------------------------------

def scale_spectrum_by_median(spectrum, spectrum_err):

    '''
    dividing spectrum by its median
    '''

    for i in range(spectrum.shape[0]):
        scale_factor = 1./np.median(spectrum[i,:])
        spectrum[i,:] = spectrum[i,:]*scale_factor
        spectrum_err[i,:] = spectrum_err[i,:]*scale_factor
    return spectrum, spectrum_err

#---------------------------------------------------------------------

def whitten_wavelength(wavelength):

    '''
    normalize the wavelength of each order to facilitate the polynomial continuum fit
    '''

    wavelength_normalized = np.zeros(wavelength.shape)
    for k in range(wavelength.shape[0]):
        mean_wave = np.mean(wavelength[k,:])
        wavelength_normalized[k,:] = (wavelength[k,:]-mean_wave)/mean_wave
    return wavelength_normalized

#---------------------------------------------------------------------

def transform_coefficients(popt, NN_coeffs=None):

    '''
    Transform coefficients into human-readable
    '''

    if NN_coeffs is None:
        NN_coeffs, dummy = read_in_neural_network()
    w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs
    
    popt_new = popt.copy()
    popt_new[:4] = (popt_new[:4] + 0.5)*(x_max-x_min) + x_min
    popt_new[0] = popt_new[0]*1000.
    popt_new[-1] = popt_new[-1]*100.
    return popt_new

def normalize_stellar_parameter_labels(labels, NN_coeffs=None):
    '''
    Turn physical stellar parameter values into normalized values.
    Teff (K), logg (dex), FeH (solar), aFe (solar)
    '''
    assert len(labels)==4, "Input Teff, logg, FeH, aFe"
    # Teff, logg, FeH, aFe = labels
    labels = np.ravel(labels)
    labels[0] = labels[0]/1000.

    if NN_coeffs is None:
        NN_coeffs, dummy = read_in_neural_network()
    w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs
    new_labels = (labels - x_min) / (x_max - x_min) - 0.5
    assert np.all(new_labels >= -0.5), new_labels
    assert np.all(new_labels <=  0.5), new_labels
    return new_labels

#---------------------------------------------------------------------

