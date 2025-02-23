# code for fitting spectra, using the models in spectral_model.py
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import os
from . import spectral_model
from . import utils
from .read_spectrum import read_carpy_fits
from .fitting import evaluate_model, fit_continuum

def read_in_solar_spectrum():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/NN_normalized_spectra_float16.npz')
    wavelength, spectrum, spectrum_err = read_carpy_fits(path)
    return wavelength_solar, spectrum_solar, spectrum_err_solar
    
if __name__=="__main__":
    wavelength_solar, spectrum_solar, spectrum_err_solar = read_in_solar_spectrum()
    normalized_labels = utils.normalize_stellar_parameter_labels([5771, 4.44, 0.02, 0.00])
    rv_init = -14.6/100
    
    NN_coeffs, wavelength_payne = utils.read_in_neural_network()
    errors_payne = np.zeros_like(wavelength_payne)
    num_order, num_pixel = wavelength_solar.shape
    coeff_poly = 6 + 1
    
    spec_predict, errs_predict = evaluate_model(labels, NN_coeffs, wavelength_payne, errors_payne, coeff_poly, wavelength, num_order, num_pixel)
    
    def fit_func(labels):

        spec_predict, errs_predict = evaluate_model(labels, NN_coeffs, wavelength_payne, errors_payne,
                                                    coeff_poly, wavelength, num_order, num_pixel,
                                                    wavelength_normalized)

        # Calculate resids: set all potentially bad errors to 999.
        # We take errs > 300 as bad to account for interpolation issues on the mask
        errs = np.sqrt(errs_predict**2 + spectrum_err.ravel()**2)
        errs[(~np.isfinite(errs)) | (errs > 300) | (errs < 0)] = 999.
        resids = (spectrum.ravel() - spec_predict) / errs
        return resids

    p0 = np.zeros(4 + coeff_poly*num_order + 1 + 1)
    p0[0:4] = normalized_labels
    p0[4::coeff_poly] = 1
    p0[5::coeff_poly] = 0
    p0[6::coeff_poly] = 0
    p0[-2] = 1.0
    p0[-1] = rv_init
    
    bounds = np.zeros((2,p0.size))
    bounds[0,4:] = -1000 # polynomial coefficients
    bounds[1,4:] = 1000
    bounds[0,:4] = -0.5 # teff, logg, feh, alphafe
    bounds[1,:4] = 0.5
    bounds[0,-2] = 0.1 # vbroad
    bounds[1,-2] = 10.
    bounds[0,-1] = -0.2. # RV [100 km/s]
    bounds[1,-1] = 0.0.
    
    
    popt_best, model_spec_best, chi_square = fitting_mike(spectrum, spectrum_err, spectrum_blaze,\
                                                          wavelength, NN_coeffs, wavelength_payne,\
                                                          errors_payne=errors_payne,\
                                                          p0_initial=p0_initial, RV_prefit=False, blaze_normalized=True,\
                                                          RV_array=RV_array, polynomial_order=2, bounds_set=bounds_set)
    poly_initial = fit_continuum(spectrum, spectrum_err, wavelength, popt_best,\
                                         model_spec_best, polynomial_order=polynomial_order, previous_polynomial_order=2)
