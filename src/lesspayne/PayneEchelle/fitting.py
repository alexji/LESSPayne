# code for fitting spectra, using the models in spectral_model.py
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy import interpolate
from scipy import signal
from scipy import linalg
from scipy.stats import norm
from . import spectral_model
from . import utils

#------------------------------------------------------------------------------------------

def fit_global(spectrum, spectrum_err, spectrum_blaze, wavelength,
               model,
               rv_model=None,
               prefit_model=None,
               RV_array=np.linspace(-1,1.,6), order_choice=[20],\
               default_rv_polynomial_order=2,
               bounds_set=None,
               initial_stellar_parameters=None,
               skip_rv_prefit=False, RV_range=500,
               poly_coeff_min=-1000, poly_coeff_max=1000,
               get_perr=False):

    '''
    Fitting MIKE spectrum
    Fitting stellar labels, polynomial continuum, vbroad, and radial velocity simultaneously

    spectrum and spectrum_err are the spectrum to be fitted and its uncertainties
    spectrum_blaze is a hot star spectrum used as a reference spectrum
    wavelength is the wavelength of the pixels
    they are all 2D array with [number of spectral order, number of pixels]

    NN_coeffs are the neural network emulator coefficients
    we adopt Kurucz models in this study
    wavelength_payne is the output wavelength of the emulator

    RV_array is the radial velocity initialization that we will run the fit
    order_choice is the specific order that we will use to pre-determine the radial velocitiy
    the radial velocity is then used as the initialization for the global fit
    MIKE spectrum typically has about ~35 orders in the red

    polynomial_order is the final order of polynomial that we will assume for the continuum of individual orders
    A 6th order polynomial does a decent job

    initial_stellar_parameters: input initial values for 
       Teff, logg, Fe/H, a/Fe (in physical units of K, dex, solar, solar)
    '''

    # first we fit for a specific order while looping over all RV initalization
    # the spectrum is pre-normalized with the blaze function
    # we assume a quadratic polynomial for the residual continuum
    if skip_rv_prefit:
        RV_array = RV_array[0:1]
        print('Pre Fit: skipping radial velocity initialization, using', str(RV_array))
    else:
        if rv_model is None:
            rv_model = type(model)(model.NN_coeffs, model.num_stellar_labels,
                                   model.x_min, model.x_max,
                                   model.wavelength_payne, model.errors_payne,
                                   len(order_choice), default_rv_polynomial_order, 1)
        popt_best, model_spec_best, chi_square, perr = fitting_mike(spectrum, spectrum_err, spectrum_blaze,\
                                                              wavelength, rv_model,
                                                              p0_initial=None, 
                                                              RV_prefit=True, blaze_normalized=True,\
                                                              RV_array=RV_array, bounds_set=bounds_set,\
                                                              order_choice=order_choice, RV_range=RV_range,
                                                              poly_coeff_min=poly_coeff_min, poly_coeff_max=poly_coeff_max)
        RV_array = np.array([popt_best[-1]])

    # we then fit for all the orders
    # we adopt the RV from the previous fit as the sole initialization
    # the spectrum is still pre-normalized by the blaze function
    if prefit_model is None:
        ## same model, but with order 2
        prefit_model = type(model)(model.NN_coeffs, model.num_stellar_labels,
                                   model.x_min, model.x_max,
                                   model.wavelength_payne, model.errors_payne,
                                   model.num_order, default_rv_polynomial_order, model.num_chunk,
                                   chunk_order_min=model.chunk_order_min,
                                   chunk_order_max=model.chunk_order_max
        )
    if initial_stellar_parameters is not None:
        p0_initial = prefit_model.get_p0_initial_normspec(initial_stellar_parameters,
                                                          initial_rv=RV_array[0])
    else:
        p0_initial = None

    popt_best, model_spec_best, chi_square, perr = fitting_mike(spectrum, spectrum_err, spectrum_blaze,\
                                                          wavelength, prefit_model,
                                                          p0_initial=p0_initial, 
                                                          RV_prefit=False, blaze_normalized=True,\
                                                          RV_array=RV_array, bounds_set=bounds_set, RV_range=RV_range,
                                                          poly_coeff_min=poly_coeff_min, poly_coeff_max=poly_coeff_max)

    # using this fit, we can subtract the raw spectrum with the best fit model of the normalized spectrum
    # with which we can then estimate the continuum for the raw specturm
    poly_initial = fit_continuum(spectrum, spectrum_err, wavelength, popt_best,\
                                 model_spec_best, start_index=model.num_stellar_labels,
                                 polynomial_order=model.polynomial_order,
                                 previous_polynomial_order=prefit_model.polynomial_order)

    # using all these as intialization, we are ready to do the final fit
    RV_array = np.array([popt_best[-1]])
    p0_initial = np.concatenate([popt_best[:model.num_stellar_labels],
                                 poly_initial.ravel(),
                                 popt_best[-2*model.num_chunk:]])
    
    popt_best, model_spec_best, chi_square, perr = fitting_mike(spectrum, spectrum_err, spectrum_blaze,\
                                                          wavelength, model,
                                                          p0_initial=p0_initial, bounds_set=bounds_set,\
                                                          RV_prefit=False, blaze_normalized=False,\
                                                          RV_array=RV_array, RV_range=RV_range,
                                                          poly_coeff_min=poly_coeff_min, poly_coeff_max=poly_coeff_max)
    if get_perr:
        return popt_best, model_spec_best, chi_square, perr
    return popt_best, model_spec_best, chi_square

#------------------------------------------------------------------------------------------

def fit_continuum(spectrum, spectrum_err, wavelength, previous_poly_fit, previous_model_spec,\
                  start_index=4, polynomial_order=6, previous_polynomial_order=2):

    '''
    Fit the continuum while fixing other stellar labels

    The end results will be used as initial condition in the global fit (continuum + stellar labels)
    '''

    print('Pre Fit: Finding the best continuum initialization')

    # normalize wavelength grid
    wavelength_normalized = utils.whitten_wavelength(wavelength)*100.

    # number of polynomial coefficients
    coeff_poly = polynomial_order + 1
    pre_coeff_poly = previous_polynomial_order + 1

    # initiate results array for the polynomial coefficients
    fit_poly = np.zeros((wavelength_normalized.shape[0],coeff_poly))

    # loop over all order and fit for the polynomial (weighted by the error)
    for k in range(wavelength_normalized.shape[0]):
        pre_poly = 0
        for m in range(pre_coeff_poly):
            pre_poly += (wavelength_normalized[k,:]**m)*previous_poly_fit[start_index+m+pre_coeff_poly*k]
        substract_factor =  (previous_model_spec[k,:]/pre_poly) ## subtract away the previous fit
        try:
            fit_poly[k,:] = np.polyfit(wavelength_normalized[k,:], spectrum[k,:]/substract_factor,\
                                       polynomial_order, w=1./(spectrum_err[k,:]/substract_factor))[::-1]
        except:
            import pdb; pdb.set_trace()

    return fit_poly


#------------------------------------------------------------------------------------------

def evaluate_model(labels, NN_coeffs, wavelength_payne, errors_payne, coeff_poly, wavelength, num_order, num_pixel,
                   wavelength_normalized=None):

    # get wavelength_normalized
    if wavelength_normalized is None:
        wavelength_normalized = utils.whitten_wavelength(wavelength)*100.

    # make payne models
    full_spec = spectral_model.get_spectrum_from_neural_net(\
                                scaled_labels = labels[:4],
                                NN_coeffs = NN_coeffs)
    # broadening kernel
    win = norm.pdf((np.arange(21)-10.)*(wavelength_payne[1]-wavelength_payne[0]),\
                            scale=labels[-2]/3e5*5000)
    win = win/np.sum(win)

    # vbroad -> RV
    full_spec = signal.convolve(full_spec, win, mode='same')
    full_spec = utils.doppler_shift(wavelength_payne, full_spec, labels[-1]*100.)
    errors_spec = utils.doppler_shift(wavelength_payne, errors_payne, labels[-1]*100.)
    
    # interpolate into the observed wavelength
    f_flux_spec = interpolate.interp1d(wavelength_payne, full_spec)
    f_errs_spec = interpolate.interp1d(wavelength_payne, errors_spec)

    # loop over all orders
    spec_predict = np.zeros(num_order*num_pixel)
    errs_predict = np.zeros(num_order*num_pixel)
    for k in range(num_order):
        scale_poly = 0
        for m in range(coeff_poly):
            scale_poly += (wavelength_normalized[k,:]**m)*labels[4+coeff_poly*k+m]
        spec_predict[k*num_pixel:(k+1)*num_pixel] = scale_poly*f_flux_spec(wavelength[k,:])
        errs_predict[k*num_pixel:(k+1)*num_pixel] = scale_poly*f_errs_spec(wavelength[k,:])

    return spec_predict, errs_predict

def fitting_mike(spectrum, spectrum_err, spectrum_blaze,\
                 wavelength, model, \
                 p0_initial=None, bounds_set=None,\
                 RV_prefit=False, blaze_normalized=False, RV_array=np.linspace(-1,1.,6),\
                 order_choice=[20], RV_range=500,
                 poly_coeff_min=-1000, poly_coeff_max=1000):

    '''
    Fitting MIKE spectrum

    Fitting radial velocity can be very multimodal. The best strategy is to initalize over
    different RVs. When RV_prefit is true, we first fit a single order to estimate
    radial velocity that we will adopt as the initial guess for the global fit.

    RV_array is the range of RV that we will consider
    RV array is in the unit of 100 km/s
    order_choice is the order that we choose to fit when RV_prefit is TRUE

    When blaze_normalized is True, we first normalize spectrum with the blaze

    Returns:
        Best fitted parameter (Teff, logg, Fe/H, Alpha/Fe, polynomial coefficients, vmacro, RV)
    '''

    #NN_coeffs, wavelength_payne,\
    #    errors_payne=None,\
    # assume no model error if not specified
    #if errors_payne is None:
    #    errors_payne = np.zeros_like(wavelength_payne)

    # normalize wavelength grid
    wavelength_normalized = utils.whitten_wavelength(wavelength)*100.

    # number of polynomial coefficients
    coeff_poly = model.coeff_poly

    # specify a order for the (pre-) RV fit
    if RV_prefit:
        spectrum = spectrum[order_choice,:]
        spectrum_err = spectrum_err[order_choice,:]
        spectrum_blaze = spectrum_blaze[order_choice,:]
        wavelength_normalized = wavelength_normalized[order_choice,:]
        wavelength = wavelength[order_choice,:]

    # normalize spectra with the blaze function
    if blaze_normalized:
        spectrum = spectrum/spectrum_blaze
        spectrum_err = spectrum_err/spectrum_blaze

    # number of pixel per order, number of order
    num_pixel = spectrum.shape[1]
    num_order = spectrum.shape[0]

#------------------------------------------------------------------------------------------
    # the objective function
    def fit_func(labels):

        #spec_predict, errs_predict = evaluate_model(labels, NN_coeffs, wavelength_payne, errors_payne,
        #                                            coeff_poly, wavelength, num_order, num_pixel,
        #                                            wavelength_normalized)
        spec_predict, errs_predict = model.evaluate(labels, wavelength, wavelength_normalized)

        # Calculate resids: set all potentially bad errors to 999.
        # We take errs > 300 as bad to account for interpolation issues on the mask
        errs = np.sqrt(errs_predict**2 + spectrum_err.ravel()**2)
        errs[(~np.isfinite(errs)) | (errs > 300) | (errs < 0)] = 999.
        resids = (spectrum.ravel() - spec_predict) / errs
        return resids

#------------------------------------------------------------------------------------------
    # loop over all possible
    chi_2 = np.inf

    if RV_prefit:
        print('Pre Fit: Finding the best radial velocity initialization')

    if not(RV_prefit) and blaze_normalized:
        print('Pre Fit: Fitting the blaze-normalized spectrum')

    if not(RV_prefit) and not(blaze_normalized):
        print('Final Fit: Fitting the whole spectrum with all parameters simultaneously')
        #popt_for_printing = model.transform_coefficients(p0_initial)
        #print('p0 = Teff={:.0f} logg={:.2f} FeH={:.2f} aFe={:.2f} vbroad={:.2f} rv={:.1f}'.format(
        #    *[popt_for_printing[i] for i in [0,1,2,3,-2,-1]]))
        printstr = model.get_print_string(p0_initial)
        print('p0:',printstr)

    for i in range(RV_array.size):
        print(i+1, "/", RV_array.size)

        # initialize the parameters (Teff, logg, Fe/H, alpha/Fe, polynomial continuum, vbroad, RV)
        if p0_initial is None:
            p0 = model.get_p0_initial_normspec(initial_rv=RV_array[i])
            ## TODO!!!!
            #p0 = np.zeros(4 + coeff_poly*num_order + 1 + 1)
            #p0[4::coeff_poly] = 1
            #p0[5::coeff_poly] = 0
            #p0[6::coeff_poly] = 0
            #p0[-2] = 0.5
            #p0[-1] = RV_array[i]
        else:
            p0 = p0_initial

        # set fitting bound
        bounds = model.get_initial_bounds(bounds_set, rvmin=100*RV_array[i]-RV_range, rvmax=100*RV_array[i]+RV_range,
                                          poly_coeff_min=poly_coeff_min, poly_coeff_max=poly_coeff_max)

        if (not(bounds_set is None)) and (p0_initial is None):
            p0[:model.num_stellar_labels] = np.mean(bounds_set[:,:model.num_stellar_labels], axis=0)

        # run the optimizer
        tol = 5e-4
        #popt, pcov = curve_fit(fit_func, xdata=[],\
        #                       ydata = spectrum.ravel(), sigma = spectrum_err.ravel(),\
        #                       p0 = p0, bounds=bounds, ftol = tol, xtol = tol, absolute_sigma = True,\
        #                       method = 'trf')
        try:
            res = least_squares(fit_func, p0,
                                bounds=bounds, ftol=tol, xtol=tol, method='trf')
        except ValueError as e:
            print("Error in optimization")
            #print("Error: p0 = ",p0)
            #print("bounds min = ",bounds[0])
            #print("bounds max = ",bounds[1])
            for i, (_p0, _b0, _b1) in enumerate(zip(p0, bounds[0], bounds[1])):
                if _p0 <= _b0 or _p0 >= _b1:
                    print(f"Bad param {i}: fit {_p0} lower {_b0} upper {_b1}")
            print("If this is a continuum coefficient, you can change poly_coeff_min/max")
            raise(e)

        if not res.success:
            raise RuntimeError("Optimal parameters not found: " + res.message)
        popt = res.x
        # Based on Yupeng's code and https://stackoverflow.com/questions/42388139/how-to-compute-standard-deviation-errors-with-scipy-optimize-least-squares
        U, s, Vh = linalg.svd(res.jac, full_matrices=False)
        tol = np.finfo(float).eps*s[0]*max(res.jac.shape)
        w = s > tol
        cov = (Vh[w].T/s[w]**2) @ Vh[w] # robust covariance matrix
        perr = np.sqrt(np.diag(cov)) # 1sigma uncertainty on fitted parameters

        # calculate chi^2
        model_spec, model_errs = model.evaluate(popt, wavelength, wavelength_normalized)
        #model_spec, model_errs = evaluate_model(popt, NN_coeffs, wavelength_payne, errors_payne,
        #                                        coeff_poly, wavelength, num_order, num_pixel,
        #                                        wavelength_normalized)
        chi_2_temp = np.mean((spectrum.ravel() - model_spec)**2/(model_errs + spectrum_err.ravel()**2))

        # check if this gives a better fit
        if chi_2_temp < chi_2:
            chi_2 = chi_2_temp
            model_spec_best = model_spec
            popt_best = popt

    return popt_best, model_spec_best.reshape(num_order,num_pixel), chi_2, perr
