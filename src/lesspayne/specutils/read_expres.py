from astropy.table import Table
from collections import OrderedDict
import numpy as np
from copy import deepcopy
from scipy import signal

def read_expres(fname, full_output=False, as_arrays=False, as_order_dict=False, as_raw_table=False):
    if full_output:
        raise NotImplementedError("For now use as_raw_table=True")
    tab = Table.read(fname, hdu=1)
    if as_raw_table: return tab
    
    try:
        orders = tab["order"]
    except:
        orders = tab["orders"]
    cols = ["wavelength", "spectrum", "uncertainty", "continuum",
            "offset","offset_uncertainty","n_pixels","reduced_chi",
            "continuum_mask","pixel_mask","tellurics",
            "bary_wavelength"]
    Nord = len(orders)
    
    if as_order_dict:
        alloutput = OrderedDict()
    elif as_arrays:
        alloutput = [[], [], []]
    else:
        alloutput = []
    
    meta = {"file":fname}
    for iord in range(Nord):
        meta["order"] = orders[iord]
        wave = tab["bary_wavelength"][iord]
        flux = tab["spectrum"][iord]
        errs = tab["uncertainty"][iord]
        if as_arrays:
            alloutput[0].append(wave)
            alloutput[1].append(flux)
            alloutput[2].append(errs)
        else:
            from .spectrum import Spectrum1D
            spec = Spectrum1D(wave, flux, errs**-2, metadata=meta)
            if as_order_dict:
                alloutput[orders[iord]] = spec
            else:
                alloutput.append(spec)
    if as_arrays:
        alloutput[0] = np.array(alloutput[0])
        alloutput[1] = np.array(alloutput[1])
        alloutput[2] = np.array(alloutput[2])
    return alloutput

def rebin_spec(spec, n_rebin):
    """
    Sum n_rebin pixels together
    """
    from .spectrum import Spectrum1D
    
    n_new = len(spec.dispersion) // n_rebin
    n_orig = n_new * n_rebin
    
    wave = spec.dispersion[0:n_orig].reshape((-1,n_rebin))
    flux = spec.flux[0:n_orig].reshape((-1,n_rebin))
    errs = (spec.ivar[0:n_orig]**-0.5).reshape((-1,n_rebin))

    wave = np.mean(wave, axis=1)
    flux = np.sum(flux, axis=1)
    errs = np.sqrt(np.sum(errs**2, axis=1))
    return Spectrum1D(wave, flux, errs**-2, spec.metadata)
