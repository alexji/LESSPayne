# code for masking
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import os
from . import spectral_model
from . import utils

def get_norm_params(popt, model):
    """ Create parameters to output normalized spectra with model.evaluate """
    ## Eventually we can modify these but for now assume the simple case of 1 chunk
    assert model.num_order == 1, model.num_order
    assert model.num_chunk == 1, model.num_chunk
    params = np.zeros(model.num_stellar_labels + 1*model.coeff_poly + 2*model.num_chunk)
    params[0:model.num_stellar_labels] = popt[0:model.num_stellar_labels]
    params[model.num_stellar_labels] = 1.0
    params[-2] = popt[-2]
    params[-1] = popt[-1]
    return params

def merge_exclude_regions(super_wave, exclude_regions):
    super_mask = np.zeros_like(super_wave, dtype=bool)
    for excludes in exclude_regions:
        super_mask[(super_wave > excludes[0]) & (super_wave < excludes[1])] = True
    maskdiff = np.diff(np.concatenate([[False], super_mask]).astype(int))
    starts = np.where(maskdiff == 1)[0]
    stops = np.where(maskdiff == -1)[0]
    if len(stops) < len(starts):
        assert len(stops) + 1 == len(starts)
        stops = list(stops) + [len(wave)]
    all_exclude_regions = [[super_wave[start], super_wave[stop]] for start, stop in zip(starts, stops)]
    return all_exclude_regions

def get_mask(
        payne_output,
        mask_sigma = 1.0,
        mask_smooth = 3,
        mask_thresh = 0.1,
):
    """
    Find line masks using the model
    
    payne_output: output file from running a standard Payne4MIKE fit
    mask_sigma
    mask_smooth
    mask_thresh
    """
    #### TODO: update this to all read info from payne output


    ## Step 2: get median S/N for each order
    waves = [spec.dispersion for spec in session.input_spectra]
    fluxs = [spec.flux for spec in session.input_spectra]
    errs = [spec.ivar**-0.5 for spec in session.input_spectra]
    snrs = [np.nanmedian(flux/err) for flux,err in zip(fluxs, errs)]
    ## Step 3: get normalized Payne spectrum at the right RV, smoothing, wavelengths
    ##         get masks based on pixels that deviate by more than mask_sigma/snr
    normalized_payne_spectra = []
    payne_masks = []
    norm_params = get_norm_params(popt_best, model)
    for wave, snr in zip(waves, snrs):
        print(f"Wave={np.median(wave):.0f} SNR={snr:.1f}")
        try:
            payne_spec = model.evaluate(norm_params, wave[np.newaxis,:])[0]
        except Exception as e:
            print(f"Model evalutaion exception at wave={wave[0]:.0f}-{wave[-1]:.0f}, no masks")
            print(e)
            payne_spec = np.ones_like(wave)
        mask = 1 - payne_spec > mask_sigma/snr
        # Smooth out the masks and make them a bit wider
        mask = gaussian_filter1d(mask.astype(np.float), mask_smooth) > mask_thresh
        normalized_payne_spectra.append(payne_spec)
        payne_masks.append(mask)
        print(f"  num_mask={mask.sum()}/{mask.size}")
    #"""
    for wave, mask, snr, payne_spec in zip(waves, payne_masks, snrs, normalized_payne_spectra):
        plt.plot(wave, payne_spec, lw=.5, alpha=.5, color='k')
        plt.vlines(wave[mask], 0.0, 1.0, color='r', alpha=.5)
        plt.hlines([1-mask_sigma/snr], wave[0], wave[-1], color='b')
    #"""
    ## Convert masks into exclude regions
    exclude_regions = []
    exclude_regions_2 = []
    for wave, mask, payne_spec in zip(waves, payne_masks, normalized_payne_spectra):
        mask2 = np.concatenate([[False], mask]).astype(int)
        maskdiff = np.diff(mask2)
        starts = np.where(maskdiff == 1)[0]
        stops = np.where(maskdiff == -1)[0]
        if len(stops) < len(starts):
            assert len(stops) + 1 == len(starts)
            stops = list(stops) + [len(wave)]
        assert len(stops) == len(starts)
        drv = rv/3e5 * np.median(wave)
        dwave = np.median(np.diff(wave)) 
        print(f"{np.median(wave):.0f}, {len(starts)} regions, drv={drv:.1f}A")
        excludes = [[drv + wave[start]-dwave, drv + wave[stop-1]+dwave] for start, stop in zip(starts, stops)]
        exclude_regions.extend(excludes)
        excludes = [[-drv + wave[start]-dwave, -drv + wave[stop-1]+dwave] for start, stop in zip(starts, stops)]
        exclude_regions_2.extend(excludes)
    ## Merge exclude regions
    super_wave = np.arange(np.min(waves), np.max(waves), np.min(np.diff(waves, axis=1)))
    all_exclude_regions = merge_exclude_regions(super_wave, exclude_regions)
    all_exclude_regions_2 = merge_exclude_regions(super_wave, exclude_regions_2)
    print(f"Total excluded regions = {len(all_exclude_regions)}")

