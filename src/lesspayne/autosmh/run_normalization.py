import numpy as np
import sys, os, time
import yaml
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d

from lesspayne.smh import Session
from lesspayne.specutils import Spectrum1D 
from lesspayne.smh.spectral_models import ProfileFittingModel, SpectralSynthesisModel
from lesspayne.smh.photospheres.abundances import asplund_2009 as solar_composition
from lesspayne.PayneEchelle.spectral_model import DefaultPayneModel

## this doesn't work right now, and the Payne has it hardcoded anyway
telluric_mask = []

def plot_norm_one_session(session, outfname, telescope=None, indices=None):
    default_Ncol = 10
    width, height = 6, 3 # inches
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    specs = session.input_spectra
    rv = session.metadata["rv"]["rv_applied"]
    rvc = rv/3e5
    meta = session.metadata["normalization"]
    if telescope is None:
        Ncol = default_Ncol
        Nspec = len(specs)
        Nrow = (Nspec // Ncol) + int(Nspec % Ncol > 0)
        assert Nrow*Ncol >= Nspec
        indices = np.arange(Nrow*Ncol)
    else:
        if telescope == "MAGEL":
            Nrow, Ncol = 7, 10
        elif telescope == "DUPON":
            Nrow, Ncol = 8, 10
        elif telescope == "APO35":
            Nrow, Ncol = 11, 10
        elif telescope == "MCD27":
            Nrow, Ncol = 7, 10

    assert len(specs) <= Nrow * Ncol, (len(specs), Nrow*Ncol)
    fig1, axes1 = plt.subplots(Nrow, Ncol,
                               figsize=(Ncol*width, Ncol*height))
    #fig2, axes2 = plt.subplots(Nrow, Ncol,
    #                           figsize=(Ncol*width, Ncol*height))
    for i in range(len(specs)):
        ix = indices[i]
        if ix < 0: continue # not a good order to plot
        
        ax1 = axes1.flat[ix]
        #ax2 = axes2.flat[ix]
        spec = specs[i]
        wave = spec.dispersion
        flux = spec.flux
        try:
            cont = meta["continuum"][i]
            excludes = meta["normalization_kwargs"][i]["exclude"]
        except:
            cont = np.zeros_like(flux) + np.nan
            excludes = []
        
        ax1.plot(wave, flux, 'k-', lw=1, rasterized=True)
        ax1.plot(wave, cont, 'r-', lw=1, rasterized=True)
        #ax2.plot(wave, flux/cont, 'k-', lw=1, rasterized=True)
        #ax2.axhline(1, color='grey', zorder=-9)

        #for ax, (y1,y2) in zip(
        #        [ax1, ax2], [(0, np.nanpercentile(flux,99)), (0, 1.1)]):
        for ax, (y1,y2) in zip(
                [ax1], [(0, np.nanpercentile(flux,99))]):
            trimming = (wave[-1] - wave[0]) * 0.02
            ax.set_xlim(wave[0] - trimming, wave[-1] + trimming)
            trimming = (y2 - y1) * 0.02
            ax.set_ylim(y1 - trimming, y2 + trimming)
            
            for [x1, x2] in excludes:
                kwds = {
                    "xmin": x1*(1+rvc),
                    "xmax": x2*(1+rvc),
                    "ymin": -1e8,
                    "ymax": 1e8,
                    "facecolor": "r",
                    "edgecolor": "none",
                    "alpha": 0.25,
                    "zorder": -1
                }
                ax.axvspan(**kwds)
    fig1.tight_layout()
    fig1.savefig(outfname, dpi=150)
    #fig2.tight_layout()
    #fig2.savefig(outfname2, dpi=200)
    #fig2 = None
    return fig1

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

def merge_exclude_regions(super_wave, exclude_regions, Nwave):
    super_mask = np.zeros_like(super_wave, dtype=bool)
    for excludes in exclude_regions:
        super_mask[(super_wave > excludes[0]) & (super_wave < excludes[1])] = True
    maskdiff = np.diff(np.concatenate([[False], super_mask]).astype(int))
    starts = np.where(maskdiff == 1)[0]
    stops = np.where(maskdiff == -1)[0]
    if len(stops) < len(starts):
        assert len(stops) + 1 == len(starts)
        stops = list(stops) + [Nwave]
    all_exclude_regions = [[super_wave[start], super_wave[stop]] for start, stop in zip(starts, stops)]
    return all_exclude_regions

def get_normalization_keywords(spec, ncfg):
    """
    Sets knot spacing and sigma clipping
    """
    wave = spec.dispersion
    w1, w2 = wave[0], wave[-1]
    wc = (w1+w2)/2

    ## Knot Spacing
    default_knot_spacing = ncfg.get("default_knot_spacing", 15)
    ## parameters for setting knot spacing based on wavelength
    # Format: [(w1,w2,ks), (w1,w2,ks)]
    knot_wave_list = ncfg.get("knot_wave_list",
                              [(3000, 6800, 15), (6800, 10000, 25)])
    for (kwave1, kwave2, kspace) in knot_wave_list:
        if (wc > kwave1) & (wc <= kwave2):
            knot_spacing = kspace
            break
    else:
        knot_spacing = 15
    ## The above should re-implement this by default
    # if wc < 6800:
    #     knot_spacing = 15
    # else: # tellurics and red and fewer lines
    #     knot_spacing = 25
    
    ## Sigma clip. These are hardcoded for now. TODO PUT INTO CFG
    snr = np.nanmedian(spec.flux * spec.ivar**.5)
    if snr < ncfg.get("lowsnr_snr_threshold", 5.0):
        high_sigma_clip = ncfg.get("lowsnr_high_sigma_clip",1.0)
        low_sigma_clip = ncfg.get("lowsnr_low_sigma_clip",2.0)
    else:
        sigma_clip_wave_list = ncfg.get("sigma_clip_wave_list",
            [(3000, 10000, 1.0, 5.0)])
        for (kwave1, kwave2, hiclip, loclip) in sigma_clip_wave_list:
            if (wc > kwave1) & (wc <= kwave2):
                high_sigma_clip = hiclip
                low_sigma_clip = loclip
                break
        else:
            high_sigma_clip = 1.0
            low_sigma_clip = 5.0
    
    return knot_spacing, high_sigma_clip, low_sigma_clip

def get_exclude_regions(spec, all_exclude_regions, telluric_mask, 
                        rv=0, blue_trim=0, red_trim=0):
    wave = spec.dispersion
    w1, w2 = wave[0], wave[-1]
    wc = (w1+w2)/2
    # Create masks for this order
    this_exclude = [x for x in all_exclude_regions if (x[1] > w1 and x[0] < w2)]
    this_exclude_telluric = [x for x in telluric_mask if (x[1] > w1 and x[0] < w2)]
    
    # Calculate rv offset in wavelength units
    drv = rv/3e5 * np.median(wave)
    
    ## Add trim to telluric exclusion region, so it's always added in
    if blue_trim > 0:
        this_exclude_telluric.append([wave[0]-1e-3, wave[blue_trim]+1e-3])
    if red_trim > 0:
        this_exclude_telluric.append([wave[-red_trim]-1e-3, wave[-1]+1e-3])
    
    all_exclude = this_exclude + this_exclude_telluric
    return all_exclude


def make_masks(wave, snr, model, norm_params, mask_sigma, mask_smooth, mask_thresh):
    try:
        payne_spec = model.evaluate(norm_params, wave[np.newaxis,:])[0]
    except Exception as e:
        print(f"Model evaluation exception at wave={wave[0]:.0f}-{wave[-1]:.0f}, no masks")
        print(e)
        payne_spec = np.ones_like(wave)
    mask = 1 - payne_spec > mask_sigma/snr
    # Smooth out the masks and make them a bit wider
    mask = gaussian_filter1d(mask.astype(np.float), mask_smooth) > mask_thresh
    return payne_spec, mask

def run_normalization(cfg):
    name = cfg["output_name"]
    NNpath = cfg["NN_file"]
    outdir = cfg["output_directory"]
    figdir = cfg["figure_directory"]
    if not os.path.exists(outdir): os.makedirs(outdir)
    if not os.path.exists(figdir): os.makedirs(figdir)
    print("Saving to output directory:",outdir)
    print("Saving figures to output directory:",figdir)
    
    popt_fname = os.path.join(outdir, cfg["payne_fname"])
    out_fname = os.path.join(outdir, cfg["smh_fname"])
    spectrum_fnames = cfg["spectrum_fnames"]

    ncfg = cfg["run_normalization"]
    # threshold for deciding whether to mask a line in the Payne model
    mask_sigma = ncfg["mask_sigma"]
    # parameters for growing the mask size a bit
    mask_smooth = ncfg["mask_smooth"]
    mask_thresh = ncfg["mask_thresh"]
    # maximum fraction of ; increases mask_sigma by 0.5 until reached
    max_mask_frac = ncfg["max_mask_frac"]
    # minimum fraction of unmasked pixels/knot; increases knot_spacing by 2 until reached
    min_frac_per_knot = ncfg["min_frac_per_knot"]
    ## default masking on each side
    blue_trim = ncfg["blue_trim"]
    red_trim = ncfg["red_trim"]
    ## default spline kws
    function = ncfg.get("function","spline")
    if function == "spline":
        default_kwds = dict(
            function="spline",
            order=ncfg["continuum_spline_order"], max_iterations=ncfg["continuum_max_iterations"],
            full_output=True,
            blue_trim=None, red_trim=None,
        )
    elif function == "polynomial":
        default_kwds = dict(
            function="polynomial",
            order=ncfg["continuum_spline_order"], max_iterations=ncfg["continuum_max_iterations"],
            full_output=True,
            blue_trim=None, red_trim=None,
        )
    else:
        raise ValueError(f"normalization function = {function}")
    
    

    ## Saving these to the session
    normalization_parameters = {}
    notes = f"run_normalization parameters:\n  {popt_fname}\n"
    for key in ["mask_sigma","mask_smooth","mask_thresh","max_mask_frac","min_frac_per_knot",
                "blue_trim","red_trim","continuum_spline_order","continuum_max_iterations"]:
        normalization_parameters[key] = ncfg[key]
        notes += f"  {key}: {ncfg[key]}\n"
    notes = notes[:-1] # remove the last \n

    assert popt_fname.endswith(".npz") or popt_fname.endswith(".npy"), popt_fname
    assert out_fname.endswith(".smh"), out_fname
    #assert not os.path.exists(out_fname), f"{out_fname} already exists! exiting"
    assert os.path.exists(NNpath), NNpath
    
    startall = time.time()
    
    ## Step 1: load in data
    start = time.time()
    session = Session(spectrum_fnames)
    with open(Session._default_settings_path, "rb") as fp:
        defaults = yaml.load(fp, yaml.FullLoader)
    session.metadata.update(defaults)    
    
    num_order = len(session.input_spectra)
    # We will only evaluate the model 1 order at a time right now
    model = DefaultPayneModel.load(NNpath, 1)
    with np.load(popt_fname) as tmp:
        popt_best = tmp["popt_best"].copy()
        popt_print = tmp["popt_print"].copy()

    session.initialize_rv()
    session.initialize_normalization()
    rv = popt_best[-1]*100
    print(f"Correcting at rv={rv:.1f}")
    session.rv_correct(rv)
    session.metadata["rv"]["rv_measured"] = float(rv)
    print(f"Total to load in data: {time.time()-start:.1f}")
    
    ## Step 2: get median S/N for each order
    waves = [spec.dispersion for spec in session.input_spectra]
    fluxs = [spec.flux for spec in session.input_spectra]
    errs = [spec.ivar**-0.5 for spec in session.input_spectra]
    snrs = [np.nanmedian(flux/err) for flux,err in zip(fluxs, errs)]

    ## Step 3: get normalized Payne spectrum at the right RV, smoothing, wavelengths
    ##         get masks based on pixels that deviate by more than mask_sigma/snr
    start = time.time()
    normalized_payne_spectra = []
    payne_masks = []
    norm_params = get_norm_params(popt_best, model)
    print("norm params",norm_params)
    for wave, snr in zip(waves, snrs):
        print(f"Wave={np.median(wave):.0f} SNR={snr:.1f}")
        this_mask_sigma = mask_sigma
        for it in range(6): # some arbitrary parameters here
            payne_spec, mask = make_masks(wave, snr, model, norm_params, this_mask_sigma, mask_smooth, mask_thresh)
            mask_frac = mask.sum()/float(mask.size)
            if mask_frac < max_mask_frac: break
            this_mask_sigma += 0.5
            print(f"  {mask_frac:.2f} > {max_mask_frac} increasing mask_sigma to {this_mask_sigma}")
        normalized_payne_spectra.append(payne_spec)
        payne_masks.append(mask)
        print(f"  num_mask={mask.sum()}/{mask.size}")
    
    ## Convert masks into exclude regions
    ## exclude_regions = in observed wavelengths, for normalization masks
    ## exclude_regions_2 = in rest-frame wavelengths, for eqw masks
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
    Nwave = np.max([len(wave) for wave in waves])
    all_exclude_regions = merge_exclude_regions(super_wave, exclude_regions, Nwave)
    all_exclude_regions_2 = merge_exclude_regions(super_wave, exclude_regions_2, Nwave)
    print(f"Total excluded regions = {len(all_exclude_regions)}")
    print(f"Total to get excluded regions: {time.time()-start:.1f}")
    
    ## Saving the Payne masks to the session along with the metadata
    ## In principle we should be able to use this to reset normalizations
    session.metadata["payne_masks"] = [all_exclude_regions, all_exclude_regions_2, normalization_parameters]
    
    ## Step 4: create normalizations with masks
    start = time.time()
    norms = [np.nan for i in range(num_order)]
    conts = [np.nan for i in range(num_order)]
    all_kwds = [{} for i in range(num_order)]

    for i, spec in enumerate(session.input_spectra):
        kwds = deepcopy(default_kwds)
        
        # Apply RV correction before normalization, mimicking SMHR normalization gui
        spec = spec.copy()
        try:
            v = session.metadata["rv"]["rv_applied"]
        except (AttributeError, KeyError):
            v = 0
        spec._dispersion *= (1 - v/299792458e-3) # km/s

        # Adjust normalization keywords for this order
        knot_spacing, high_sigma_clip, low_sigma_clip = get_normalization_keywords(spec, ncfg)
        kwds["high_sigma_clip"] = high_sigma_clip
        kwds["low_sigma_clip"] = low_sigma_clip

        # Create masks for this order
        all_exclude = get_exclude_regions(spec, all_exclude_regions, telluric_mask, 
                                          rv=rv, blue_trim=blue_trim, red_trim=red_trim)
        kwds["exclude"] = all_exclude
        
        # Ensure each knot has a significant number of unmasked pixels
        wave = spec.dispersion
        # Recompute the mask
        mask = np.zeros(wave.size, dtype=bool)
        for w1, w2 in all_exclude:
            mask[(w1 <= wave) & (wave <= w2)] = True
        print(f"Wave={np.median(wave):.0f}")
        for it in range(10):
            try:
                knots = spec.get_knots(knot_spacing, exclude=all_exclude)
                # assign pixels to knots and count the unmasked points
                best_knot = np.argmin(np.abs(wave[:,np.newaxis] - knots[np.newaxis,:]), axis=1)
                assert best_knot.size==wave.size, (best_knot.size, wave.size)
                frac_per_knot = np.inf
                # this finds the worst knot
                for j, knot in enumerate(knots):
                    ii1 = j==best_knot
                    ii2 = ~mask
                    frac_per_knot = min(frac_per_knot, np.sum(ii1&ii2)/np.sum(ii1))
                if frac_per_knot > min_frac_per_knot: break
                knot_spacing += 2
                print(f"  {frac_per_knot:.2f} < {min_frac_per_knot} increasing knot_spacing to {knot_spacing}")
            except Exception as e:
                print(f"Could not succeed in checking knots, stopping at it={it+1}")
                print(e)
        kwds["knot_spacing"] = knot_spacing
        
        
        try:
            norm, cont, left, right = spec.fit_continuum(**kwds)
        except:
            print(f"Failed to normalize order {i} ({spec.dispersion[0]:.0f}-{spec.dispersion[-1]:.0f})")
            norm = np.nan
            cont = np.nan
        norms[i] = norm
        conts[i] = cont
        all_kwds[i] = kwds
    session.metadata["normalization"]["continuum"] = conts
    session.metadata["normalization"]["normalization_kwargs"] = all_kwds
    
    session.stitch_and_stack()
    session.add_to_notes(notes)

    ## Finish
    session.save(out_fname, overwrite=True)
    print(f"Total time to run all: {time.time()-startall:.1f}")

    if ncfg["save_figure"]:
        figoutname = os.path.join(figdir, f"{name}_norm.png")
        print(f"Saving figure to {figoutname}")
        start = time.time()
        plot_norm_one_session(session, figoutname)
        print(f"Time to save figure: {time.time()-start:.1f}")
