import numpy as np
import sys, os, time
import yaml
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d

from LESSPayne.smh import Session, utils
from LESSPayne.specutils import Spectrum1D 
from LESSPayne.smh.spectral_models import ProfileFittingModel, SpectralSynthesisModel
from LESSPayne.smh.photospheres.abundances import asplund_2009 as solar_composition
from LESSPayne.PayneEchelle.spectral_model import DefaultPayneModel

from .plotting import plot_summary_1, plot_summary_2, plot_model_fit, get_line_table, plot_fe_trends

def run_eqw_fit(cfg):
    name = cfg["output_name"]
    NNpath = cfg["NN_file"]
    outdir = cfg["output_directory"]
    figdir = cfg["figure_directory"]
    if not os.path.exists(outdir): os.makedirs(outdir)
    if not os.path.exists(figdir): os.makedirs(figdir)
    print("Saving to output directory:",outdir)
    print("Saving figures to output directory:",figdir)

    smh_fname = os.path.join(outdir, cfg["smh_fname"])
    
    ecfg = cfg["run_eqw_fit"]
    max_fwhm = ecfg["max_fwhm"]
    linelist_fname = ecfg["eqw_linelist_fname"]
    linelist_extra_fname = ecfg["extra_eqw_linelist_fname"]
    if ecfg.get("output_suffix") is None:
        smh_outfname = smh_fname
    else:
        smh_outfname = smh_fname.replace(".smh", ecfg["output_suffix"]+".smh")
    print(f"Reading from {smh_fname}, writing to {smh_outfname}")
    if smh_fname == smh_outfname:
        print("(Overwriting the file)")

    clear_all_existing_fits = ecfg["clear_all_existing_fits"]
    
    startall = time.time()
    
    ## Load results of normalization
    session = Session.load(smh_fname)
    all_exclude_regions, all_exclude_regions_2, norm_params = session.metadata["payne_masks"]

    if clear_all_existing_fits:
        old_models = session.spectral_models
        if len(old_models) > 0:
            print(f"Deleting {len(old_models)} previously existing spectral models")
        for model in old_models: del model
        session.metadata["spectral_models"] = []

    ## Step 5: create linelist with masks
    session.import_linelist_as_profile_models(linelist_fname)
    for model in session.spectral_models:
        w0 = model.wavelength
        w1, w2 = w0 - model.metadata["window"], w0 + model.metadata["window"]
        # keep if within w1 and w2 AND w0 not in region
        this_exclude = [x for x in all_exclude_regions_2 if (x[1] > w1 and x[0] < w2) and ~(w0 > x[0] and w0 < x[1])]
        model.metadata["mask"] = this_exclude
        try:
            model.fit()
        except:
            print(f"failed on species={model.species} wave={model.wavelength}")
    print(f"Time to add masks to normalizations and import models: {time.time()-startall:.1f}")
    
#    ## Step 6: some quality control
#    for model in session.spectral_models:
#        if not model.is_acceptable: continue
#        if (model.reduced_equivalent_width > -4.5) or (model.reduced_equivalent_width < -5.5):
#            model.is_acceptable = False
#    
    
    num_removed = 0
    for model in session.spectral_models:
        if model.is_acceptable and (model.fwhm > max_fwhm):
            model.is_acceptable = False
            model.user_flag = True
            num_removed += 1
    if num_removed > 0:
        print(f"Removed {num_removed}/{len(session.spectral_models)} lines with FWHM > {max_fwhm}")

    
    notes = f"run_eqw_fit:\n  {linelist_fname}"
    if linelist_extra_fname is not None:
        session.import_linelist_as_profile_models(linelist_extra_fname)
        notes += f"\n{linelist_extra_fname} imported but not fit"
    session.add_to_notes(notes)

    ## Save
    session.save(smh_outfname, overwrite=True)
    print(f"Total time to run all: {time.time()-startall:.1f}")

    ## Plot
    if ecfg["save_figure"]:
        figoutname = os.path.join(figdir, f"{name}_eqw.png")
        start = time.time()
        plot_eqw_grid(session, figoutname, name)
        print(f"Time to save figure: {time.time()-start:.1f}")
        

def plot_eqw_grid(session, outfname, name,
                  Ncol=5, width=6, height=4, dpi=150):
    import matplotlib.pyplot as plt
    eqw_models = [m for m in session.spectral_models if isinstance(m, ProfileFittingModel)]
    N = len(eqw_models) + 4
    Nrow = (N // Ncol) + int(N % Ncol > 0)
    assert Nrow*Ncol >= N, (Nrow,Ncol,N)
    
    ltab = get_line_table(session, True)
    
    fig, axes = plt.subplots(Nrow, Ncol, figsize=(width*Ncol, height*Nrow))
    plot_summary_1(axes.flat[0], session, ltab, name)
    plot_summary_2(axes.flat[1], session, ltab)
    for model, ax in zip(eqw_models, axes.flat[2:]):
        linewave = model.transitions[0]["wavelength"]
        label = f"{utils.species_to_element(model.species[0]).replace(' ','')}{model.wavelength:.0f}"
        try:
            plot_model_fit(ax, session, model, label=label,
                           linewave=linewave, inset_dwl=None)
        except Exception as e:
            print("Error:",model.species,model.wavelength)
            print(e)
            print("Skipping...")
    plot_fe_trends(axes.flat[-2], session, "expot", ltab)
    plot_fe_trends(axes.flat[-1], session, "REW", ltab)
    
    fig.tight_layout()
    fig.subplots_adjust(top=.96, wspace=0.15, hspace=.15)
    fig.suptitle(name, fontsize=30, weight='bold', fontfamily='monospace',color='k')
    fig.savefig(outfname, dpi=dpi)
    plt.close(fig)
