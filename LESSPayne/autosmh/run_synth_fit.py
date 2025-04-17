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

from .plotting import plot_summary_1, plot_summary_2, plot_model_fit, plot_model_resid, get_line_table

def update_abundance_table(session, model):
    """ Set [X/M] = 0 by default """
    assert isinstance(model, SpectralSynthesisModel)
    Teff, logg, vt, MH = session.stellar_parameters
    summary_dict = session.summarize_spectral_models(organize_by_element=True)
    for elem in model.metadata["rt_abundances"]:
        try:
            model.metadata["rt_abundances"][elem] = summary_dict[elem][1]
        except KeyError:
            model.metadata["rt_abundances"][elem] = solar_composition(elem) + MH

    # Fill in fixed abundances
    try:
        fitted_result = model.metadata["fitted_result"]
    except KeyError:
        #logger.info("Run at least one fit before setting abundances of "
        #      "fitted element {}!".format(elem))
        pass
    else:
        for i,elem in enumerate(model.elements):
            try:
                abund = summary_dict[elem][1]
            except KeyError:
                #logger.warn("No abundance found for {}, using nan".format(elem))
                abund = np.nan
            key = "log_eps({})".format(elem)
            fitted_result[0][key] = abund
            fitted_result[2]["abundances"][i] = abund

def update_abundance_table_2(session, model):
    """
    Set some more sophisticated abundances after at least one iteration
    Also now use [Fe I/H] as MH
    """
    assert isinstance(model, SpectralSynthesisModel)
    Teff, logg, vt, MH = session.stellar_parameters
    summary_dict = session.summarize_spectral_models(organize_by_element=True)
    
    ## Use Fe I instead of MH now!
    #FeH = summary_dict["Fe"][3]
    #if np.isfinite(FeH): MH = FeH

    # Assume that we already have Mg and Eu, use the model metallicity for Fe everywhere
    try:
        MgM = summary_dict["Mg"][4] - MH
        MgM = max(min(MgM, 0.4), -0.4)
    except:
        MgM = 0.
    try:
        EuM = summary_dict["Eu"][4] - MH
    except:
        EuM = 0.
    if np.isnan(MgM): MgM = 0
    if np.isnan(EuM): EuM = 0
    # [X/Eu] from Sneden+08 for the r-process
    XEudict = {
        "Sr":-0.95, "Y":-0.54,"Zr":-0.71,
        "Nb":-0.48,"Mo":-0.47,"Ru":-0.18,
        "Rh":-0.07,"Ba":-0.83,"La":-0.60,
        "Ce":-0.71,"Pr":-0.28,"Nd":-0.36,
        "Sm":-0.14,"Gd":-0.07,"Tb":-0.02,
        "Dy":-0.05,"Ho":-0.02,"Er":-0.06,
        "Tm":-0.07,"Yb":-0.16,"Lu":-0.08,
        "Hf":-0.27,"Os":-0.03,"Ir": 0.00,
    }

    for elem in model.metadata["rt_abundances"]:
        try:
            model.metadata["rt_abundances"][elem] = summary_dict[elem][1]
        except KeyError:
            if elem in ["O","Si","Ca","Ti"]:
                XM = MgM
            elif elem == "Cr":
                XM = 0.144 + 0.129 * MH
            elif elem == "Mn":
                XM = -0.234 + 0.096 * MH
            elif elem in XEudict:
                XM = EuM + XEudict[elem]
            else:
                XM = 0
            model.metadata["rt_abundances"][elem] = solar_composition(elem) + MH + XM

    # Fill in fixed abundances
    try:
        fitted_result = model.metadata["fitted_result"]
    except KeyError:
        #logger.info("Run at least one fit before setting abundances of "
        #      "fitted element {}!".format(elem))
        pass
    else:
        for i,elem in enumerate(model.elements):
            try:
                abund = summary_dict[elem][1]
            except KeyError:
                #logger.warn("No abundance found for {}, using nan".format(elem))
                abund = np.nan
            key = "log_eps({})".format(elem)
            fitted_result[0][key] = abund
            fitted_result[2]["abundances"][i] = abund

def run_synth_fit(cfg):
    name = cfg["output_name"]
    NNpath = cfg["NN_file"]
    outdir = cfg["output_directory"]
    figdir = cfg["figure_directory"]
    if not os.path.exists(outdir): os.makedirs(outdir)
    if not os.path.exists(figdir): os.makedirs(figdir)
    print("Saving to output directory:",outdir)
    print("Saving figures to output directory:",figdir)

    smh_fname = os.path.join(outdir, cfg["smh_fname"])
    
    scfg = cfg["run_synth_fit"]
    max_fwhm = scfg["max_fwhm"]
    synthesis_fname = scfg["synthesis_linelist_fname"]
    synthesis_fname_extra = scfg["extra_synthesis_linelist_fname"]
    numiter = scfg["num_iter_all"]
    smooth_approx = scfg["smooth_approx"]
    smooth_scale = scfg["smooth_scale"]
    maxiter = scfg["max_iter_each"]
    synth_mask_frac = scfg.get("synth_mask_frac",0.8) # TODO
    
    if scfg.get("output_suffix") is None:
        smh_outfname = smh_fname
    else:
        smh_outfname = smh_fname.replace(".smh", scfg["output_suffix"]+".smh")
    print(f"Reading from {smh_fname}, writing to {smh_outfname}")
    if smh_fname == smh_outfname:
        print("(Overwriting the file)")

    notes = "run_synth_fit:\n"
    keys = ["max_fwhm","synthesis_linelist_fname","extra_synthesis_linelist_fname",
            "num_iter_all","smooth_approx","smooth_scale","max_iter_each"]
    for key in keys:
        notes += f"  {key}: {scfg[key]}\n"
    notes = notes[:-1]
    
    startall = time.time()
    
    ## Load results of normalization
    session = Session.load(smh_fname)
    
    if scfg["clear_all_existing_syntheses"]:
        synthesis_models = [x for x in session.spectral_models if isinstance(x, SpectralSynthesisModel)]
        print(f"clear_all_existing_syntheses: removing {len(synthesis_models)} pre-existing syntheses")
        for m in synthesis_models: session.metadata["spectral_models"].remove(m)
    
    ## Step 8: measure abundances
    session.measure_abundances() # eqw
    ## Run syntheses
    if synthesis_fname is not None:
        print(f"Importing {synthesis_fname}")
        num_added = session.import_master_list(synthesis_fname)
        
    
    for it in range(numiter):
        print("===========================================")
        print("======Running synthesis iter {}=============".format(it+1))
        print("===========================================")
        for model in session.spectral_models:
            if isinstance(model, ProfileFittingModel): continue
            if model.user_flag:
                print("Skipping {} {} due to flag".format(model.wavelength, model.species))
                continue
            print("Fitting {} {}".format(model.wavelength, model.species))
            update_abundance_table(session, model)
            smoothing_index = model.parameter_names.index("sigma_smooth")
            def penalty_function(params):
                return ((params[smoothing_index] - smooth_approx)/smooth_scale)**2
            try:
                model.iterfit(maxiter=maxiter, penalty_function=penalty_function)
            except Exception as e:
                print("Failed on this one, trying without prior")
                print(e)
                try:
                    model.iterfit(maxiter=maxiter)
                except Exception as e:
                    print("Failed on this one again")
                    model.is_acceptable = False
                    model.user_flag = True
                continue
    # Reiterate fitting all syntheses, based on [Mg/M], [Eu/M]
    for it in range(numiter):
        print("===========================================")
        print("======Running synthesis iter {}=============".format(it+1+numiter))
        print("===========================================")
        for model in session.spectral_models:
            if isinstance(model, ProfileFittingModel): continue
            if model.user_flag:
                print("Skipping {} {} due to flag".format(model.wavelength, model.species))
                continue
            print("Fitting {} {}".format(model.wavelength, model.species))
            update_abundance_table_2(session, model)
            try:
                model.iterfit(maxiter=maxiter)
            except Exception as e:
                print("Failed on this one")
                print(e)
                model.is_acceptable = False
                model.user_flag = True
                continue
    # Check for detections in final fits
    for model in session.spectral_models:
        if isinstance(model, SpectralSynthesisModel):
            threesigma_detection, delta_chi2 = model.check_line_detection(sigma=3)
            if not threesigma_detection:
                model.is_acceptable = False
                model.user_flag = True
                print("Model {} {} detected at only deltachi={:.1f}, finding UL".format(model.wavelength, model.species, delta_chi2))
                try:
                    model.find_upper_limit(sigma=5)
                except:
                    print("Failed to find upper limit")
    
    # Do a FWHM sanity check
    num_removed = 0
    for model in session.spectral_models:
        if model.is_acceptable and (model.fwhm > max_fwhm):
            model.is_acceptable = False
            model.user_flag = True
            num_removed += 1
    if num_removed > 0:
        print(f"Removed {num_removed}/{len(session.spectral_models)} lines with FWHM > {max_fwhm}")
    
    if synthesis_fname_extra is not None:
        print("Importing extra synthesis master list")
        session.import_master_list(synthesis_fname_extra)
    
    session.add_to_notes(notes)

    ## Save
    session.save(smh_outfname, overwrite=True)
    print(f"Total time run_synth_fit: {time.time()-startall:.1f}")

    ## Plot
    if scfg["save_figure"]:
        figoutname = os.path.join(figdir, f"{name}_synth.png")
        start = time.time()
        plot_synth_grid(session, figoutname, name)
        print(f"Time to save figure: {time.time()-start:.1f}")

def plot_synth_grid(session, outfname, name,
                    Ncol=3, width=10, height=4, dpi=150,
                    inset_dwl=1):
    ## Plots all the residuals underneath the fits
    import matplotlib.pyplot as plt
    syn_models = [m for m in session.spectral_models if isinstance(m, SpectralSynthesisModel)]
    N = len(syn_models) + 1
    Nrow = (N // Ncol) + int(N % Ncol > 0) # Number of rows
    Nrow = Nrow*2
    assert Nrow*Ncol >= 2*N, (Nrow,Ncol,N)
    
    ltab = get_line_table(session, True)
    
    fig, axes = plt.subplots(Nrow, Ncol, figsize=(width*Ncol, height*Nrow))
    plot_summary_1(axes[0,0], session, ltab, name)
    plot_summary_2(axes[1,0], session, ltab)
    irow = 0
    icol = 1
    for i, model in enumerate(syn_models):
        ax1 = axes[irow*2,   icol]
        ax2 = axes[irow*2+1, icol]
        species = model.species
        for j in range(5):
            species = species[0]
            if isinstance(species, float): break
        label = f"{utils.species_to_element(species).replace(' ','')}{model.wavelength:.0f}"
        
        try:
            plot_model_fit(ax1, session, model, label=label,
                           linewave = model.wavelength, inset_dwl=inset_dwl)
            plot_model_resid(ax2, session, model, label=label,
                             linewave = model.wavelength)
        except Exception as e:
            print("Error:",model.species,model.wavelength)
            print(e)
            print("Skipping...")
            
        icol += 1
        if icol == Ncol:
            icol = 0
            irow += 1
    
    fig.tight_layout()
    fig.subplots_adjust(top=.96, wspace=0.15, hspace=.15)
    fig.suptitle(name, fontsize=30, weight='bold', fontfamily='monospace',color='k')
    fig.savefig(outfname, dpi=dpi)
    plt.close(fig)
