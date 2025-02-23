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

def run_errors(cfg):
    name = cfg["output_name"]
    NNpath = cfg["NN_file"]
    outdir = cfg["output_directory"]
    figdir = cfg["figure_directory"]
    if not os.path.exists(outdir): os.makedirs(outdir)
    if not os.path.exists(figdir): os.makedirs(figdir)
    print("Saving to output directory:",outdir)
    print("Saving figures to output directory:",figdir)
    
    smh_fname = os.path.join(outdir, cfg["smh_fname"])
    
    errcfg = cfg["run_errors"]
    print(errcfg)
    
    if errcfg.get("output_suffix") is None:
        smh_outfname = smh_fname
    else:
        smh_outfname = smh_fname.replace(".smh", errcfg["output_suffix"]+".smh")
    print(f"Reading from {smh_fname}, writing to {smh_outfname}")
    if smh_fname == smh_outfname:
        print("(Overwriting the file)")

    startall = time.time()
    
    session = Session.load(smh_fname)
    
    ## Set default errors or override
    if errcfg["e_Teff"] is not None:
        e_Teff = int(errcfg["e_Teff"])
        print(f"Setting e_Teff={e_Teff}")
    else:
        e_Teff = 50
    if errcfg["e_logg"] is not None:
        e_logg = round(float(errcfg["e_logg"]),2)
        print(f"Setting e_logg={e_logg}")
    else:
        e_logg = 0.1
    if errcfg["e_vt"] is not None:
        e_vt = round(float(errcfg["e_vt"]),2)
        print(f"Setting e_vt={e_vt}")
    else:
        e_vt = 0.1
    if errcfg["e_MH"] is not None:
        e_MH = round(float(errcfg["e_MH"]),2)
        print(f"Setting e_MH={e_MH}")
    else:
        e_MH = 0.1
    
    #print(f"Final stellar parameters: T/g/v/M/a = {Teff:.0f}/{logg:.2f}/{vt:.2f}/{MH:.2f}/{aFe:.2f}")
    #session.set_stellar_parameters(Teff, logg, vt, MH, aFe)
    
    if errcfg["calculate_stellar_params_spectroscopic_uncertainties"]:
        print("Calculating stellar parameter uncertainties from spectroscopic analysis")
        session.stellar_parameter_uncertainty_analysis(systematic_errors=[e_Teff, e_logg, e_vt, e_MH])
    else:
        session.set_stellar_parameters_errors("stat", 0, 0, 0, 0)
        session.set_stellar_parameters_errors("sys", e_Teff, e_logg, e_vt, e_MH)
    
    etot_Teff, etot_logg, etot_vt, etot_MH = session.stellar_parameters_err
    notes = f"run_errors: e_[T/g/v/M] = {etot_Teff:.0f}/{etot_logg:.2f}/{etot_vt:.2f}/{etot_MH:.2f}"
    print(notes)
    
    ## Propagate abundance uncertainties
    session.compute_all_abundance_uncertainties(print_memory_usage=True)
    
    ## Save
    session.add_to_notes(notes)
    session.save(smh_outfname, overwrite=True)
    print(f"Total time run_errors: {time.time()-startall:.1f}")
    
