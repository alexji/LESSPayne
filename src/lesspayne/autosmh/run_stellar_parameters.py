import numpy as np
import logging
import sys, os, time
from timeit import default_timer as timer
import yaml
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d

from lesspayne.smh import Session
from lesspayne.specutils import Spectrum1D
from lesspayne.smh.spectral_models import ProfileFittingModel, SpectralSynthesisModel
from lesspayne.smh.photospheres.abundances import asplund_2009 as solar_composition
from lesspayne.PayneEchelle.spectral_model import DefaultPayneModel, YYLiPayneModel

from .run_eqw_fit import plot_eqw_grid


def run_stellar_parameters(cfg):
    name = cfg["output_name"]
    NNpath = cfg["NN_file"]
    NNtype = cfg["NN_type"]
    outdir = cfg["output_directory"]
    figdir = cfg["figure_directory"]
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    print("Saving to output directory:", outdir)
    print("Saving figures to output directory:", figdir)

    popt_fname = os.path.join(outdir, cfg["payne_fname"])
    smh_fname = os.path.join(outdir, cfg["smh_fname"])

    spcfg = cfg["run_stellar_parameters"]
    #print(spcfg)
    if spcfg["method"] not in ["rpa_calibration", "manual_all"]:
        raise ValueError(
            f"run_stellar_parameters method={spcfg['method']} is not valid\n(rpa_calibration, manual_all)"
        )

    if spcfg.get("output_suffix") is None:
        smh_outfname = smh_fname
    else:
        smh_outfname = smh_fname.replace(".smh",
                                         spcfg["output_suffix"] + ".smh")
    print(f"Reading from {smh_fname}, writing to {smh_outfname}")
    if smh_fname == smh_outfname:
        print("(Overwriting the file)")

    startall = time.time()

    ## Load results of normalization
    session = Session.load(smh_fname)

    ## Step 7: initialize stellar parameters
    if spcfg["method"] == "rpa_calibration":
        if NNtype == "default":
            model = DefaultPayneModel.load(NNpath, 1)
            with np.load(popt_fname) as tmp:
                popt_best = tmp["popt_best"].copy()
                popt_print = tmp["popt_print"].copy()
            Teff, logg, MH, aFe = round(popt_print[0]), round(
                popt_print[1], 2), round(popt_print[2],
                                         2), round(popt_print[3], 2)
            outstr1 = f"run_stellar_parameters:\n  PayneEchelle {NNpath}: T/g/v/M/a = {Teff}/{logg}/1.00/{MH}/{aFe}"

            ## Corrections from empirical fit to RPA duplicates
            dT = -.0732691466 * Teff + 247.57
            dg = 8.11486e-5 * logg - 0.28526
            dM = -0.06242672 * MH - 0.3167661
            Teff, logg, MH = int(Teff - dT), round(logg - dg,
                                                   2), round(MH - dM, 2)

            ## TODO
            #if aFe > 0.2: aFe = 0.4
            #else: aFe = 0.0
            aFe = 0.4

            ## TODO offer different vt methods
            #vt = round(2.13 - 0.23 * logg,2) # kirby09
            vt = round(0.060 * logg**2 - 0.569 * logg + 2.585,
                       2)  # RPA duplicates

            outstr2 = f"  Calibrated = {Teff}/{logg}/{vt}/{MH}/{aFe}"
            session.add_to_notes(outstr1 + "\n" + outstr2)
        elif NNtype == "yyli":
            model = DefaultPayneModel.load(NNpath, 1)
            with np.load(popt_fname) as tmp:
                popt_best = tmp["popt_best"].copy()
                popt_print = tmp["popt_print"].copy()
            Teff, logg, vt, MH = round(popt_print[0]), round(
                popt_print[1], 2), round(popt_print[2],
                                         2), round(popt_print[3], 2)
            CFe, MgFe, CaFe, TiFe = round(popt_print[4]), round(
                popt_print[5], 2), round(popt_print[6],
                                         2), round(popt_print[7], 2)
            aFe = round((MgFe + CaFe + TiFe) / 3., 2)
            outstr1 = f"run_stellar_parameters:\n  PayneEchelleYYLi {NNpath}: T/g/v/M = {Teff}/{logg}/{vt}/{MH}"
            outstr2 = f"  PayneEchelleYYLi {NNpath}: CFe/MgFe/CaFe/TiFe = {CFe}/{MgFe}/{CaFe}/{TiFe} (setting aFe=0.4 by default)"
            aFe = 0.4  # TODO
            session.add_to_notes(outstr1 + "\n" + outstr2)

    elif spcfg["method"] == "manual_all":
        Teff = logg = vt = MH = aFe = None

    ## Override manual
    if spcfg["manual_Teff"] is not None:
        Teff = int(spcfg["manual_Teff"])
        print(f"Setting Teff={Teff}")
    if spcfg["manual_logg"] is not None:
        logg = round(float(spcfg["manual_logg"]), 2)
        print(f"Setting logg={logg}")
    if spcfg["manual_vt"] is not None:
        vt = round(float(spcfg["manual_vt"]), 2)
        print(f"Setting vt={vt}")
    if spcfg["manual_MH"] is not None:
        MH = round(float(spcfg["manual_MH"]), 2)
        print(f"Setting MH={MH}")
    if spcfg["manual_aFe"] is not None:
        aFe = round(float(spcfg["manual_aFe"]), 2)
        print(f"Setting aFe={aFe}")

    if Teff is None: raise ValueError("Teff is not specified")
    if logg is None: raise ValueError("logg is not specified")
    if vt is None: raise ValueError("vt is not specified")
    if MH is None: raise ValueError("MH is not specified")
    if aFe is None: raise ValueError("aFe is not specified")

    if (spcfg['manual_vt'] is None) & (spcfg['solve_vt']):
        session = optimize_vt(session, spcfg['solve_vt_settings'])

    print(
        f"Final stellar parameters: T/g/v/M/a = {Teff:.0f}/{logg:.2f}/{vt:.2f}/{MH:.2f}/{aFe:.2f}"
    )
    session.set_stellar_parameters(Teff, logg, vt, MH, aFe)

    if spcfg["measure_eqw_abundances"]:
        print("Measuring EQW abundances")
        session.measure_abundances()  # eqw

    ## Save
    session.save(smh_outfname, overwrite=True)
    print(f"Total time run_stellar_parameters: {time.time()-startall:.1f}")

    ## Plot
    if spcfg["save_figure_eqw"]:
        figoutname = os.path.join(figdir, f"{name}_eqw.pdf")
        start = time.time()
        plot_eqw_grid(session, figoutname, name)
        print(f"Time to save eqw figure: {time.time()-start:.1f}")


def optimize_vt(session: Session,
                numiter: int = 8,
                feh_tol: float = 0.001,
                vt_condition: int = 1,
                aFe: float = 0.4,
                verbose: bool = False,
                **kwargs):
    """
    Enforce optimization of microturbulence for set stellar parameters given that [M/H] = [Fe I /H]. 
    Will run for an acceptable number of iterations before lowering precision of [M/H] = [Fe I/H]. 
    Will completely stop when tolerance hits 0.1 and the max acceptable number of iterations is reached
    
    Author: Riley Thai

    params:
        session :: LESSPayne/SMHR Session instance
        numiter :: number of acceptable iterations before dropping the tolerance
        feh_tol :: starting tolerance for proximity; note that SMHR only shows to 2 decimals.
        vt_condition :: whether to minimize dA/dREW for Fe I or Fe II or both ( {1,2,3} ) [TODO]
        aFe :: alpha abundance to use for Kurucz atmospheres, defaults to [a/Fe] = 0.4
        verbose :: whether to send debug statements to stdout/console
        **kwargs :: keywords overload to Session.optimize_feh
    """
    # error check inputs
    assert np.log10(
        feh_tol) % 1 == 0, 'feh_tol must be a power of 10 (i.e. 0.01, 1e-3)'

    if vt_condition != 1:
        raise NotImplementedError  # im lazy

    # setup logger
    # TODO: for alex, do you want to have a LESSPayne logger instance?
    logger = logging.getLogger('vt_optimizer')

    # measure eqws for current parameters
    session.measure_abundances()

    # Check if the logger already has handlers (to avoid duplicates)
    if not logger.hasHandlers():
        # Create a console handler TODO: support fileout, or just let user handle it?
        console_handler = logging.StreamHandler()

        # Set the log level based on the verbose flag
        if verbose:
            console_handler.setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)
            logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

    start = timer()
    logger.info('Running vt optimization...')
    Teff, logg, vt, MH = session.stellar_parameters
    logger.info(
        'Starting parameters are' +
        "Teff = {0:.0f} K, vt = {1:.2f} km/s, logg = {2:.2f}, [M/H] = {3:.2f}".
        format(Teff, vt, logg, MH))

    feh_tol = int(np.log10(1 / feh_tol))
    for i in range(feh_tol * numiter + 1):
        if (i % numiter == 0) & (i != 0):
            # if num acceptable reached for this tolerance
            if feh_tol == 1:
                # fail case!
                # TODO: alex needs to decide whether this should crash or return as is
                logger.error(
                    'failed to unify [M/H] = [Fe I/H]! are the stellar parameters okay?'
                )
                #raise RuntimeError(
                #    'failed to unify [M/H] = [Fe I/H]! are the stellar parameters okay?'
                #)
                return session
            else:
                logging.info(
                    f'Failed to converge in {i+1} iterations, decreasing tolerance to {feh_tol - 1} d.p.'
                )
                feh_tol -= 1

        # get current params + Fe I abundance
        Teff, logg, vt, MH = session.stellar_parameters
        abundances = session.summarize_spectral_models()

        # NOTE: solar values are Asplund+2009
        feh = abundances[26.0][1] - solar_composition(26.0)

        ## Exists if current [M/H] = [Fe I /H]
        logger.debug('Current [Fe I / H] =', feh, ', MH =', MH)
        if np.round(feh, feh_tol) == np.round(float(MH), feh_tol):
            logger.debug('Reached equilibrium')
            break

        # set and run optimizer
        session.set_stellar_parameters(Teff, logg, vt, feh, alpha=aFe)

        # BUG: when run, may sometime have size mismatch -- is this the MOOG failcase?
        try:
            # NOTE: holmbeck made order Teff, vt, logg, MH in the optimizer
            session.optimize_feh(np.array([False, True, False, False]),
                                 **kwargs)
        except ValueError:
            # except a strange bug that seems to occur on holmbeck's routine
            logger.debug('MOOG/Holmbeck bug encountered')
            continue

        session.measure_abundances()  # remeasure on new parameters

    # show user the current stellar parameters
    Teff, logg, vt, MH = session.stellar_parameters
    logger.info(
        'Final parameters are' +
        "Teff = {0:.0f} K, vt = {1:.2f} km/s, logg = {2:.2f}, [M/H] = {3:.2f}".
        format(Teff, vt, logg, MH))
    logger.info(f'vt solver took {timer() - start:.4f} seconds')

    return session
