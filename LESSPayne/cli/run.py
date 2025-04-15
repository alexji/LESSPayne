from optparse import OptionParser
import yaml
import os, sys, time

from LESSPayne.PayneEchelle.run_payne_echelle import run_payne_echelle
from LESSPayne.autosmh.run_normalization import run_normalization
from LESSPayne.autosmh.run_eqw_fit import run_eqw_fit
from LESSPayne.autosmh.run_stellar_parameters import run_stellar_parameters
from LESSPayne.autosmh.run_synth_fit import run_synth_fit
from LESSPayne.autosmh.run_errors import run_errors
from LESSPayne.autosmh.run_summary import run_summary


if __name__=="__main__":
    start = time.time()
    
    parser = OptionParser()
    # Flags indicating which parts of the pipeline to run
    parser.add_option("-a", "--all", dest="run_all", action="store_true", default=False)
    parser.add_option("-1", "--payne", dest="run_payneechelle", action="store_true", default=False)
    parser.add_option("-2", "--norm", dest="run_normalization", action="store_true", default=False)
    parser.add_option("-3", "--eqw", dest="run_equivalent_width", action="store_true", default=False)
    parser.add_option("-4", "--params", dest="run_stellar_parameters", action="store_true", default=False)
    parser.add_option("-5", "--synth", dest="run_synthesis", action="store_true", default=False)
    parser.add_option("-6", "--resynth", dest="run_synthesis_resynth", action="store_true", default=False)
    parser.add_option("-7", "--errors", dest="run_errors", action="store_true", default=False)
    parser.add_option("-8", "--summary", dest="run_summary", action="store_true", default=False)
    
    (options, args) = parser.parse_args()
    
    if options.run_all:
        print("Running all phases of LESSPayne")
        options.run_payneechelle = True
        options.run_normalization = True
        options.run_equivalent_width = True
        options.run_stellar_parameters = True
        options.run_synthesis = True
        options.run_synthesis_resynth = False # no need to resynth
        options.run_errors = True
    
    cfg_file = args[0]
    if not os.path.exists(cfg_file):
        sys.exit(f"input file '{cfg_file}' does not exist")
    with open(cfg_file, "rb") as fp:
        cfg = yaml.load(fp, yaml.FullLoader)
    print("Running cfg file:",cfg_file)
    print(cfg)
    ## set defaults
    ## TODO: we should write out a CFG file with the defaults filled in
    cfg.setdefault("NN_type","default")
    
    outdir = cfg["output_directory"]
    figdir = cfg["figure_directory"]
    if not os.path.exists(outdir):
        print("Creating output directory:",outdir)
        os.makedirs(outdir)
    if not os.path.exists(figdir):
        print("Creating figure directory:",figdir)
        os.makedirs(figdir)
    print("Saving to output directory:",outdir)
    print("Saving figures to output directory:",figdir)
    
    
    ## PayneEchelle
    # Input: config file, spectrum files to analyze
    # Output: payneechelle fit npz file
    # Optional output: figure for each order showing fit
    if options.run_payneechelle:
        run_payne_echelle(cfg)
    
    ## Normalization
    # Input: payneechelle fit, spectrum files to analyze
    # Output: SMHR file with normalization done, mask file
    # Optional output: order normalization stamps
    if options.run_normalization:
        run_normalization(cfg)
    
    ## EQW
    # Input: SMHR file with normalization done, mask file
    # Output: SMHR file with EQWs fit
    # Optional output: EQW stamp plot
    # Optional output: line abundance table
    if options.run_equivalent_width:
        run_eqw_fit(cfg)
    
    ## StellarParams
    # Input: SMHR file
    # Optional input: Payneechelle file
    # Optional input: manual stellar parameters
    # Output: SMHR file with stellar parameters added
    if options.run_stellar_parameters:
        run_stellar_parameters(cfg)
    
    ## Synth
    # Input: SMHR file with normalization and eqw done
    # Output: SMHR file with syntheses fit
    # Optional output: synth stamp plot
    # Optional output: line abundance table
    if options.run_synthesis:
        run_synth_fit(cfg, resynth=False)

    ## Resynth
    # Input: SMHR file with syntheses fit
    # Output: SMHR file with syntheses refit
    # TODO Optional output: synth stamp plot
    # TODO Optional output: line abundance table
    if options.run_synthesis_resynth:
        run_synth_fit(cfg, resynth=True)
    
    ## Errors
    # Input: SMHR file with all abundances you want done
    # Optional input: SP covariance matrix
    # Output: SMHR file with errors propagated
    # Optional output: 
    if options.run_errors:
        run_errors(cfg)

    ## Summary
    # Input: SMHR file with all abundances done
    # Output: line and abundance tables
    if options.run_summary:
        run_summary(cfg)

    print(f"Time to run LESSPayne: {time.time()-start:.1f}")
