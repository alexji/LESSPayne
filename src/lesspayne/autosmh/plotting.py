import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import numpy as np

import os, sys, time
from optparse import OptionParser
from collections import OrderedDict

from astropy.table import Table, vstack
from astropy.stats import biweight_scale, biweight_location

from lesspayne.smh import Session
from lesspayne.smh.spectral_models import ProfileFittingModel, SpectralSynthesisModel
from lesspayne.smh.photospheres.abundances import asplund_2009 as solar_composition

## for coloring the text black or red
scat_thresh = {
    12.0: 0.2,
    26.0: 0.2,
    26.1: 0.2,
    38.1: 0.3,
    56.1: 0.3,
    63.1: 0.3,
    106.0: 0.2,
}

def get_title_color(srfe, bafe, eufe):
    ## Colors taken from Erika Holmbeck
    srba = srfe - bafe
    baeu = bafe - eufe
    sreu = srfe - eufe
    if baeu < 0: # r-II or r-I
        if eufe > 0.7: return "r-II", "#D14793"
        if eufe > 0.3: return "r-I", "#56ae57" #"#006666"
    if (eufe <= 0.3) and (srba > 0.5) and (sreu > 0.0):
        # limited-r
        return "r-lim", "#79ECCF"
    if (baeu > 0.5) and (bafe > 1.0):
        # s-process
        return "s", "#FF4500"
    # Non-rpe
    return "", "#cccccc"
def get_line_table(session, get_all=False):
    cols = ["index","wavelength","species","expot","loggf",
            "logeps","e_stat","eqw","e_eqw","fwhm","is_acceptable","is_upper_limit",
            "redchi2","slope","slope_err"]
    data = OrderedDict(zip(cols, [[] for col in cols]))
    for i, model in enumerate(session.spectral_models):
        if (not model.is_acceptable) and (not get_all): continue
        if (model.is_upper_limit) and (not get_all): continue
        wavelength = model.wavelength
        species = np.ravel(model.species)[0]
        expot = model.expot
        loggf = model.loggf
        if np.isnan(expot) or np.isnan(loggf):
            print(i, species, model.expot, model.loggf)
        if model.abundances is None:
            logeps = np.nan
        else:
            logeps = model.abundances[0]
        fwhm = model.fwhm or np.nan
        e_stat = np.nan # figure this out later
        if isinstance(model, ProfileFittingModel):
            eqw = model.equivalent_width or np.nan
            e_eqw = model.equivalent_width_uncertainty or np.nan
        else:
            eqw = -999
            e_eqw = -999
        slope, slope_err = model.residual_slope_and_err
        input_data = [i, wavelength, species, expot, loggf,
                      logeps, e_stat, eqw, e_eqw, fwhm,
                      model.is_acceptable, model.is_upper_limit,
                      model.reduced_chi2,slope, slope_err]
        for col, x in zip(cols, input_data):
            data[col].append(x)
    t = Table(data)
    for col in ["eqw","e_eqw"]: t[col].format=".1f"
    for col in ["expot","loggf","redchi2"]: t[col].format=".2f"
    for col in ["logeps","fwhm"]: t[col].format=".3f"
    for col in ["slope","slope_err"]: t[col].format=".4f"
    return t

def get_feh_values(tab):
    try:
        x = np.array(tab[tab["species"]==26.0]["logeps"])
        feh1 = biweight_location(x) - epsfe_sun
        efeh1 = biweight_scale(x)
    except Exception as e:
        feh1 = np.nan
        efeh1 = np.nan
        print(e)
    try:
        x = np.array(tab[tab["species"]==26.1]["logeps"])
        feh2 = biweight_location(x) - epsfe_sun
        efeh2 = biweight_scale(x)
    except Exception as e:
        feh2 = np.nan
        efeh2 = np.nan
        print(e)
    return feh1, efeh1, feh2, efeh2

def plot_spectrum(spec, wlmin=None, wlmax=None, ax=None,
                  dxmaj=None, dxmin=None, dymaj=None, dymin=None,
                  fillcolor="#cccccc",fillalpha=1,
                  **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    wave = spec.dispersion
    flux = spec.flux
    errs = spec.ivar**-0.5
    ii = np.ones(len(wave), dtype=bool)
    if wlmin is not None:
        ii = ii & (wave > wlmin)
    if wlmax is not None:
        ii = ii & (wave < wlmax)

    wave = wave[ii]
    flux = flux[ii]
    errs = errs[ii]
    y1 = flux-errs
    y2 = flux+errs

    fill_between_steps(ax, wave, y1, y2, alpha=fillalpha, facecolor=fillcolor, edgecolor=fillcolor)
    ax.plot(wave, flux, **kwargs)

    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    if dxmaj is not None: ax.xaxis.set_major_locator(MultipleLocator(dxmaj))
    if dxmin is not None: ax.xaxis.set_minor_locator(MultipleLocator(dxmin))
    if dymaj is not None: ax.yaxis.set_major_locator(MultipleLocator(dymaj))
    if dymin is not None: ax.yaxis.set_minor_locator(MultipleLocator(dymin))
    return ax
def fill_between_steps(ax, x, y1, y2=0, h_align='mid', **kwargs):
    """
    Fill between for step plots in matplotlib.

    **kwargs will be passed to the matplotlib fill_between() function.
    """

    # First, duplicate the x values
    xx = x.repeat(2)[1:]
    # Now: the average x binwidth
    xstep = np.repeat((x[1:] - x[:-1]), 2)
    xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    # Now: add one step at end of row.
    xx = np.append(xx, xx.max() + xstep[-1])

    # Make it possible to chenge step alignment.
    if h_align == 'mid':
        xx -= xstep / 2.
    elif h_align == 'right':
        xx -= xstep

    # Also, duplicate each y coordinate in both arrays
    y1 = y1.repeat(2)#[:-1]
    if type(y2) == np.ndarray:
        y2 = y2.repeat(2)#[:-1]

    # now to the plotting part:
    return ax.fill_between(xx, y1, y2=y2, **kwargs)

def _get_rv(plotmodel, named_p_opt):
    if isinstance(plotmodel, SpectralSynthesisModel):
        try:
            rv = named_p_opt["vrad"]
        except:
            rv = plotmodel.metadata["manual_rv"]
    else:
        try:
            w0 = plotmodel.transitions[0]["wavelength"]
            w1 = named_p_opt["mean"]
            rv = 3e5 * (w1 - w0)/w0
        except:
            rv = np.nan
    return rv
            
            
def plot_model_fit(ax, session, imodel, label=None, linewave=None,
                   inset_dwl=None):
    """ Plot SMH model fit """
    if imodel is None: return
    try:
        plotmodel = session.spectral_models[imodel]
    except:
        plotmodel = imodel
    spectrum = session.normalized_spectrum
    if isinstance(plotmodel, SpectralSynthesisModel):
        transition1 = plotmodel.transitions[0]
        transition2 = plotmodel.transitions[-1]
        w1, w2 = transition1["wavelength"], transition2["wavelength"]
        species = plotmodel.species[0][0]
        wave = plotmodel.wavelength
        if np.isnan(wave):
            wave = round((w1+w2)/2,1)
    else:
        transition = plotmodel.transitions[0]
        w1, w2 = transition["wavelength"]-1, transition["wavelength"]+1
        species = transition["species"]
        wave = transition["wavelength"]
    
    plot_spectrum(spectrum, wlmin=w1, wlmax=w2, ax=ax, lw=3, color='k')
    try:
        named_p_opt, cov, meta = plotmodel.metadata["fitted_result"]
        ax.set(ylim=(0,1.1), xlim=(w1,w2))
        modelcolor = "r" if plotmodel.is_acceptable else "b"
        modelx, modely = meta["model_x"], meta["model_y"]
        ax.plot(modelx, modely, '-', lw=2, color=modelcolor)
        try:
            vrad = _get_rv(plotmodel, named_p_opt)
        except:
            vrad = np.nan
    except KeyError:
        modelx, modely = [np.nan], [np.nan]
        vrad = np.nan
        modelcolor = 'b'
    try:
        logeps = plotmodel.abundances[0]
    except:
        logeps = np.nan
    try:
        fwhm = plotmodel.fwhm
    except:
         fwhm = np.nan
    if label is not None:
        ax.text(.01,.01,f"A({label})={logeps:.2f} FWHM={fwhm:.2f}",ha='left',va='bottom',transform=ax.transAxes, fontsize=18, fontfamily='monospace')
    
    if linewave is not None:
        ax.axvline(linewave, color='grey', ls='-', lw=1)
        if inset_dwl is not None:
            axin = ax.inset_axes([0.75, 0.02, 0.24, 0.24])
            w1, w2 = linewave - inset_dwl/2, linewave + inset_dwl/2
            # set to the data range, expanding by 10% in each direction
            ii = (spectrum.dispersion > w1-0.1) & (spectrum.dispersion < w2+0.1)
            y1, y2 = np.nanpercentile(spectrum.flux[ii], [1,99])
            if np.isnan(y1) or np.isnan(y2): return # probably no data in here
            dy = y2-y1
            y1, y2 = y1 - dy*0.1, y2 + dy*0.1
            plot_spectrum(spectrum, wlmin=w1, wlmax=w2, ax=axin, lw=3, color='k')
            axin.plot(modelx, modely, '-', lw=2, color=modelcolor)
            axin.axvline(linewave, color='grey', ls='-', lw=1)
            axin.set_xlim(w1, w2)
            axin.set_ylim(y1, y2)
            axin.set_xticks([])
            axin.set_yticks([])
            ax.indicate_inset_zoom(axin)
    
def plot_model_resid(ax, session, imodel, label=None, linewave=None,
                     inset_dwl=None):
    """ Plot SMH model fit """
    if imodel is None: return
    try:
        plotmodel = session.spectral_models[imodel]
    except:
        plotmodel = imodel
    spectrum = session.normalized_spectrum
    if isinstance(plotmodel, SpectralSynthesisModel):
        transition1 = plotmodel.transitions[0]
        transition2 = plotmodel.transitions[-1]
        w1, w2 = transition1["wavelength"], transition2["wavelength"]
        species = plotmodel.species[0][0]
        wave = plotmodel.wavelength
        if np.isnan(wave):
            wave = round((w1+w2)/2,1)
    else:
        transition = plotmodel.transitions[0]
        w1, w2 = transition["wavelength"]-1, transition["wavelength"]+1
        species = transition["species"]
        wave = transition["wavelength"]
    try:
        fwhm = plotmodel.fwhm
    except:
        fwhm = np.nan
    try:
        redchi2 = plotmodel.reduced_chi2
    except:
        redchi2 = np.nan
    try:
        slope, slope_err = plotmodel.residual_slope_and_err
        slope_ratio = slope/slope_err
    except:
        redchi2 = slope_ratio = np.nan
    
    
    ax.axhline(0, c="#666666")
    ii = (spectrum.dispersion >= w1-0.2) & (spectrum.dispersion <= w2+0.2)
    sigma = 1.0 / np.sqrt(spectrum.ivar[ii])
    fill_between_steps(ax, spectrum.dispersion[ii], -sigma, sigma,
                       facecolor="#CCCCCC", edgecolor="none", alpha=1)
    finite = np.isfinite(sigma)
    if np.sum(finite) == 0:
        three_sigma = 1
    else:
        three_sigma = 3*np.median(sigma[finite])
        if not np.isfinite(three_sigma):
            three_sigma = np.nanmax(np.abs(sigma[finite]))*1.1
    ax.set_ylim(-three_sigma, three_sigma)
    try:
        named_p_opt, cov, meta = plotmodel.metadata["fitted_result"]
        plotx, ploty = meta["model_x"], meta["residual"]
    except KeyError:
        plotx, ploty = [np.nan], [np.nan]
    ax.plot(plotx, ploty, 'k-', lw=3, drawstyle='steps-mid')
    ax.set_xlim(w1, w2)
    
    if label is not None:
        ax.text(.01,.01,f"{label} rchi2={redchi2:.1f} |m/σ|={np.abs(slope_ratio):.1f}",ha='left',va='bottom',transform=ax.transAxes, fontsize=18, fontfamily='monospace')
    
    if linewave is not None:
        ax.axvline(linewave, color='grey', ls='-', lw=1)
        
        if False: # this was too busy
            axin = ax.inset_axes([0.75, 0.02, 0.24, 0.24])
            w1, w2 = linewave - inset_dwl/2, linewave + inset_dwl/2
            fill_between_steps(ax, spectrum.dispersion[ii], -sigma, sigma,
                               facecolor="#CCCCCC", edgecolor="none", alpha=1)
            axin.plot(plotx, ploty, 'k-', lw=3, drawstyle='steps-mid')
            axin.axvline(linewave, color='grey', ls='-', lw=1)
            axin.set_xlim(w1, w2)
            axin.set_ylim(-three_sigma,three_sigma)
            axin.set_xticks([])
            axin.set_yticks([])
            ax.indicate_inset_zoom(axin)
    
def plot_fe_trends(ax, session, plotname, ltab):
    t1 = ltab[(ltab["species"]==26.0) & np.isfinite(ltab["eqw"])]
    t2 = ltab[(ltab["species"]==26.1) & np.isfinite(ltab["eqw"])]
    y1, y2 = t1["logeps"] - 7.5, t2["logeps"] - 7.5
    if plotname == "expot":
        x1, x2 = t1["expot"], t2["expot"]
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.set_xlim(0,5); ax.set_xlabel("expot")
    elif plotname == "REW":
        x1 = np.log10(t1["eqw"]) - np.log10(t1["wavelength"]) - 3
        x2 = np.log10(t2["eqw"]) - np.log10(t2["wavelength"]) - 3
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        xmin = min(np.min(np.concatenate([x1,x2])), -5.5)
        xmax = max(np.max(np.concatenate([x1,x2])), -4.5)
        ax.set_xlim(xmin, xmax); ax.set_xlabel("REW")
    elif plotname in ["wave","wavelength"]:
        x1, x2 = t1["wavelength"], t2["wavelength"]
    else:
        raise ValueError(f"{plotname} has to be 'expot' or 'REW'")
    ax.axhline(biweight_location(y1, ignore_nan=True), color='k', lw=2, alpha=.7)
    ax.axhline(biweight_location(y2, ignore_nan=True), color='r', lw=2, alpha=.7)
    ax.plot(x1, y1, 'o', mfc='none', mec='k')
    ax.plot(x2, y2, 's', mfc='r', mec='k')
    ax.set_ylabel("[Fe/H]")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.20))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

def _find_order_wavelength(wave, specs):
    wcs = np.array([(spec.dispersion[0]+spec.dispersion[-1])/2 for spec in specs])
    ix = np.argmin(np.abs(wave - wcs))
    return specs[ix]
def _find_snr(wave, specs):
    spec = _find_order_wavelength(wave, specs)
    return round(np.nanmedian(spec.flux * spec.ivar**.5), 1)
def _find_ltab_summary(ltab):
    outdict = {}
    # Set some defaults of nans so everything has the right keys
    empty_dict = dict(Ntot=0, N=0, XH=np.nan, XFe=np.nan, sigma=np.nan)
    for species in [106.0, 12.0, 26.0, 26.1, 38.1, 56.1, 63.1]:
        outdict.setdefault(species, empty_dict.copy())
    
    speciess = list(np.unique(ltab["species"]))
    # Do 26.0 firstfor MH
    speciess.remove(26.0)
    speciess = [26.0] + speciess
    MH = np.nan
    
    for species in speciess:
        tt = ltab[(ltab["species"]==species) & (ltab["is_acceptable"]) & (~ltab["is_upper_limit"])]
        ft = tt[np.isfinite(tt["logeps"])]
        Ntot, N = len(tt), len(ft)
        if N == 0:
            XH = XFe = sigma = np.nan
        else:
            XH = biweight_location(ft["logeps"]) - solar_composition(species)
            sigma = biweight_scale(ft["logeps"])
            if species == 26.0: MH, XFe = XH, 0.0
            else: XFe = XH - MH
        outdata = dict(Ntot=Ntot, N=N, XH=XH, XFe=XFe, sigma=sigma)
        outdict[species] = outdata
    return outdict
def plot_summary_1(ax, session, ltab, name):
    ax.set_ylim(0,5)
    ax.set_xlim(0,100)
    ax.set_xticks([]); ax.set_yticks([])
    fontsize = 18
    
    Teff, logg, vt, MH = session.stellar_parameters    
    ax.text(5, 4, f"Teff/logg/vt/MH\n{Teff:.0f}/{logg:.2f}/{vt:.2f}/{MH:+5.2f}", ha='left', va='bottom', fontsize=fontsize, fontfamily='monospace')
    
    snr4000 = _find_snr(4000, session.input_spectra)
    snr4500 = _find_snr(4500, session.input_spectra)
    snr6500 = _find_snr(6500, session.input_spectra)
    ax.text(5, 3, f"4500/6500 = {snr4500:.1f}/{snr6500:.1f}", ha='left', va='bottom', fontsize=fontsize, fontfamily='monospace')
    
    ab = _find_ltab_summary(ltab)
    try: feh = ab[26.0]["XH"]
    except: feh = np.nan
    try: srfe = ab[38.1]["XFe"]
    except: srfe = np.nan
    try: bafe = ab[56.1]["XFe"]
    except: bafe = np.nan
    try: eufe = ab[63.1]["XFe"]
    except: eufe = np.nan
        
    ax.text(5, 2.5, f" [Fe/H]={feh:+.2f}\n[Sr/Fe]={srfe:+.2f}\n[Ba/Fe]={bafe:+.2f}\n[Eu/Fe]={eufe:+.2f}",
            ha='left', va='top', fontsize=fontsize, fontfamily='monospace')
def plot_summary_2(ax, session, ltab):
    ax.set_ylim(0,7)
    ax.set_xlim(0,100)
    ax.set_xticks([]); ax.set_yticks([])
    fontsize = 18
    
    ab = _find_ltab_summary(ltab)
    printstr = []
    for species, name, ytext in zip(
            [26.0, 26.1, 12.0, 38.1, 56.1, 63.1, 106.0],
            ["Fe I ","Fe II", "Mg I ","Sr II","Ba II","Eu II","CH   "],
            [6,5,4,3,2,1,0]
    ):
        sigma, N, XFe = ab[species]["sigma"], ab[species]["N"], ab[species]["XFe"]
        color = 'k' if sigma < scat_thresh[species] else 'r'
        ax.text(1, ytext+0.3, f"{name}: σ={sigma:.2f} [X/Fe]={XFe:+.2f} N={N}",
                ha='left', va='bottom', fontsize=fontsize, color=color, fontfamily='monospace')

def plot_one_grid(smh_fname, OUTDIR="./"):
    assert smh_fname.endswith(".smh"), f"{smh_fname} is not an SMH file"
    assert os.path.exists(smh_fname), f"{smh_fname} does not exist"
    
    if not os.path.exists(OUTDIR+"abund_data"):
        os.makedirs(OUTDIR+"abund_data")
    if not os.path.exists(OUTDIR+"abundstamp"):
        os.makedirs(OUTDIR+"abundstamp")

    name = os.path.basename(smh_fname)[:-4]
    
    out_fname = name+"_abundstamp.png"
    dpi = 150
    
    plotname_grid = [
        ["Summary1","Summary2","expot"],
        ["Mg5528","Sr4077","REW"],
        ["Mg5711","Sr4215","Eu6645"],
        ["CH4313","Ba4554","Eu4129"],
        ["CH4323","Ba5853","Eu4205"]
    ]
    plotname_grid = [
        ["Summary1","Summary2","expot","REW",None],
        ["Mg5528",      "Sr4077",      "Sr4215",      "Ba4554",      "Ba5853"],
        ["Mg5711","resid_Sr4077","resid_Sr4215","resid_Ba4554","resid_Ba5853"],
        [      "CH4313",      "CH4323",      "Eu4129",      "Eu4205",      "Eu6645"],
        ["resid_CH4313","resid_CH4323","resid_Eu4129","resid_Eu4205","resid_Eu6645"]
    ]
    
    ## This is hardcoded: from the grid position to which model to plot
    imodel_dict = dict(zip(
        ["CH4313","CH4323","Mg5528","Mg5711",
         "Sr4077","Sr4215","Ba4554","Ba5853",
         "Eu4129","Eu4205","Eu6645"],
        [40, 41, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    ))

    ## This is hardcoded: from the grid position to which model to plot
    imodel_line = dict(zip(
        ["CH4313","CH4323","Mg5528","Mg5711",
         "Sr4077","Sr4215","Ba4554","Ba5853",
         "Eu4129","Eu4205","Eu6645"],
        [None, None, 5528.405, 5711.088,
         4077.714, 4215.524, 4554.029, 5853.668,
         4129.720, 4205.040, 6645.060]
    ))
    imodel_insetdwl = dict(zip(
        ["CH4313","CH4323","Mg5528","Mg5711",
         "Sr4077","Sr4215","Ba4554","Ba5853",
         "Eu4129","Eu4205","Eu6645"],
        [None, None, None, None,
         1, 1, 1, 1,
         1, 1, 1]
    ))

    session = Session.load(smh_fname)
    ltab = get_line_table(session, True)
    ltab.write(OUTDIR+f"abund_data/{name}.txt", format="ascii", overwrite=True)
    print(f"Took {time.time()-start:.1f}s saved to abund_data/{name}.txt")
    
    Nrow, Ncol = 5, 5
    width, height = 6, 4
    fig, axes = plt.subplots(Nrow, Ncol, figsize=(width*Ncol, height*Nrow))
    
    for irow in range(Nrow):
        for icol in range(Ncol):
            ax = axes[irow, icol]
            plotname = plotname_grid[irow][icol]
            if plotname is None:
                ax.axis('off')
                continue
            if plotname in imodel_dict:
                plot_model_fit(ax, session, imodel_dict[plotname], plotname,
                               linewave=imodel_line[plotname], inset_dwl=imodel_insetdwl[plotname])
            elif plotname.startswith("resid_") and plotname[6:] in imodel_dict:
                pn = plotname[6:]
                plot_model_resid(ax, session, imodel_dict[pn], pn,
                                 linewave=imodel_line[pn], inset_dwl=imodel_insetdwl[pn])
            elif plotname in ["expot","REW"]:
                plot_fe_trends(ax, session, plotname, ltab)
            elif plotname == "Summary1":
                plot_summary_1(ax, session, ltab, name)
            elif plotname == "Summary2":
                plot_summary_2(ax, session, ltab)
            
    fig.tight_layout()
    fig.subplots_adjust(top=.95, wspace=0.12, hspace=.1)
    
    ab = _find_ltab_summary(ltab)
    feh, srfe, bafe, eufe = ab[26.0]["XH"], ab[38.1]["XFe"], ab[56.1]["XFe"], ab[63.1]["XFe"]
    label, color = get_title_color(srfe, bafe, eufe)
    fig.suptitle(name+" "+label, fontsize=30, weight='bold', fontfamily='monospace', color=color)
    
    fig.savefig(OUTDIR+"abundstamp/"+out_fname, dpi=dpi)
    plt.close(fig)
    print(f"Took {time.time()-start:.1f}s saved to {out_fname}")
    
