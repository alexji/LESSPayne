from __future__ import absolute_import, division, print_function # python2 compatibility
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import spectral_model
from . import utils
from . import fitting
from .read_spectrum import read_carpy_fits
from . import read_spectrum

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import gridspec

from cycler import cycler

# define plot properties
import matplotlib.cm as cm

from matplotlib import rcParams
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

def rgb(r,g,b):
    return (float(r)/256.,float(g)/256.,float(b)/256.)

cb2 = [rgb(31,120,180), rgb(255,127,0), rgb(51,160,44), rgb(227,26,28), \
       rgb(10,10,10), rgb(253,191,111), rgb(178,223,138), rgb(251,154,153)]

from copy import deepcopy

default_rcParams = deepcopy(rcParams)

def update_rcParams():
    rcParams['figure.figsize'] = (11,7.5)
    rcParams['font.family'] = 'Bitstream Vera Sans' 
    rcParams['font.size'] = 25
    rcParams['font.weight'] = 300
    
    rcParams['axes.prop_cycle'] = cycler('color', cb2)
    rcParams['axes.facecolor'] = 'white'
    rcParams['axes.grid'] = False
    rcParams['lines.linewidth'] = 1
    rcParams['patch.facecolor'] = cb2[0]
    rcParams['patch.edgecolor'] = 'white'

def reset_rcParams():
    rcParams = default_rcParams

def save_figures(name, wavelength, spectrum, spectrum_err, model_spec_best,
                 errors_payne=None, popt_best=None, model=None,
                 outdir=".", outfname_format="korder"):
    assert outfname_format in ["korder","wave"], outfname_format
    update_rcParams()
    rcParams['figure.dpi'] = 300
    
    model_errs = np.zeros(wavelength.shape)
    if (errors_payne is not None) and (popt_best is not None):
        #NN_coeffs, wavelength_payne = utils.read_in_neural_network()
        num_order, num_pixel = wavelength.shape
        coeff_poly = (len(popt_best) - 4 - 1 - 1) // num_order
        model_spec, model_errs = model.evaluate(popt_best, wavelength)
        model_errs = model_errs.reshape(wavelength.shape)

    plot_err = np.sqrt(spectrum_err**2 + model_errs**2)
    #plot_err = spectrum_err
    #print(np.nanmedian(plot_err))
    plot_err[plot_err > 999] = 999
    
    # make plot for individual order
    for k in range(wavelength.shape[0]):
        fig = plt.figure(figsize=[18,20]);
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    
    #----------------------------------------------------------------------
        # zooming in the wavelength by plotting in a few panels
        for i in range(5):
        
            # wavelength range
            wavelength_min = np.min(wavelength[k,:])-10.
            wavelength_max = np.max(wavelength[k,:])+10.
            wave_period = (wavelength_max-wavelength_min)/5.
        
            # the yaxis range
            spec_min = np.nanpercentile(spectrum[k,:],5)
            spec_max = np.nanpercentile(spectrum[k,:],95)
            
            ax = fig.add_subplot(5,1,i+1)
            plt.xlim([wavelength_min+wave_period*(i),wavelength_min+wave_period*(i+1)])
            plt.ylim([spec_min-0.2,spec_max+0.2])
            
            # observe spectrum
            plt.plot(wavelength[k,:], spectrum[k,:], lw=2, label="Data", color=cb2[0])
            
            # best prediction
            plt.plot(wavelength[k,:], model_spec_best[k,:], label="Payne", lw=2, color=cb2[1])
            
            # plotting errors
            plt.fill_between(wavelength[k,:], model_spec_best[k,:]-plot_err[k,:],\
                             model_spec_best[k,:]+plot_err[k,:], alpha=0.5, color=cb2[1])
        
            # shade the telluric region in gray
            telluric_region = np.where(spectrum_err[k,:] == 999.)[0]
            start_telluric = np.where(np.diff(telluric_region) != 1)[0] ## find the blocks
            start_telluric = np.concatenate([[0], start_telluric+1, [telluric_region.size-1]])
            for m in range(start_telluric.size-1):
                telluric_block = wavelength[k,telluric_region[start_telluric[m]:start_telluric[m+1]]]
                num_telluric = telluric_block.size
                plt.fill_between(telluric_block, np.ones(num_telluric)*-10., np.ones(num_telluric)*10.,\
                                 alpha=0.5, color="gray")
            
    #----------------------------------------------------------------------
        # add axis and legend
        plt.xlabel("Wavelength [A]")
        plt.legend(loc="lower right", fontsize=28, frameon=False,\
                    borderpad=0.05, labelspacing=0.1)
    
        # save figure
        plt.tight_layout()
        if outfname_format=="korder":
            plt.savefig("{}/{}_Order_{:02}.pdf".format(outdir, name, k+1))
        elif outfname_format=="wave":
            roundwave = 10 * int(np.median(wavelength[k,:]) // 10)
            plt.savefig("{}/{}_Order_{:02}_{}A.pdf".format(outdir, name, k+1, roundwave))
        plt.close()
    reset_rcParams()

def save_figures_multipdf(name, wavelength, spectrum, spectrum_err, model_spec_best,
                          errors_payne=None, popt_best=None, model=None,
                          outdir="."):
    update_rcParams()
    outfname = f"{outdir}/p4m_{name}.pdf"
    
    model_errs = np.zeros(wavelength.shape)
    if (errors_payne is not None) and (popt_best is not None):
        #NN_coeffs, wavelength_payne = utils.read_in_neural_network()
        num_order, num_pixel = wavelength.shape
        coeff_poly = (len(popt_best) - 4 - 1 - 1) // num_order
        model_spec, model_errs = model.evaluate(popt_best, wavelength)
        model_errs = model_errs.reshape(wavelength.shape)

    plot_err = np.sqrt(spectrum_err**2 + model_errs**2)
    #plot_err = spectrum_err
    #print(np.nanmedian(plot_err))
    plot_err[plot_err > 999] = 999
    
    with PdfPages(outfname) as pdf:

        # make plot for individual order
        for k in range(wavelength.shape[0]):
            fig = plt.figure(figsize=[18,20]);
            ax = fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        
        #----------------------------------------------------------------------
            # zooming in the wavelength by plotting in a few panels
            for i in range(5):
            
                # wavelength range
                wavelength_min = np.min(wavelength[k,:])-10.
                wavelength_max = np.max(wavelength[k,:])+10.
                wave_period = (wavelength_max-wavelength_min)/5.
            
                # the yaxis range
                spec_min = np.nanpercentile(spectrum[k,:],5)
                spec_max = np.nanpercentile(spectrum[k,:],95)
                
                ax = fig.add_subplot(5,1,i+1)
                plt.xlim([wavelength_min+wave_period*(i),wavelength_min+wave_period*(i+1)])
                plt.ylim([spec_min-0.2,spec_max+0.2])
                
                # observe spectrum
                plt.plot(wavelength[k,:], spectrum[k,:], lw=2, label="Data", color=cb2[0])
                
                # best prediction
                plt.plot(wavelength[k,:], model_spec_best[k,:], label="Payne", lw=2, color=cb2[1])
                
                # plotting errors
                plt.fill_between(wavelength[k,:], model_spec_best[k,:]-plot_err[k,:],\
                                 model_spec_best[k,:]+plot_err[k,:], alpha=0.5, color=cb2[1])
            
                # shade the telluric region in gray
                telluric_region = np.where(spectrum_err[k,:] == 999.)[0]
                start_telluric = np.where(np.diff(telluric_region) != 1)[0] ## find the blocks
                start_telluric = np.concatenate([[0], start_telluric+1, [telluric_region.size-1]])
                for m in range(start_telluric.size-1):
                    telluric_block = wavelength[k,telluric_region[start_telluric[m]:start_telluric[m+1]]]
                    num_telluric = telluric_block.size
                    plt.fill_between(telluric_block, np.ones(num_telluric)*-10., np.ones(num_telluric)*10.,\
                                     alpha=0.5, color="gray")
                
        #----------------------------------------------------------------------
            # add axis and legend
            plt.xlabel("Wavelength [A]")
            plt.legend(loc="lower right", fontsize=28, frameon=False,\
                        borderpad=0.05, labelspacing=0.1)
        
            # save figure
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    reset_rcParams()
