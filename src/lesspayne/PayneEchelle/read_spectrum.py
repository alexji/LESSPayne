import numpy as np
import re
from collections import OrderedDict
from astropy.io import fits


#===========================================================================================================
def read_carpy_fits(path):

    '''
    read in a carpy reduced spectrum.
    '''

    specs = Spectrum1D.read(path)

    wavelength = []
    spectrum = []
    spectrum_err = []

    for spec in specs:
        wavelength.append(spec.wavelength)
        spectrum.append(spec.spectrum)
        spectrum_err.append(spec.spectrum_err)

    wavelength = np.array(wavelength)
    spectrum = np.array(spectrum)
    spectrum_err = np.array(spectrum_err)

    return wavelength, spectrum, spectrum_err


#===========================================================================================================
# a class to read in carpy reduced spectra
# adapted from Alex Ji's alexmod

# initiate class
class Spectrum1D(object):

    def __init__(self, dispersion, flux, ivar, metadata=None):
        self.wavelength = np.array(dispersion)
        self.spectrum = np.array(flux)
        self.spectrum_err = np.sqrt(1./(np.array(ivar)+1e-6))
        return

#------------------------------------------------------------------------------------------------------------
    # read in spectrum
    @classmethod
    def read(cls, path, fluxband=2, **kwargs):
        """
        Create a Spectrum1D class from a path on disk.
        """
        dispersion, flux, ivar = cls.read_fits_multispec(path, fluxband=fluxband, **kwargs)
        orders = [cls(dispersion=d, flux=f, ivar=i) for d, f, i in zip(dispersion, flux, ivar)]
        return orders

    @classmethod
    def read_fits_multispec(cls, path, fluxband, **kwargs):
        """
        Create multiple Spectrum1D classes from a multi-spec file on disk.
        """

        assert fluxband in [1,2,3,4,5,6,7], fluxband
        
        WAT_LENGTH=68
        image = fits.open(path)

        # Merge headers into a metadata dictionary.
        metadata = OrderedDict()
        for key, value in image[0].header.items():
            if key in metadata:
                metadata[key] += value
            else:
                metadata[key] = value

        # Read data
        flux = image[0].data

        # Join the WAT keywords for dispersion mapping.
        i, concatenated_wat, key_fmt = (1, str(""), "WAT2_{0:03d}")
        while key_fmt.format(i) in metadata:
            value = metadata[key_fmt.format(i)]
            concatenated_wat += value + (" "  * (WAT_LENGTH - len(value)))
            i += 1

        # Split the concatenated header into individual orders.
        order_mapping = np.array([list(map(float, each.rstrip('" ').split())) \
                for each in re.split('spec[0-9]+ ?= ?"', concatenated_wat)[1:]])

        # Parse the order mapping into dispersion values.
        num_pixels, num_orders = metadata["NAXIS1"], metadata["NAXIS2"]
        dispersion = np.zeros((num_orders, num_pixels), dtype=np.float) + np.nan
        for j in range(num_orders):
            _dispersion = compute_dispersion(*order_mapping[j])
            dispersion[j,0:len(_dispersion)] = _dispersion

        # inverse variance array
        if fluxband in [1,2,3,4,5,6]:
            flux_ext = fluxband-1
            noise_ext = 2
            flux = image[0].data[flux_ext]
            ivar = image[0].data[noise_ext]**(-2)
        else:
            flux_ext = 6
            Norder = image[0].data.shape[1]
            flats = [image[0].data[2,iorder]/image[0].data[6,iorder] for iorder in range(Norder)]
            ivar = [(flats[iorder]/image[0].data[2,iorder])**2. for iorder in range(Norder)]
            ivar = np.array(ivar)
        
        dispersion = np.atleast_2d(dispersion)
        flux = np.atleast_2d(flux)
        ivar = np.atleast_2d(ivar)

        # Ensure dispersion maps from blue to red direction.
        if np.min(dispersion[0]) > np.min(dispersion[-1]):
            dispersion = dispersion[::-1]
            if len(flux.shape) > 2:
                flux = flux[:, ::-1]
                ivar = ivar[:, ::-1]
            else:
                flux = flux[::-1]
                ivar = ivar[::-1]

        # Do something sensible regarding zero or negative fluxes.
        ivar[0 >= flux] = 0.000000000001

        return (dispersion, flux, ivar)

#------------------------------------------------------------------------------------------------------------
# calculate the wavelength grid for individual orders
def compute_dispersion(aperture, beam, dispersion_type, dispersion_start,
    mean_dispersion_delta, num_pixels, redshift, aperture_low, aperture_high):
    """
    Compute a dispersion mapping from a IRAF multi-spec description.
    """

    dispersion = dispersion_start + np.arange(num_pixels) * mean_dispersion_delta
    return dispersion
