#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" An object for dealing with one-dimensional spectra. """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from six import iteritems
from .read_expres import read_expres as read_expres_func

__all__ = ["Spectrum1D", "stitch", "coadd", "read_mike_spectrum", "write_fits_linear"]

import logging
import numpy as np
import os
import re
from collections import OrderedDict
from hashlib import md5

from astropy.io import fits
from astropy.stats import biweight_scale
from scipy import interpolate, ndimage, polyfit, poly1d, optimize as op, signal
from .robust_polyfit import polyfit as rpolyfit

logger = logging.getLogger(__name__)

class Spectrum1D(object):
    """ A one-dimensional spectrum. """

    def __init__(self, dispersion, flux, ivar, metadata=None):
        """
        Initialie a `Spectrum1D` object with the given dispersion, flux and
        inverse variance arrays.

        :param dispersion:
            An array containing the dispersion values for every pixel.

        :param flux:
            An array containing flux values for all pixels.

        :param ivar:
            An array containing the inverse variances for the flux values in all
            pixels.

        :param metadata: [optional]
            A dictionary containing metadata for this spectrum.
        """

        dispersion = np.array(dispersion)
        flux = np.array(flux)
        ivar = np.array(ivar)

        if max([len(_.shape) for _ in (dispersion, flux, ivar)]) > 1:
            raise ValueError(
                "dispersion, flux and ivar must be one dimensional arrays")
        
        if flux.size != dispersion.size:
            raise ValueError(
                "dispersion and flux arrays must have the same "
                "size ({0} != {1})".format(dispersion.size, flux.size, ))

        if ivar.size != dispersion.size:
            raise ValueError(
                "dispersion and ivar arrays must have the same "
                "size ({0} != {1})".format(dispersion.size, ivar.size, ))

        self.metadata = metadata or {}

        # Don't allow orders to be back-to-front.
        if np.all(np.diff(dispersion) < 0):
            dispersion = dispersion[::-1]
            flux = flux[::-1]
            ivar = ivar[::-1]

        # HACK so that *something* can be done with spectra when there is no
        # inverse variance array.
        if not np.any(np.isfinite(ivar)) or np.nanmax(ivar) == 0:
            ivar = np.ones_like(flux) * 1.0/np.nanmean(flux)

        self._dispersion = dispersion
        self._flux = flux
        self._ivar = ivar

        return None
    

    @property
    def dispersion(self):
        """ Return the dispersion points for all pixels. """
        return self._dispersion


    @property
    def flux(self):
        """ Return the flux for all pixels. """
        return self._flux


    @property
    def ivar(self):
        """ Return the inverse variance of the flux for all pixels. """
        return self._ivar


    @classmethod
    def read(cls, path, debug=False, **kwargs):
        """
        Create a Spectrum1D class from a path on disk.

        :param path:
            The path of the filename to load.
        """

        if not os.path.exists(path):
            raise IOError("filename '{}' does not exist".format(path))

        # Try multi-spec first since this is currently the most common use case.
        methods = (
            (cls.read_fits_multispec,"read_fits_multispec"),
            (cls.read_fits_spectrum1d,"read_fits_spectrum1d"),
            (cls.read_ascii_spectrum1d,"read_ascii_spectrum1d"),
            (cls.read_alex_spectrum,"read_alex_spectrum"),
            (cls.read_ceres,"read_ceres"),
            (cls.read_multispec,"read_multispec"),
            (cls.read_ascii_spectrum1d_noivar,"read_ascii_spectrum1d_noivar"),
            (cls.read_expres,"read_expres"),
        )

        for method, name in methods:
            try:
                dispersion, flux, ivar, metadata = method(path, **kwargs)
                
            except Exception as e:
                if debug:
                    print("=====")
                    print(name)
                    print(e)
                continue

            else:
                orders = [cls(dispersion=d, flux=f, ivar=i, metadata=metadata) \
                    for d, f, i in zip(dispersion, flux, ivar)]
                break
        else:
            raise ValueError("cannot read spectrum from path {}".format(path))

        # If it's a single order, just return that instead of a 1-length list.
        orders = orders if len(orders) > 1 else orders[0]
        return orders


    @classmethod
    def read_alex_spectrum(cls, path):
        image = fits.open(path)

        # Merge headers into a metadata dictionary.
        metadata = OrderedDict()
        for key, value in image[0].header.items():
            if key in metadata:
                metadata[key] += value
            else:
                metadata[key] = value
        metadata["smh_read_path"] = path
        
        md5_hash = md5(";".join([v for k, v in metadata.items() \
                                 if k.startswith("BANDID")]).encode("utf-8")).hexdigest()
        assert md5_hash == "8538046d98bf8a760b04690e53e394a1"

        data = image[0].data
        waves, fluxs, ivars = data[0], data[1], data[2]

        return (waves, fluxs, ivars, metadata)

    @classmethod
    def write_alex_spectrum_from_specs(cls, path, specs, overwrite=False):
        
        Nspec = len(specs)
        Npix = np.array([len(spec.dispersion) for spec in specs])
        if not np.all(Npix[0] == Npix):
            print("Not all spectra have the same number of pixels!")
            print(Npix)
            print("Padding with last wavelength of each order, zero flux, zero ivar")
        Npix = np.max(Npix)
        wave, flux, ivar = np.zeros((Nspec,Npix)), np.zeros((Nspec,Npix)), np.zeros((Nspec,Npix))
        for i, spec in enumerate(specs):
            N = len(spec.dispersion)
            wave[i,0:N] = spec.dispersion
            flux[i,0:N] = spec.flux
            ivar[i,0:N] = spec.ivar
            if N < Npix: wave[i,N:Npix] = spec.dispersion[-1]
        #wave = np.array([spec.dispersion for spec in specs])
        #flux = np.array([spec.flux for spec in specs])
        #ivar = np.array([spec.ivar for spec in specs])
        
        ### Write a 3 extension fits file: wave, flux, ivar
        # make the output array: 3 x Norder x Npix
        outdata = np.array([wave, flux, ivar], dtype=float)

        hdu = fits.PrimaryHDU(outdata)
        
        # header: add new BANDID
        hdu.header["BANDID1"] = "wavelength array"
        hdu.header["BANDID2"] = "flux array"
        hdu.header["BANDID3"] = "ivar array"
        
        hdu.writeto(path, output_verify="fix", overwrite=overwrite)

    @classmethod
    def write_alex_spectrum_from_arrays(cls, path, wave, flux, ivar, overwrite=False):
        
        ### Write a 3 extension fits file: wave, flux, ivar
        # make the output array: 3 x Norder x Npix
        outdata = np.array([wave, flux, ivar], dtype=float)

        hdu = fits.PrimaryHDU(outdata)
        
        # header: add new BANDID
        hdu.header["BANDID1"] = "wavelength array"
        hdu.header["BANDID2"] = "flux array"
        hdu.header["BANDID3"] = "ivar array"
        
        hdu.writeto(path, output_verify="fix", overwrite=overwrite)

    @classmethod
    def read_ceres(cls, fname):
        with fits.open(fname) as hdul:
            assert len(hdul)==1, len(hdul)
            header = hdul[0].header
            assert header["PIPELINE"] == "CERES", header["PIPELINE"]
            data = hdul[0].data
            Nband, Norder, Npix = data.shape
            # https://github.com/rabrahm/ceres
            # by default it looks sorted from red to blue orders, so we'll flip it in the output
            waves = data[0,::-1,:]
            fluxs = data[1,::-1,:]
            ivars = data[2,::-1,:]
            # 3, 4 = blaze corrected flux and error
            # 5, 6 = continuum normalized flux and error
            # 7 = continuum
            # 8 = s/n
            # 9, 10 = Continumm normalized flux multiplied by the derivative of the wavelength with respect to the pixels + err

            # Merge headers into a metadata dictionary.
            metadata = OrderedDict()
            for key, value in header.items():
                if key in metadata:
                    metadata[key] += value
                else:
                    metadata[key] = value
            metadata["smh_read_path"] = fname
        return (waves, fluxs, ivars, metadata)

    @classmethod
    def read_multispec(cls, fname, full_output=False):
        """
        There are some files that are not reduced with Dan Kelson's pipeline.
        So we have to read those manually and define ivar
        """
        #print("READ MULTISPEC OLD")
        # Hardcoded file with old CarPy format: 5 bands instead of 7
        if "hd13979red_multi_200311ibt" in fname:
            WAT_LENGTH=67
        else:
            WAT_LENGTH=68
        
        try:
            orders = cls.read(fname, WAT_LENGTH=WAT_LENGTH)
            code = 1
        except:
            #print("OLD FORMAT STARTING")
            # This is the old format: (Norders x Npix) with no noise spec...
            with fits.open(fname) as hdulist:
                assert len(hdulist)==1, len(hdulist)
                header = hdulist[0].header
                data = hdulist[0].data
                # orders x pixels
                assert len(data.shape)==2, data.shape
                
                metadata = OrderedDict()
                for k, v in header.items():
                    if k in metadata:
                        metadata[k] += v
                    else:
                        metadata[k] = v

            ## Compute dispersion
            assert metadata["CTYPE1"].upper().startswith("MULTISPE") \
                or metadata["WAT0_001"].lower() == "system=multispec"

            # Join the WAT keywords for dispersion mapping.
            i, concatenated_wat, key_fmt = (1, str(""), "WAT2_{0:03d}")
            while key_fmt.format(i) in metadata:
                value = metadata[key_fmt.format(i)]
                concatenated_wat += value + (" "  * (68 - len(value)))
                i += 1

            # Split the concatenated header into individual orders.
            try:
                order_mapping = np.array([list(map(float, each.rstrip('" ').split())) \
                    for each in re.split('spec[0-9]+ ?= ?"', concatenated_wat)[1:]])
            except Exception as e:
                logger.debug("Error in parsing concatenated_wat")
                logger.debug(e)
                logger.debug(concatenated_wat)
                raise
            
            if len(order_mapping)==0:
                NAXIS1 = metadata["NAXIS1"]
                NAXIS2 = metadata["NAXIS2"]
                dispersion = np.array([np.arange(NAXIS1) for _ in range(NAXIS2)])
            else:
                dispersion = np.array(
                    [compute_dispersion(*mapping) for 
                     mapping in order_mapping])
            
            ## Compute flux
            flux = data
            #flux[0 > flux] = np.nan
            
            ## Compute ivar assuming Poisson noise
            ivar = 1./flux
            
        return (dispersion, flux, ivar, metadata)

    @classmethod
    def read_fits_multispec(cls, path, flux_ext=None, ivar_ext=None,
                            WAT_LENGTH=68, override_bad=False, **kwargs):
        """
        Create multiple Spectrum1D classes from a multi-spec file on disk.

        :param path:
            The path of the multi-spec filename to load.

        :param flux_ext: [optional]
            The zero-indexed extension number containing flux values.

        :param ivar_ext: [optional]
            The zero-indexed extension number containing the inverse variance of
            the flux values.

        :param WAT_LENGTH: [optional]
            The multispec format fits uses 68, but some files are broken
        """

        image = fits.open(path)

        # Merge headers into a metadata dictionary.
        metadata = OrderedDict()
        for key, value in image[0].header.items():

            # NOTE: In the old SMH we did a try-except block to string-ify and
            #       JSON-dump the header values, and if they could not be
            #       forced to a string we didn't keep that header.

            #       I can't remember what types caused that problem, but it was
            #       to prevent SMH being unable to save a session.
            
            #       Since we are pickling now, that shouldn't be a problem
            #       anymore, but this note is here to speed up debugging in case
            #       that issue returns.

            if key in metadata:
                try:
                    metadata[key] += value
                except:
                    metadata[key] += str(value)
            else:
                metadata[key] = value

        metadata["smh_read_path"] = path

        flux = image[0].data

        assert metadata["CTYPE1"].upper().startswith("MULTISPE") \
            or metadata["WAT0_001"].lower() == "system=multispec"

        # Join the WAT keywords for dispersion mapping.
        i, concatenated_wat, key_fmt = (1, str(""), "WAT2_{0:03d}")
        while key_fmt.format(i) in metadata:
            # .ljust(68, " ") had str/unicode issues across Python 2/3
            value = metadata[key_fmt.format(i)]
            concatenated_wat += value + (" "  * (WAT_LENGTH - len(value)))
            i += 1

        # Split the concatenated header into individual orders.
        order_mapping = np.array([list(map(float, each.rstrip('" ').split())) \
                for each in re.split('spec[0-9]+ ?= ?"', concatenated_wat)[1:]])

        # Parse the order mapping into dispersion values.
        # Do it this way to ensure ragged arrays work
        num_pixels, num_orders = metadata["NAXIS1"], metadata["NAXIS2"]
        dispersion = np.zeros((num_orders, num_pixels), dtype=float) + np.nan
        for j in range(num_orders):
            _dispersion = compute_dispersion(*order_mapping[j])
            dispersion[j,0:len(_dispersion)] = _dispersion
        #dispersion = np.array(
        #    [compute_dispersion(*mapping) for mapping in order_mapping])

        # Get the flux and inverse variance arrays.
        # NOTE: Most multi-spec data previously used with SMH have been from
        #       Magellan/MIKE, and reduced with CarPy.
        md5_hash = md5(";".join([v for k, v in metadata.items() \
            if k.startswith("BANDID")]).encode('utf-8')).hexdigest()
        is_carpy_mike_product = (md5_hash == "0da149208a3c8ba608226544605ed600")
        is_carpy_mike_product_old = (md5_hash == "e802331006006930ee0e60c7fbc66cec")
        is_carpy_mage_product = (md5_hash == "6b2c2ec1c4e1b122ccab15eb9bd305bc")
        is_iraf_3band_product = (md5_hash == "a4d8f6f51a7260fce1642f7b42012969")
        is_iraf_4band_product = (md5_hash == "30fdecbbc94c9393c73537eefb02efda")
        is_apo_product = (image[0].header.get("OBSERVAT", None) == "APO")
        is_dupont_product = (md5_hash == "2ab648afed96dcff5ccd10e5b45730c1")
        is_mcdonald_product = (image[0].header.get("OBSERVAT", "").strip() == "MCDONALD")
        
        if is_carpy_mike_product or is_carpy_mage_product or is_carpy_mike_product_old or is_dupont_product:
            # CarPy gives a 'noise' spectrum, which we must convert to an
            # inverse variance array
            flux_ext = flux_ext or 1
            noise_ext = ivar_ext or 2

            logger.info(
                "Recognized CarPy product. Using zero-indexed flux/noise "
                "extensions (bands) {}/{}".format(flux_ext, noise_ext))

            flux = image[0].data[flux_ext]
            if flux_ext == 6:
                flat = image[0].data[1]/image[0].data[6]
                ivar = (flat/image[0].data[2])**2.
            else:
                ivar = image[0].data[noise_ext]**(-2)
            
        elif is_iraf_3band_product:
            flux_ext = flux_ext or 0
            noise_ext = ivar_ext or 2
            
            logger.info(
                "Recognized IRAF 3band product. Using zero-indexed flux/noise "
                "extensions (bands) {}/{}".format(flux_ext, noise_ext))
            logger.info(
                image[0].header["BANDID{}".format(flux_ext+1)]
            )
            logger.info(
                image[0].header["BANDID{}".format(noise_ext+1)]
            )
            
            flux = image[0].data[flux_ext]
            ivar = image[0].data[noise_ext]**(-2)

        elif is_iraf_4band_product:
            flux_ext = flux_ext or 0
            noise_ext = ivar_ext or 3
            
            logger.info(
                "Recognized IRAF 4band product. Using zero-indexed flux/noise "
                "extensions (bands) {}/{}".format(flux_ext, noise_ext))
            logger.info(
                image[0].header["BANDID{}".format(flux_ext+1)]
            )
            logger.info(
                image[0].header["BANDID{}".format(noise_ext+1)]
            )
            
            flux = image[0].data[flux_ext]
            ivar = image[0].data[noise_ext]**(-2)

        elif is_apo_product or is_mcdonald_product:
            flux_ext = flux_ext or 0

            if md5_hash == "9d008ba2c3dc15549fd8ffe8a605ec15":
                default_ivar_ext = 3
            elif md5_hash == "a4d8f6f51a7260fce1642f7b42012969":
                default_ivar_ext = 2
            else:
                default_ivar_ext = -1
            noise_ext = ivar_ext or default_ivar_ext

            if md5_hash == "9d008ba2c3dc15549fd8ffe8a605ec15":
                logger.info(
                    "Recognized APO product. Using zero-indexed flux/noise "
                    "extensions (bands) {}/{}".format(flux_ext, noise_ext))
            elif md5_hash == "a4d8f6f51a7260fce1642f7b42012969":
                logger.info(
                    "Recognized McDonald product. Using zero-indexed flux/noise "
                    "extensions (bands) {}/{}".format(flux_ext, noise_ext))
            
            logger.info(
                image[0].header["BANDID{}".format(flux_ext+1)]
            )
            logger.info(
                image[0].header["BANDID{}".format(noise_ext+1)]
            )

            flux = image[0].data[flux_ext]
            ivar = image[0].data[noise_ext]**(-2)

        else:
            ivar = np.full_like(flux, np.nan)
            logger.info("could not identify flux and ivar extensions "
                        "(using nan for ivar)")
            # It turns out this can mess you up badly so it's better to
            # just throw the error.
            if not override_bad:
                raise NotImplementedError

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
        #flux[0 >= flux] = np.nan

        # turn into list of arrays if it's ragged
        if np.any(np.isnan(dispersion)):
            newdispersion = []
            newflux = []
            newivar = []
            for j in range(num_orders):
                d = dispersion[j,:]
                ii = np.isfinite(d)
                newdispersion.append(dispersion[j,ii])
                newflux.append(flux[j,ii])
                newivar.append(ivar[j,ii])
            dispersion = newdispersion
            flux = newflux
            ivar = newivar

        return (dispersion, flux, ivar, metadata)

    @classmethod
    def read_expres(cls, path, **kwargs):
        """
        Read Spectrum1D data from an EXPRES FITS file.

        :param path:
            The path of the FITS filename to read.

        Other kwargs
        full_output=False, as_arrays=False, as_order_dict=False, as_raw_table=False
        (forces as_arrays=True)
        """
        
        meta = {"ALT_OBS":2360, "LAT_OBS":34.744444444, "LONG_OBS":-111.421944444}
        kwargs["as_arrays"] = True
        alloutput = read_expres_func(path, **kwargs)
        dispersion = alloutput[0]
        flux = alloutput[1]
        ivar = alloutput[2]**-2
        return (dispersion, flux, ivar, meta)

    @classmethod
    def read_fits_spectrum1d(cls, path, **kwargs):
        """
        Read Spectrum1D data from a binary FITS file.

        :param path:
            The path of the FITS filename to read.
        """

        image = fits.open(path)

        # Merge headers into a metadata dictionary.
        metadata = OrderedDict()
        for key, value in image[0].header.items():
            if key in metadata:
                metadata[key] += value
            else:
                metadata[key] = value
        metadata["smh_read_path"] = path

        # Find the first HDU with data in it.
        for hdu_index, hdu in enumerate(image):
            if hdu.data is not None: break

        ctype1 = image[0].header.get("CTYPE1", None)

        if len(image) == 2 and hdu_index == 1:

            dispersion_keys = ("dispersion", "disp", "WAVELENGTH[COORD]")
            for key in dispersion_keys:
                try:
                    dispersion = image[hdu_index].data[key]

                except KeyError:
                    continue

                else:
                    break

            else:
                raise KeyError("could not find any dispersion key: {}".format(
                    ", ".join(dispersion_keys)))

            flux_keys = ("flux", "SPECTRUM[FLUX]")
            for key in flux_keys:
                try:
                    flux = image[hdu_index].data[key]
                except KeyError:
                    continue
                else:
                    break
            else:
                raise KeyError("could not find any flux key: {}".format(
                    ", ".join(flux_keys)))

            # Try ivar, then error, then variance.
            try:
                ivar = image[hdu_index].data["ivar"]
            except KeyError:
                try:
                    errs = image[hdu_index].data["SPECTRUM[SIGMA]"]
                    ivar = 1.0/errs**2.
                except KeyError:
                    variance = image[hdu_index].data["variance"]
                    ivar = 1.0/variance

        else:
            # Build a simple linear dispersion map from the headers.
            # See http://iraf.net/irafdocs/specwcs.php
            crval = image[0].header["CRVAL1"]
            naxis = image[0].header["NAXIS1"]
            crpix = image[0].header.get("CRPIX1", 1)
            cdelt = image[0].header["CDELT1"]
            ltv = image[0].header.get("LTV1", 0)

            # + 1 presumably because fits is 1-indexed instead of 0-indexed
            dispersion = \
                crval + (np.arange(naxis) + 1 - crpix) * cdelt - ltv * cdelt

            flux = image[0].data
            if len(image) == 1:
                ivar = np.ones_like(flux)*1e+5 # HACK S/N ~300 just for training/verification purposes
            else:
                ivar = image[1].data

        dispersion = np.atleast_2d(dispersion)
        flux = np.atleast_2d(flux)
        ivar = np.atleast_2d(ivar)

        return (dispersion, flux, ivar, metadata)


    @classmethod
    def read_ascii_spectrum1d(cls, path, **kwargs):
        """
        Read Spectrum1D data from an ASCII-formatted file on disk.

        :param path:
            The path of the ASCII filename to read.
        """

        kwds = kwargs.copy()
        kwds.update({
            "unpack": True
        })
        kwds.setdefault("usecols", (0, 1, 2))

        if len(kwds["usecols"])==3:
            try:
                dispersion, flux, ivar = np.loadtxt(path, **kwds)
            except:
                # Try by ignoring the first row.
                kwds.setdefault("skiprows", 1)
                dispersion, flux, ivar = np.loadtxt(path, **kwds)
        elif len(kwds["usecols"])==2:
            try:
                dispersion, flux = np.loadtxt(path, **kwds)
            except:
                # Try by ignoring the first row.
                kwds.setdefault("skiprows", 1)
                dispersion, flux = np.loadtxt(path, **kwds)
            
            ivar = np.ones_like(flux)*1e+5 # HACK S/N ~300 just for training/verification purposes

        dispersion = np.atleast_2d(dispersion)
        flux = np.atleast_2d(flux)
        ivar = np.atleast_2d(ivar)
        metadata = { "smh_read_path": path }
        
        return (dispersion, flux, ivar, metadata)


    @classmethod
    def read_ascii_spectrum1d_noivar(cls, path, **kwargs):
        """
        Read Spectrum1D data from an ASCII-formatted file on disk.

        :param path:
            The path of the ASCII filename to read.
        """

        kwds = kwargs.copy()
        kwds["usecols"] = (0,1)

        return cls.read_ascii_spectrum1d(path, **kwds)


    def write(self, filename, overwrite=True, output_verify="warn", header=None):
        """ Write spectrum to disk. """

        if os.path.exists(filename) and not overwrite:
            raise IOError("Filename '%s' already exists and we have been asked not to clobber it." % (filename, ))
        
        if not filename.endswith('fits'):
            if header is not None:
                print("Writing text, cannot include header!")
            a = np.array([self.dispersion, self.flux, self.ivar]).T
            if self.ivar.min() < 1e-4 and self.ivar.min() > 0.000000000001:
                fmt = "%.12f"
            else:
                fmt = "%.4f"
            np.savetxt(filename, a, fmt=fmt.encode('ascii'))
            return
        
        else:

            crpix1, crval1 = 1, self.dispersion.min()
            cdelt1 = np.mean(np.diff(self.dispersion))
            naxis1 = len(self.dispersion)
            
            linear_dispersion = crval1 + (np.arange(naxis1) + 1 - crpix1) * cdelt1

            ## Check for linear dispersion map
            maxdiff = np.max(np.abs(linear_dispersion - self.dispersion))
            if maxdiff > 1e-3:
                ## TODO Come up with something better...
                ## Frustratingly, it seems like there's no easy way to make an IRAF splot-compatible
                ## way of storing the dispersion data for an arbitrary dispersion sampling.
                ## The standard way of doing it requires putting every dispersion point in the
                ## WAT2_xxx header.
                
                ## So just implemented it as a binary table with names according to 
                ## http://iraf.noao.edu/projects/spectroscopy/formats/sptable.html
                ## It doesn't work because I have not put in WCS headers, but I do not know
                ## how to tell it to look for those.
                
                ## I think some of these links below may provide a better solution though
                ## http://iraf.noao.edu/projects/spectroscopy/formats/onedspec.html
                ## http://www.cv.nrao.edu/fits/
                ## http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?specwcs
                
                ## python 2(?) hack needs the b prefix
                ## https://github.com/numpy/numpy/issues/2407
                
                if header is None:
                    headers = {}
                elif header is True:
                    headers = self.metadata
                else:
                    headers = header

                dispcol = fits.Column(name="WAVELENGTH[COORD]",
                                      format="D",
                                      array=self.dispersion)
                fluxcol = fits.Column(name="SPECTRUM[FLUX]",
                                      format="D",
                                      array=self.flux)
                errscol = fits.Column(name="SPECTRUM[SIGMA]",
                                      format="D",
                                      array=(self.ivar)**-0.5)
                
                coldefs = fits.ColDefs([dispcol, fluxcol, errscol])
                hdu = fits.BinTableHDU.from_columns(coldefs)
                hdu.header.update(header)
                hdu.writeto(filename, output_verify=output_verify, overwrite=overwrite)

                return

            else:
                # We have a linear dispersion!
                hdu = fits.PrimaryHDU(np.array(self.flux))
                hdu2 = fits.ImageHDU(np.array(self.ivar))
    
                #headers = self.headers.copy()
                #headers = {}
                if header is None:
                    headers = {}
                elif header is True:
                    headers = self.metadata
                else:
                    headers = header
                    
                headers.update({
                    'CTYPE1': 'LINEAR  ',
                    'CRVAL1': crval1,
                    'CRPIX1': crpix1,
                    'CDELT1': cdelt1
                })
                
                for key, value in iteritems(headers):
                    try:
                        hdu.header[key] = value
                    except ValueError:
                        #logger.warn("Could not save header key/value combination: %s = %s" % (key, value, ))
                        print("Could not save header key/value combination: %s = %s".format(key, value))
                hdulist = fits.HDUList([hdu,hdu2])
                hdulist.writeto(filename, output_verify=output_verify, overwrite=overwrite)
                return

    def redshift(self, v=None, z=None, reinterpolate=False):
        """
        Redshift the spectrum.

        :param v:
            The velocity in km/s.

        :param z:
            A redshift.
            
        :param reinterpolate:
            If True, interpolates result onto original dispersion
        """

        if (v is None and z is None) or (v is not None and z is not None):
            raise ValueError("either v or z must be given, but not both")

        if reinterpolate:
            olddisp = self._dispersion.copy()

        z = z or v/299792458e-3
        self._dispersion *= 1 + z
        
        if reinterpolate:
            newflux = np.interp(olddisp, self._dispersion, self._flux)
            self._dispersion = olddisp
            self._flux = newflux
        return True


    # State functionality for serialization.
    def __getstate__(self):
        """ Return the spectrum state. """
        return (self.dispersion, self.flux, self.ivar, self.metadata)


    def __setstate__(self, state):
        """
        Set the state of the spectrum.

        :param state:
            A four-length tuple containing the dispersion array, flux array, the
            inverse variance of the fluxes, and a metadata dictionary.
        """
        
        dispersion, flux, ivar, metadata = state
        self._dispersion = dispersion
        self._flux = flux
        self._ivar = ivar
        self.metadata = metadata
        return None


    def copy(self):
        """
        Create a copy of the spectrum.
        """
        return self.__class__(
            dispersion=self.dispersion.copy(),
            flux=self.flux.copy(), ivar=self.ivar.copy(),
            metadata=self.metadata.copy())


    def gaussian_smooth(self, fwhm, **kwargs):
        
        profile_sigma = fwhm / (2 * (2*np.log(2))**0.5)
        
        # The requested FWHM is in Angstroms, but the dispersion between each
        # pixel is likely less than an Angstrom, so we must calculate the true
        # smoothing value
        
        true_profile_sigma = profile_sigma / np.median(np.diff(self.dispersion))
        smoothed_flux = ndimage.gaussian_filter1d(self.flux, true_profile_sigma, **kwargs)
        
        # TODO modify ivar based on smoothing?
        return self.__class__(self.dispersion, smoothed_flux, self.ivar.copy(), metadata=self.metadata.copy())
        
    
    def linterpolate(self, new_dispersion, fill_value=0.):
        """ Straight up linear interpolation of flux and ivar onto a new dispersion """
        new_flux = np.interp(new_dispersion, self.dispersion, self.flux, left=fill_value, right=fill_value)
        new_ivar = np.interp(new_dispersion, self.dispersion, self.ivar, left=fill_value, right=fill_value)
        return self.__class__(new_dispersion, new_flux, new_ivar, metadata=self.metadata.copy())
        

    def smooth(self, window_len=11, window='hanning'):
        """Smooth the data using a window with requested size.
        
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        
        input:
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed spectra
            
        example:

        (This function from Heather Jacobson)
        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if self.flux.size < window_len:
            raise ValueError("input vector must be bigger than the window size")

        if window_len<3:
            return self

        available = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
        if window not in available:
            raise ValueError("window is one of {}".format(", ".join(available)))


        s = np.r_[self.flux[window_len-1:0:-1], self.flux, self.flux[-1:-window_len:-1]]
        
        if window == 'flat': #moving average
            w = np.ones(window_len, 'd')

        else:
            w = eval('np.' + window + '(window_len)', {'__builtins__': None})

        smoothed_flux = np.convolve(w/w.sum(), s,mode='valid')
        smoothed_flux = smoothed_flux[(window_len/2-1):-(window_len/2)]

        return self.__class__(self.disp, smoothed_flux, headers=self.headers)
    


    def fit_rpoly_continuum(self, order=3, getsnr=False, getcont=False):
        """
        Use robust polyfit to get normalized spectrum.
        If getcont=True, return continuum rather than normalized spectrum
        """
        dispersion = self.dispersion.copy()
        flux = self.flux.copy()
        finite = np.isfinite(dispersion) & np.isfinite(flux)
        coeff, rms = rpolyfit(dispersion[finite], flux[finite], order)
        cont = np.poly1d(coeff)(dispersion)
        if getsnr: return np.median(cont)/rms
        if getcont: return cont
        return self.__class__(dispersion, flux/cont, cont*cont*self.ivar, self.metadata)

    def fit_continuum(self, knot_spacing=200, low_sigma_clip=1.0, \
        high_sigma_clip=0.2, max_iterations=3, order=3, exclude=None, \
        include=None, additional_points=None, function='spline', scale=1.0,
        **kwargs):
        """
        Fits the continuum for a given `Spectrum1D` spectrum.
        
        Parameters
        ----
        knot_spacing : float or None, optional
            The knot spacing for the continuum spline function in Angstroms. Optional.
            If not provided then the knot spacing will be determined automatically.
        
        sigma_clip : a tuple of two floats, optional
            This is the lower and upper sigma clipping level respectively. Optional.
            
        max_iterations : int, optional
            Maximum number of spline-fitting operations.
            
        order : int, optional
            The order of the spline function to fit.
            
        exclude : list of tuple-types containing floats, optional
            A list of wavelength regions to always exclude when determining the
            continuum. Example:
            
            >> exclude = [
            >>    (3890.0, 4110.0),
            >>    (4310.0, 4340.0)
            >>  ]
            
            In the example above the regions between 3890 A and 4110 A, as well as
            4310 A to 4340 A will always be excluded when determining the continuum
            regions.

        function: only 'spline', 'poly', 'leg', 'cheb'

        scale : float
            A scaling factor to apply to the normalised flux levels.
            
        include : list of tuple-types containing floats, optional
            A list of wavelength regions to always include when determining the
            continuum.

        :param rv:
            A radial velocity correction (in km/s) to apply to the spectrum.
        """

        dispersion = self.dispersion.copy()

        exclusions = []
        continuum_indices = range(len(self.flux))

        # Snip left and right
        finite_positive_flux = np.isfinite(self.flux) * self.flux > 0

        if np.sum(finite_positive_flux) == 0:
            # No valid continuum points, return nans
            no_continuum = np.nan * np.ones_like(dispersion)
            failed_spectrum = self.__class__(dispersion=dispersion,
                flux=no_continuum, ivar=no_continuum, metadata=self.metadata)

            if kwargs.get("full_output", False):
                return (failed_spectrum, no_continuum, 0, dispersion.size - 1)
            return failed_spectrum

        function = str(function).lower()
        left = np.where(finite_positive_flux)[0][0]
        right = np.where(finite_positive_flux)[0][-1]

        # See if there are any regions we need to exclude
        if exclude is not None and len(exclude) > 0:
            exclude_indices = []
            
            if isinstance(exclude[0], float) and len(exclude) == 2:
                # Only two floats given, so we only have one region to exclude
                exclude_indices.extend(range(*np.searchsorted(dispersion, exclude)))
                
            else:
                # Multiple regions provided
                for exclude_region in exclude:
                    exclude_indices.extend(
                        range(*np.searchsorted(dispersion, exclude_region)))
        
            continuum_indices = np.sort(list(set(continuum_indices).difference(
                np.sort(exclude_indices))))
            
        # See if there are any regions we should always include
        if include is not None and len(include) > 0:
            include_indices = []
            
            if isinstance(include[0], float) and len(include) == 2:
                # Only two floats given, so we can only have one region to include
                include_indices.extend(range(*np.searchsorted(dispersion, include)))
                
            else:
                # Multiple regions provided
                for include_region in include:
                    include_indices.extend(range(*np.searchsorted(
                        dispersion, include_region)))
        
        # We should exclude non-finite values from the fit.
        non_finite_indices = np.where(~np.isfinite(self.flux * self.ivar))[0]
        continuum_indices = np.sort(list(set(continuum_indices).difference(
            non_finite_indices)))

        # We should also exclude zero or negative flux points from the fit
        zero_flux_indices = np.where(0 >= self.flux)[0]
        continuum_indices = np.sort(list(set(continuum_indices).difference(
            zero_flux_indices)))

        # Fix from Erika Holmbeck
        if order > continuum_indices.size:
        #if 1 > continuum_indices.size:
            no_continuum = np.nan * np.ones_like(dispersion)
            failed_spectrum = self.__class__(dispersion=dispersion,
                flux=no_continuum, ivar=no_continuum, metadata=self.metadata)

            if kwargs.get("full_output", False):
                return (failed_spectrum, no_continuum, 0, dispersion.size - 1)

            return failed_spectrum        

        original_continuum_indices = continuum_indices.copy()

        if knot_spacing is None or knot_spacing == 0:
            knots = []

        else:
            knot_spacing = abs(knot_spacing)
            
            end_spacing = ((dispersion[-1] - dispersion[0]) % knot_spacing) /2.
            if knot_spacing/2. > end_spacing: end_spacing += knot_spacing/2.
                
            knots = np.arange(dispersion[0] + end_spacing, 
                dispersion[-1] - end_spacing + knot_spacing, 
                knot_spacing)

            if len(knots) > 0 and knots[-1] > dispersion[continuum_indices][-1]:
                knots = knots[:knots.searchsorted(dispersion[continuum_indices][-1])]
                
            if len(knots) > 0 and knots[0] < dispersion[continuum_indices][0]:
                knots = knots[knots.searchsorted(dispersion[continuum_indices][0]):]

        # TODO: Use inverse variance array when fitting polynomial/spline.
        for iteration in range(max_iterations):
            
            # Fix from Erika Holmbeck
            if order > continuum_indices.size:
            #if 1 > continuum_indices.size:

                no_continuum = np.nan * np.ones_like(dispersion)
                failed_spectrum = self.__class__(dispersion=dispersion,
                    flux=no_continuum, ivar=no_continuum, metadata=self.metadata)

                if kwargs.get("full_output", False):
                    return (failed_spectrum, no_continuum, 0, dispersion.size - 1)

                return failed_spectrum

            splrep_disp = dispersion[continuum_indices]
            splrep_flux = self.flux[continuum_indices]
            splrep_weights = self.ivar[continuum_indices]

            median_weight = np.nanmedian(splrep_weights)

            # We need to add in additional points at the last minute here
            if additional_points is not None and len(additional_points) > 0:

                for point, flux, weight in additional_points:

                    # Get the index of the fit
                    insert_index = int(np.searchsorted(splrep_disp, point))
                    
                    # Insert the values
                    splrep_disp = np.insert(splrep_disp, insert_index, point)
                    splrep_flux = np.insert(splrep_flux, insert_index, flux)
                    splrep_weights = np.insert(splrep_weights, insert_index, 
                        median_weight * weight)

            if function == 'spline':
                if order > 5:
                    logger.warn("Splines can only have a maximum order of 5. "
                        "Limiting order value to 5.")
                    order = 5

                try:
                    tck = interpolate.splrep(splrep_disp, splrep_flux,
                        k=order, task=-1, t=knots, w=splrep_weights)

                except:
                    logger.exception("Exception in fitting continuum:")
                    continuum = np.nan * np.ones_like(dispersion)

                else:
                    continuum = interpolate.splev(dispersion, tck)

            elif function in ("poly", "polynomial"):

                coeffs = polyfit(splrep_disp, splrep_flux, order)

                popt, pcov = op.curve_fit(lambda x, *c: np.polyval(c, x), 
                    splrep_disp, splrep_flux, coeffs, 
                    # Note: Erika changed sigma to splrep_weights, not changing yet
                    sigma=self.ivar[continuum_indices], absolute_sigma=False)
                continuum = np.polyval(popt, dispersion)

            elif function in ("leg", "legendre"):
                
                coeffs = np.polynomial.legendre.legfit(splrep_disp, splrep_flux, order,
                                                       w=splrep_weights)
                
                popt, pcov = op.curve_fit(lambda x, *c: np.polynomial.legendre.legval(x, c), 
                    splrep_disp, splrep_flux, coeffs, 
                    # Note: Erika changed sigma to splrep_weights, not changing yet
                    sigma=self.ivar[continuum_indices], absolute_sigma=False)
                continuum = np.polynomial.legendre.legval(dispersion, popt)


            elif function in ("cheb", "chebyshev"):
                
                coeffs = np.polynomial.chebyshev.chebfit(splrep_disp, splrep_flux, order,
                                                         w=splrep_weights)
                
                popt, pcov = op.curve_fit(lambda x, *c: np.polynomial.chebyshev.chebval(x, c), 
                    splrep_disp, splrep_flux, coeffs, 
                    # Note: Erika changed sigma to splrep_weights, not changing yet
                    sigma=self.ivar[continuum_indices], absolute_sigma=False)
                continuum = np.polynomial.chebyshev.chebval(dispersion, popt)


            elif function in ("polysinc"):
                # sinc^2(x) * polynomial
                # sinc has 3 parameters: norm, center, shape
                # polynomial has <order> parameters
                # This is really slow because of maxfev

                # Initialize sinc
                p0 = [np.percentile(splrep_flux, 95), np.median(splrep_disp),
                      np.percentile(splrep_disp, 75) - np.percentile(splrep_disp, 25)]
                p0, p0cov = op.curve_fit(lambda x, *p: p[0]*np.sinc((x-p[1])/p[2])**2,
                                         splrep_disp, splrep_flux, p0)
                
                # Fit polysinc (this is slow but seems to work)
                _func = lambda x, *p: p[0]*np.sinc((x-p[1])/p[2])**2 * np.polyval(p[3:], x)
                p0 = [p0[0], p0[1], p0[2]] + [0. for O in range(order)]
                popt, pcov = op.curve_fit(_func, 
                    splrep_disp, splrep_flux, p0, 
                    # Note: Erika changed sigma to splrep_weights, not changing yet
                    sigma=self.ivar[continuum_indices], absolute_sigma=False, maxfev=100000)
                
                continuum = _func(dispersion, *popt)

            elif function in ("trig","sincos"):
                # Casey+2016
                
                raise NotImplementedError


            else:
                raise ValueError("Unknown function type: only spline or poly "\
                    "available ({} given)".format(function))
            
            difference = continuum - self.flux
            sigma_difference = difference \
                / np.std(difference[np.isfinite(self.flux)])

            # Clipping
            upper_exclude = np.where(sigma_difference > high_sigma_clip)[0]
            lower_exclude = np.where(sigma_difference < -low_sigma_clip)[0]
            
            exclude_indices = list(upper_exclude)
            exclude_indices.extend(lower_exclude)
            exclude_indices = np.array(exclude_indices)
            
            if len(exclude_indices) == 0: break
            
            exclusions.extend(exclude_indices)
            
            # Before excluding anything, we must check to see if there are regions
            # which we should never exclude
            if include is not None:
                exclude_indices \
                    = set(exclude_indices).difference(include_indices)
            
            # Remove regions that have been excluded
            continuum_indices = np.sort(list(set(continuum_indices).difference(
                exclude_indices)))
        
        # Snip the edges based on exclude regions
        if exclude is not None and len(exclude) > 0:

            # If there are exclusion regions that extend past the left/right,
            # then we will need to adjust left/right accordingly

            left = np.max([left, np.min(original_continuum_indices)])
            right = np.min([right, np.max(original_continuum_indices)])
        
        # Apply flux scaling
        continuum *= scale

        normalized_spectrum = self.__class__(
            dispersion=dispersion[left:right],
            flux=(self.flux/continuum)[left:right], 
            ivar=(continuum * self.ivar * continuum)[left:right],
            metadata=self.metadata)

        # Return a normalized spectrum.
        continuum[:left] = np.nan
        continuum[right:] = np.nan
        if kwargs.get("full_output", False):
            return (normalized_spectrum, continuum, left, right)
        return normalized_spectrum


    def cut_wavelength(self, wlmin, wlmax=None):
        if wlmax is None: wlmin, wlmax = wlmin
        ii = np.logical_and(self.dispersion > wlmin, self.dispersion < wlmax)
        return self.__class__(self.dispersion[ii], self.flux[ii], self.ivar[ii], self.metadata)

    def get_knots(self, knot_spacing, exclude=None):
        """
        This is a hack to get the knots used in the fit_continuum spline
        """
        dispersion = self.dispersion.copy()

        continuum_indices = range(len(self.flux))
        
        # See if there are any regions we need to exclude
        if exclude is not None and len(exclude) > 0:
            exclude_indices = []
            
            if isinstance(exclude[0], float) and len(exclude) == 2:
                # Only two floats given, so we only have one region to exclude
                exclude_indices.extend(range(*np.searchsorted(dispersion, exclude)))
                
            else:
                # Multiple regions provided
                for exclude_region in exclude:
                    exclude_indices.extend(
                        range(*np.searchsorted(dispersion, exclude_region)))
        
            continuum_indices = np.sort(list(set(continuum_indices).difference(
                np.sort(exclude_indices))))

        # We should exclude non-finite values from the fit.
        non_finite_indices = np.where(~np.isfinite(self.flux * self.ivar))[0]
        continuum_indices = np.sort(list(set(continuum_indices).difference(
            non_finite_indices)))

        # We should also exclude zero or negative flux points from the fit
        zero_flux_indices = np.where(0 >= self.flux)[0]
        continuum_indices = np.sort(list(set(continuum_indices).difference(
            zero_flux_indices)))


        if knot_spacing is None or knot_spacing == 0 or continuum_indices == []:
            knots = []
        else:
            knot_spacing = abs(knot_spacing)
            
            end_spacing = ((dispersion[-1] - dispersion[0]) % knot_spacing) /2.
            if knot_spacing/2. > end_spacing: end_spacing += knot_spacing/2.
                
            knots = np.arange(dispersion[0] + end_spacing, 
                dispersion[-1] - end_spacing + knot_spacing, 
                knot_spacing)

            try:
                if len(knots) > 0 and knots[-1] > dispersion[continuum_indices][-1]:
                    knots = knots[:knots.searchsorted(dispersion[continuum_indices][-1])]
                
                if len(knots) > 0 and knots[0] < dispersion[continuum_indices][0]:
                    knots = knots[knots.searchsorted(dispersion[continuum_indices][0]):]
            except IndexError:
                print("Spectrum error: continuum indices:",continuum_indices)
                knots = []
        return knots
    
    def add_noise(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        noise = np.sqrt(1./self.ivar) * np.random.randn(len(self.flux))
        return self.__class__(self.dispersion, self.flux + noise, self.ivar, self.metadata)

    

def compute_dispersion(aperture, beam, dispersion_type, dispersion_start,
    mean_dispersion_delta, num_pixels, redshift, aperture_low, aperture_high,
    weight=1, offset=0, function_type=None, order=None, Pmin=None, Pmax=None,
    *coefficients):
    """
    Compute a dispersion mapping from a IRAF multi-spec description.

    :param aperture:
        The aperture number.

    :param beam:
        The beam number.

    :param dispersion_type:
        An integer representing the dispersion type:

        0: linear dispersion
        1: log-linear dispersion
        2: non-linear dispersion

    :param dispersion_start:
        The value of the dispersion at the first physical pixel.

    :param mean_dispersion_delta:
        The mean difference between dispersion pixels.

    :param num_pixels:
        The number of pixels.

    :param redshift:
        The redshift of the object. This is accounted for by adjusting the
        dispersion scale without rebinning:

        >> dispersion_adjusted = dispersion / (1 + redshift)

    :param aperture_low:
        The lower limit of the spatial axis used to compute the dispersion.

    :param aperture_high:
        The upper limit of the spatial axis used to compute the dispersion.

    :param weight: [optional]
        A multiplier to apply to all dispersion values.

    :param offset: [optional]
        A zero-point offset to be applied to all the dispersion values.

    :param function_type: [optional]
        An integer representing the function type to use when a non-linear 
        dispersion mapping (i.e. `dispersion_type = 2`) has been specified:

        1: Chebyshev polynomial
        2: Legendre polynomial
        3: Cubic spline
        4: Linear spline
        5: Pixel coordinate array
        6: Sampled coordinate array

    :param order: [optional]
        The order of the Legendre or Chebyshev function supplied.

    :param Pmin: [optional]
        The minimum pixel value, or lower limit of the range of physical pixel
        coordinates.

    :param Pmax: [optional]
        The maximum pixel value, or upper limit of the range of physical pixel
        coordinates.

    :param coefficients: [optional]
        The `order` number of coefficients that define the Legendre or Chebyshev
        polynomial functions.

    :returns:
        An array containing the computed dispersion values.
    """

    if dispersion_type in (0, 1):
        # Simple linear or logarithmic spacing
        dispersion = \
            dispersion_start + np.arange(num_pixels) * mean_dispersion_delta

        if dispersion_start == 1:
            dispersion = 10.**dispersion

    elif dispersion_type == 2:
        # Non-linear mapping.
        if function_type is None:
            raise ValueError("function type required for non-linear mapping")
        elif function_type not in range(1, 7):
            raise ValueError(
                "function type {0} not recognised".format(function_type))

        assert Pmax == int(Pmax), Pmax; Pmax = int(Pmax)
        assert Pmin == int(Pmin), Pmin; Pmin = int(Pmin)

        if function_type == 1:
            # Chebyshev polynomial.
            if None in (order, Pmin, Pmax, coefficients):
                raise TypeError("order, Pmin, Pmax and coefficients required "
                                "for a Chebyshev or Legendre polynomial")

            order = int(order)
            n = np.linspace(-1, 1, Pmax - Pmin + 1)
            temp = np.zeros((Pmax - Pmin + 1, order), dtype=float)
            temp[:, 0] = 1
            temp[:, 1] = n
            for i in range(2, order):
                temp[:, i] = 2 * n * temp[:, i-1] - temp[:, i-2]
            
            for i in range(0, order):
                temp[:, i] *= coefficients[i]

            dispersion = temp.sum(axis=1)


        elif function_type == 2:
            # Legendre polynomial.
            if None in (order, Pmin, Pmax, coefficients):
                raise TypeError("order, Pmin, Pmax and coefficients required "
                                "for a Chebyshev or Legendre polynomial")

            Pmean = (Pmax + Pmin)/2
            Pptp = Pmax - Pmin
            x = (np.arange(num_pixels) + 1 - Pmean)/(Pptp/2)
            p0 = np.ones(num_pixels)
            p1 = mean_dispersion_delta

            dispersion = coefficients[0] * p0 + coefficients[1] * p1
            for i in range(2, int(order)):
                if function_type == 1:
                    # Chebyshev
                    p2 = 2 * x * p1 - p0
                else:
                    # Legendre
                    p2 = ((2*i - 1)*x*p1 - (i - 1)*p0) / i

                dispersion += p2 * coefficients[i]
                p0, p1 = (p1, p2)

        elif function_type == 3:
            # Cubic spline.
            if None in (order, Pmin, Pmax, coefficients):
                raise TypeError("order, Pmin, Pmax and coefficients required "
                                "for a cubic spline mapping")
            s = (np.arange(num_pixels, dtype=float) + 1 - Pmin)/(Pmax - Pmin) \
              * order
            j = s.astype(int).clip(0, order - 1)
            a, b = (j + 1 - s, s - j)
            x = np.array([
                a**3,
                1 + 3*a*(1 + a*b),
                1 + 3*b*(1 + a*b),
                b**3])
            dispersion = np.dot(np.array(coefficients), x.T)

        else:
            raise NotImplementedError("function type not implemented yet")

    else:
        raise ValueError(
            "dispersion type {0} not recognised".format(dispersion_type))

    # Apply redshift correction.
    dispersion = weight * (dispersion + offset) / (1 + redshift)
    return dispersion


def common_dispersion_map(spectra, full_output=True):
    """
    Produce a common dispersion mapping for (potentially overlapping) spectra
    and minimize the number of resamples required. Pixel bin edges are
    preferred from bluer to redder wavelengths.

    :param spectra:
        A list of spectra to produce the mapping for.

    :param full_output: [optional]
        Optinally return the indices, and the sorted spectra.

    :returns:
        An array of common dispersion values. If `full_output` is set to `True`,
        then a three-length tuple will be returned, containing the common
        dispersion pixels, the common dispersion map indices for each spectrum,
        and a list of the sorted spectra.
    """

    # Make spectra blue to right.
    spectra = sorted(spectra, key=lambda s: s.dispersion[0])

    common = []
    discard_bluest_pixels = None
    for i, blue_spectrum in enumerate(spectra[:-1]):
        red_spectrum = spectra[i + 1]

        # Do they overlap?
        if blue_spectrum.dispersion[-1] >= red_spectrum.dispersion[0] \
        and red_spectrum.dispersion[-1] >= blue_spectrum.dispersion[0]:
            
            # Take the "entire" blue spectrum then discard some blue pixels from
            # the red spectrum.
            common.extend(blue_spectrum.dispersion[discard_bluest_pixels:])
            discard_bluest_pixels = red_spectrum.dispersion.searchsorted(
                blue_spectrum.dispersion[-1])

        else:
            # Can just extend the existing map, modulo the first N-ish pixels.
            common.extend(blue_spectrum.dispersion[discard_bluest_pixels:])
            discard_bluest_pixels = None

    # For the last spectrum.
    if len(spectra) > 1:
        common.extend(red_spectrum.dispersion[discard_bluest_pixels:])

    else:
        common = spectra[0].dispersion.copy()

    # Ensure that we have sorted, unique values from blue to red.
    common = np.unique(common)

    assert np.all(np.diff(common) > 0), \
        "Spectra must be contiguous from blue to red"

    if full_output:
        indices = [common.searchsorted(s.dispersion) for s in spectra]
        return (common, indices, spectra)

    return common

def stitch_old(spectra, linearize_dispersion = False, min_disp_step = 0.001):
    """
    Stitch spectra together, some of which may have overlapping dispersion
    ranges. This is a crude (knowingly incorrect) approximation: we interpolate
    fluxes without ensuring total conservation.

    :param spectra:
        A list of potentially overlapping spectra.
    
    :param linearize_dispersion:
        If True, return a linear dispersion spectrum
    :param min_disp_step:
        The minimum linear dispersion step (to avoid super huge files)
    """

    # Create common mapping.
    if linearize_dispersion:
        
        min_disp, max_disp = np.inf, -np.inf
        default_min_disp_step = min_disp_step
        min_disp_step = 999
        for spectrum in spectra:
            min_disp_step = min(min_disp_step, np.min(np.diff(spectrum.dispersion)))
            min_disp = min(min_disp, np.min(spectrum.dispersion))
            max_disp = max(max_disp, np.max(spectrum.dispersion))
        if min_disp_step < default_min_disp_step:
            min_disp_step = default_min_disp_step
        linear_dispersion = np.arange(min_disp, max_disp+min_disp_step, min_disp_step)
    
    N = len(spectra)
    dispersion, indices, spectra = common_dispersion_map(spectra, True)
    common_flux = np.zeros((N, dispersion.size))
    common_ivar = np.zeros_like(common_flux)

    for i, (j, spectrum) in enumerate(zip(indices, spectra)):
        common_flux[i, j] = np.interp(
            dispersion[j], spectrum.dispersion, spectrum.flux,
            left=0, right=0)
        common_ivar[i, j] = spectrum.ivar

    finite = np.isfinite(common_flux * common_ivar)
    common_flux[~finite] = 0
    common_ivar[~finite] = 0

    numerator = np.sum(common_flux * common_ivar, axis=0)
    denominator = np.sum(common_ivar, axis=0)

    flux, ivar = (numerator/denominator, denominator)
    
    if linearize_dispersion:
        new_flux = np.interp(linear_dispersion, dispersion, flux, left=0, right=0)
        new_ivar = np.interp(linear_dispersion, dispersion, ivar, left=0, right=0)
        return Spectrum1D(linear_dispersion, new_flux, new_ivar)

    # Create a spectrum with no header provenance.
    return Spectrum1D(dispersion, flux, ivar)
    

def common_dispersion_map2(spectra):
    # Find regions that will have individual dispersions
    lefts = []; rights = []
    dwls = []
    for spectrum in spectra:
        lefts.append(spectrum.dispersion.min())
        rights.append(spectrum.dispersion.max())
        dwls.append(np.median(np.diff(spectrum.dispersion)))
    points = np.sort(lefts + rights)
    Nregions = len(points)-1
    
    # Create dispersion map
    # Find orders in each region and use minimum dwl
    alldisp = []
    for i in range(Nregions):
        r_left = points[i]
        r_right = points[i+1]
        r_dwl = 99999.
        #print(i)
        num_spectra = 0
        for j, (left, right, dwl) in enumerate(zip(lefts, rights, dwls)):
            # if order in range, use its dwl
            if right <= r_left or left >= r_right:
                pass
            else:
                r_dwl = min(r_dwl, dwl)
                #print("{:3} {:.2f} {:.2f} {:2} {:.2f} {:.2f}".format(i, r_left, r_right, j, left, right))
                num_spectra += 1
        #print(i,num_spectra)

        # Use smallest dwl to create linear dispersion
        # Drop the last point since that will be in the next one
        alldisp.append(np.arange(r_left, r_right, r_dwl))
    alldisp = np.concatenate(alldisp)
    return alldisp

def common_dispersion_map3(spectra):
    # Find regions that will have individual dispersions
    lefts = []; rights = []
    dwls = []
    for spectrum in spectra:
        lefts.append(spectrum.dispersion.min())
        rights.append(spectrum.dispersion.max())
        dwls.append(np.median(np.diff(spectrum.dispersion)))
    points = np.sort(lefts + rights)
    Nregions = len(points)-1
    
    # Create dispersion map
    # Find orders in each region and use minimum dwl
    alldisp = []
    for i in range(Nregions):
        r_left = points[i]
        r_right = points[i+1]
        r_dwl = 99999.
        #print(i)
        num_spectra = 0
        for j, (left, right, dwl) in enumerate(zip(lefts, rights, dwls)):
            # if order in range, use its dwl
            if right <= r_left or left >= r_right:
                pass
            else:
                r_dwl = min(r_dwl, dwl)
                #print("{:3} {:.2f} {:.2f} {:2} {:.2f} {:.2f}".format(i, r_left, r_right, j, left, right))
                num_spectra += 1
        #print(i,num_spectra)

        # Use smallest dwl to create linear dispersion
        # Drop the last point since that will be in the next one
        disp = np.arange(r_left, r_right, r_dwl)
        if r_right - disp[-1] < r_dwl: disp = disp[:-1]
        alldisp.append(disp)
    alldisp = np.concatenate(alldisp)
    return alldisp

def stitch(spectra, new_dispersion=None, full_output=False):
    """
    Stitch spectra together, some of which may have overlapping dispersion
    ranges. This is a crude (knowingly incorrect) approximation: we interpolate
    fluxes without ensuring total conservation.
    Even worse, we interpolate ivar, which is explicitly not conserved.
    However the stitching is much better than before.

    :param spectra:
        A list of potentially overlapping spectra.
    """
    if new_dispersion is None:
        new_dispersion = common_dispersion_map2(spectra)
    
    N = len(spectra)
    common_flux = np.zeros((N, new_dispersion.size))
    common_ivar = np.zeros((N, new_dispersion.size))
    
    for i, spectrum in enumerate(spectra):
        common_flux[i, :] = np.interp(
            new_dispersion, spectrum.dispersion, spectrum.flux,
            left=0, right=0)
        common_ivar[i, :] = np.interp(
            new_dispersion, spectrum.dispersion, spectrum.ivar,
            left=0, right=0)

    finite = np.isfinite(common_flux * common_ivar)
    common_flux[~finite] = 0
    common_ivar[~finite] = 0

    numerator = np.sum(common_flux * common_ivar, axis=0)
    denominator = np.sum(common_ivar, axis=0)
    flux, ivar = (numerator/denominator, denominator)
    newspec = Spectrum1D(new_dispersion, flux, ivar)

    if full_output:
        return newspec, (common_flux, common_ivar)
    else:
        return newspec

def stitch_weighted(spectra, new_dispersion=None, full_output=False):
    """
    Stitch spectra together, some of which may have overlapping dispersion
    ranges. This is a crude (knowingly incorrect) approximation: we interpolate
    fluxes without ensuring total conservation.
    Even worse, we interpolate ivar, which is explicitly not conserved.
    However the stitching is much better than before.

    This one, compared to stitch(), uses the weighted standard error as the spectrum error.
    This is instead of the inverse variance sum.
    It is a more robust error estimate and accounts for differences between
    different orders.
    
    :param spectra:
        A list of potentially overlapping spectra.
    """
    if new_dispersion is None:
        new_dispersion = common_dispersion_map2(spectra)
    
    N = len(spectra)
    common_flux = np.zeros((N, new_dispersion.size))
    common_ivar = np.zeros((N, new_dispersion.size))
    
    for i, spectrum in enumerate(spectra):
        common_flux[i, :] = np.interp(
            new_dispersion, spectrum.dispersion, spectrum.flux,
            left=0, right=0)
        common_ivar[i, :] = np.interp(
            new_dispersion, spectrum.dispersion, spectrum.ivar,
            left=0, right=0)

    finite = np.isfinite(common_flux * common_ivar)
    common_flux[~finite] = 0
    common_ivar[~finite] = 0

    numerator = np.sum(common_flux * common_ivar, axis=0)
    denominator = np.sum(common_ivar, axis=0)
    flux = numerator/denominator
    numerator2 = np.sum((common_flux-flux[np.newaxis,:])**2 * common_ivar, axis=0)
    meansquare = numerator2/denominator
    ivar = 1/meansquare
    newspec = Spectrum1D(new_dispersion, flux, ivar)

    if full_output:
        return newspec, (common_flux, common_ivar)
    else:
        return newspec

def coadd(spectra, new_dispersion=None, full_output=False):
    """
    Add spectra together (using linear interpolation to do bad rebinning).
    ivar is also interpolated and added.
    This differs from stitch only because it is a direct sum
    (rather than being weighted pixel-by-pixel with ivar).
    
    I think this is more correct when summing raw counts.
    
    :param spectra:
        A list of potentially overlapping spectra.
    """
    if new_dispersion is None:
        new_dispersion = common_dispersion_map2(spectra)
    
    N = len(spectra)
    common_flux = np.zeros((N, new_dispersion.size))
    common_ivar = np.zeros((N, new_dispersion.size))
    
    for i, spectrum in enumerate(spectra):
        common_flux[i, :] = np.interp(
            new_dispersion, spectrum.dispersion, spectrum.flux,
            left=0, right=0)
        common_ivar[i, :] = np.interp(
            new_dispersion, spectrum.dispersion, spectrum.ivar,
            left=0, right=0)

    finite = np.isfinite(common_flux * common_ivar)
    common_flux[~finite] = 0
    common_ivar[~finite] = 0

    flux = np.sum(common_flux, axis=0)
    ivar = 1./np.sum(1./common_ivar, axis=0)
    newspec = Spectrum1D(new_dispersion, flux, ivar)

    if full_output:
        return newspec, (common_flux, common_ivar)
    else:
        return newspec
    

def read_mike_spectrum(fname, fluxband=2, skip_assert=False):
    """
    Output an OrderedDict of Spectrum1D, indexed by order number
    
    Spectrum1D.dispersion computed from multispec header
    Spectrum1D.flux depends on flux band
    
    BANDID1 = 'sky spectrum'
    BANDID2 = 'object spectrum' (default)
    BANDID3 = 'noise spectrum'
    BANDID4 = 'signal-to-noise spectrum'
    BANDID5 = 'lamp spectrum'
    BANDID6 = 'flat spectrum'
    BANDID7 = 'object spectrum divided by normed flat'
    
    Spectrum1D.ivar: 
    If fluxband==2, uses (BANDID3)**-2
    If fluxband==7, modifies ivar to account for flat
    Otherwise, just uses 1/fluxband
    """
    if not skip_assert: assert fluxband in [1,2,3,4,5,6,7]
    
    with fits.open(fname) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        metadata = OrderedDict()
        for k, v in header.items():
            if k in metadata:
                metadata[k] += v
            else:
                metadata[k] = v
        assert metadata["CTYPE1"].upper().startswith("MULTISPE") \
            or metadata["WAT0_001"].lower() == "system=multispec"
    
    if not skip_assert: assert data.shape[0] == 7, data.shape
    
    # Get order numbers
    order_nums = []
    i, key_fmt = 0, "ECORD{}"
    while key_fmt.format(i) in metadata:
        order_nums.append(metadata[key_fmt.format(i)])
        i += 1
    
    if not skip_assert: assert data.shape[1] == len(order_nums), (data.shape, len(order_nums))
    Norder = data.shape[1]
    
    Npix = data.shape[2]
    
    # Get flux data
    fluxes = [data[fluxband-1,iorder] for iorder in range(Norder)]
    # Get ivar data
    if fluxband == 2:
        ivars = [data[2,iorder]**-2. for iorder in range(Norder)]
    elif fluxband == 7:
        flats = [data[1,iorder]/data[6,iorder] for iorder in range(Norder)]
        ivars = [(flats[iorder]/data[2,iorder])**2. for iorder in range(Norder)]
    else:
        # SNR estimate
        snr = [estimate_snr(data[fluxband-1,iorder]) for iorder in range(Norder)]
        ivars = [np.full(Npix, snr**2.) for snr in snr]
        #ivars = [data[fluxband-1,iorder]**-1. for iorder in range(Norder)]
        #ivars = [np.max(np.vstack([ivar,np.zeros_like(ivar)]),axis=0) for ivar in ivars]

    for ivar in ivars:
        ivar[np.isnan(ivar)] = 0.
    
    # Join the WAT keywords for dispersion mapping.
    i, concatenated_wat, key_fmt = (1, str(""), "WAT2_{0:03d}")
    while key_fmt.format(i) in metadata:
        value = metadata[key_fmt.format(i)]
        concatenated_wat += value + (" "  * (68 - len(value)))
        i += 1
    # Split the concatenated header into individual orders.
    order_mapping = np.array([map(float, each.rstrip('" ').split()) \
        for each in re.split('spec[0-9]+ ?= ?"', concatenated_wat)[1:]])
    if len(order_mapping)==0:
        NAXIS1 = metadata["NAXIS1"]
        NAXIS2 = metadata["NAXIS2"]
        waves = np.array([np.arange(NAXIS1) for _ in range(NAXIS2)])
    else:
        waves = np.array(
                [compute_dispersion(*mapping) for
                 mapping in order_mapping])
    
    spec1ds = OrderedDict()
    for iord,wave,flux,ivar in zip(order_nums, waves, fluxes, ivars):
        meta = metadata.copy()
        meta["order"] = iord
        spec = Spectrum1D(wave, flux, ivar, meta)
        spec1ds[iord] = spec
    return spec1ds

def estimate_snr(flux, window=51):
    """Use median filter and biweight scale to estimate SNR """
    continuum = signal.medfilt(flux, window)
    noise = biweight_scale(flux/continuum)
    return 1./noise

def alex_continuum_fit(spec, knot_spacing=5., Nknots=None,
                       spline_order=2, median_window=51,
                       sigmaclip=1.5, knot_reject_threshold=0.7, maxiter=5,
                       full_output=False):
    """
    Alternate algorithm:
    * Setup initial knots according to some spacing (not knot number!)
    * Iterate:
        * Mark outlier pixels using linear interpolation between knots
        * Rolling window to count outliers
        * If a knot has < a threshold continuum pixels, remove it (but do not expand the spacing)
        * Update knot y
        * Refit continuum



    A hybrid of my own stuff and linetools
    Algorithm:
    * Setup initial knots (Nknots linearly spaced inside)
    * Get 0th order median continuum (using window)
    * Iterate:
        * Mark outlier pixels using linear interpolation between knots
        * Update knot y
        * Remove bad knots? (those with all outlier pixels)
        * Refit continuum through knots with lsq spline

    """
    x, y, w = spec.dispersion, spec.flux, spec.ivar
    # 0th order median 
    c0 = signal.medfilt(y, median_window)
    n0 = y/c0
    w0 = w*c0*c0
    n0[np.isnan(n0)] = 0.
    
    # Generate evenly spaced knots in the interior
    wmin, wmax = x.min(), x.max()
    if Nknots is None:
        waverange = wmax - wmin
        Nknots = int(waverange // knot_spacing)
        minknot = wmin + (waverange - Nknots * knot_spacing)/2.
        maxknot = wmax - (waverange - Nknots * knot_spacing)/2.
        xknots = np.arange(minknot, maxknot, knot_spacing)
    else:
        xknots = np.linspace(wmin,wmax,Nknots+2)[1:-1]
        knot_spacing = np.median(np.diff(xknots))
    assert len(xknots) == Nknots, (len(xknots), Nknots)
    
    # Generate a spacing window for convolution
    spacing_window = int(knot_spacing/np.median(np.diff(x)))
    if spacing_window % 2 == 0: spacing_window += 1
    average_filter = np.ones(spacing_window)/float(spacing_window)
    
    ynorm = n0.copy()
    for iter in range(maxiter):
        # Assign knot y-values from median windowing
        ixknots = np.searchsorted(x, xknots)
        yknots = signal.medfilt(y, spacing_window)[ixknots]

        # mark good pixels
        flin = interpolate.interp1d(xknots, yknots, fill_value="extrapolate")
        lincont = flin(x) #np.interp(x, xknots, yknots, left=1.0, right=1.0)
        good = np.abs((y-lincont)*np.sqrt(w)) < sigmaclip
        #logger.debug("iter{}: Found {} good points".format(iter+1, np.sum(good)))
        # count good pixel fraction in spacing window
        goodfrac = np.convolve(good.astype(int), average_filter, 'same')
        # reject knots that have too few continuum points
        iibadknots = goodfrac[ixknots] < knot_reject_threshold
        #logger.debug("iter{}: Rejecting {}/{} knots".format(iter+1, np.sum(iibadknots), len(iibadknots)))
        xknots = xknots[~iibadknots]
        yknots = yknots[~iibadknots]
        
        #return x, y, w, xknots, yknots, lincont, good, goodfrac
        # Fit spline through knots
        fnorm = interpolate.LSQUnivariateSpline(x[good], y[good], xknots, w[good], k=spline_order)
        #fnorm = interpolate.Akima1DInterpolator(xknots, yknots)
        ynorm = fnorm(x)
        #chi2 = np.nansum((y[good] - ynorm[good])**2.*w[good])/float(np.sum(good))
        
        #if np.sum(iibadknots) == 0: break
        #if chi2 < 1.5:
        #    logger.info("Chi2={:.2f}".format(chi2))
        #    break
    
    if full_output:
        return fnorm, ynorm, xknots, yknots, good
    return fnorm

def write_fits_linear(fname, wmin, dwave, flux):
    hdu = fits.PrimaryHDU(np.array(flux))
    headers = {}
    headers.update({
            'CTYPE1': 'LINEAR  ',
            'CRVAL1': wmin,
            'CRPIX1': 1,
            'CDELT1': dwave
    })
    for k,v in headers.items():
        hdu.header[k] = v
    hdu.writeto(fname, overwrite=True)
    
