import sys
from setuptools import setup, find_packages
from codecs import open
from os import path, system
from re import compile as re_compile
from urllib.request import urlretrieve


def read(filename):
    kwds = {"encoding": "utf-8"} if sys.version_info[0] >= 3 else {}
    with open(filename, **kwds) as fp:
        contents = fp.read()
    return contents


# Get the version information.
version = "0.1"

# External data.
if "--without-models" not in sys.argv:
    data_paths = [
        # Model photospheres:
        # Castelli & Kurucz (2004)
        (
            "https://zenodo.org/record/14964/files/castelli-kurucz-2004.pkl",
            "LESSPayne/smh/photospheres/castelli-kurucz-2004.pkl",
        ),
        # MARCS (2008)
        (
            "https://zenodo.org/record/14964/files/marcs-2011-standard.pkl",
            "LESSPayne/smh/photospheres/marcs-2011-standard.pkl",
        ),
    ]
    for url, filename in data_paths:
        if path.exists(filename):
            print("Skipping {0} because file already exists".format(filename))
            continue
        print("Downloading {0} to {1}".format(url, filename))
        try:
            urlretrieve(url, filename)
        except IOError:
            raise (
                "Error downloading file {} -- consider installing with flag "
                "--without-models".format(url))
else:
    sys.argv.remove("--without-models")

setup(
    name="LESSPayne",
    version=version,
    author="Alex Ji",
    author_email="alexji@uchicago.edu",
    description="LESSPayne",
    long_description="Labeling Echelle Spectra with SMHR and Payne",
    url="https://github.com/alexji/LESSPayne",
    license="MIT",
    packages=find_packages(exclude=["documents", "tests"]),
    install_requires=[
        "numpy", "scipy>=0.14.0", "six", "astropy", "pyyaml", "requests"
    ],
    package_data={
        "": ["LICENSE"],
        "LESSPayne": [
            "data/default.yaml",
            "data/template_spectra/*",
            "data/NN_normalized_spectra_float16_fixwave.npz",
        ],
        "LESSPayne.PayneEchelle": [
            "other_data/*.npz",
            "other_data/*.fits",
            "other_data/*.txt",
        ],
        "LESSPayne.smh": ["default_session.yaml"],
        "LESSPayne.smh.data.isotopes": [
            "asplund09_isotopes.pkl",
            "asplund09_isotopes_unicode.pkl",
            "sneden08_all_isotopes.pkl",
            "sneden08_all_isotopes_unicode.pkl",
            "sneden08_rproc_isotopes.pkl",
            "sneden08_rproc_isotopes_unicode.pkl",
            "sneden08_sproc_isotopes.pkl",
            "sneden08_sproc_isotopes_unicode.pkl",
        ],
        "LESSPayne.smh.data.spectra": [
            "cd-38_245.fits",
            "cs22892-52.fits",
            "g64-12.fits",
            "hd122563.fits",
            "hd140283.fits",
            "he1523-0901.fits",
        ],
        "LESSPayne.smh.gui": ["matplotlibrc"],
        "LESSPayne.smh.photospheres": [
            "marcs-2011-standard.pkl",
            "castelli-kurucz-2004.pkl",
            "stagger-2013-optical.pkl",
            "stagger-2013-mass-density.pkl",
            "stagger-2013-rosseland.pkl",
            "stagger-2013-height.pkl",
        ],
        "LESSPayne.smh.radiative_transfer.moog": [
            "defaults.yaml",
            "abfind.in",
            "synth.in",
        ],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": ["import_test=import_test.__main__:main"]
    },
    # data_files=None,
    # entry_points=None
)
