from . import smh, PayneEchelle, autosmh

import sys
from setuptools import setup, find_packages
from pathlib import Path
from codecs import open
from os import path, system
from re import compile as re_compile
from urllib.request import urlretrieve


def read(filename):
    kwds = {"encoding": "utf-8"} if sys.version_info[0] >= 3 else {}
    with open(filename, **kwds) as fp:
        contents = fp.read()
    return contents


dir = Path(__file__).parent.resolve()  # this filepath

data_paths = [
    # Model photospheres:
    # Castelli & Kurucz (2004)
    (
        "https://zenodo.org/record/14964/files/castelli-kurucz-2004.pkl",
        f"{dir}/smh/photospheres/castelli-kurucz-2004.pkl",
    ),
    # MARCS (2008)
    (
        "https://zenodo.org/record/14964/files/marcs-2011-standard.pkl",
        f"{dir}/smh/photospheres/marcs-2011-standard.pkl",
    ),
]
for url, filename in data_paths:
    if path.exists(filename):
        # print("Skipping {0} because file already exists".format(filename))
        continue
    print("Downloading {0} to {1}".format(url, filename))
    try:
        urlretrieve(url, filename)
    except IOError:
        print(
            "Error downloading file {} -- consider reinstalling ".format(url))
