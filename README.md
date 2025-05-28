LESSPayne
------------------------
Labeling Echelle Spectra with SMHR and Payne

Combines Spectroscopy Made Harder (smhr) with PayneEchelle (derived from Payne4MIKE)

Authors
-------
 - Alex Ji (University of Chicago)
 - Andrew R. Casey (Monash)
 - Yuan-Sen Ting (ANU)
 - Erika Holmbeck (Carnegie Observatories)

Installation
------------

It is strongly recommended to use anaconda and create a new environment.
PayneEchelle has very few requirements, but SMHR uses the Pyside2 GUI library.
GUIs are rather fickle with versions so we have to use specific versions of matplotlib and numpy

* Get anaconda https://www.anaconda.com/

* Create a new environment and install required libraries:
For M1 macs, note that pyside2 has to be run in Rosetta. Thus, you can install it in this way:
```
conda create -c conda-forge/osx-64 --name lesspayne python=3.8 scipy numpy=1.21.0 matplotlib=3.1.3 six astropy ipython python.app requests pyside2=5.13.2 yaml jupyter psutil
```
Currently (as of May 2022) anaconda on M1/ARM chips by default includes channels that search through `osx-arm64` and `noarch` but not `osx-64`.
Also, newer versions of pyside2 appear to have changed some syntax on dialog boxes. We will update this eventually but for now you can install the older pyside2 version.
We are also forced to use older versions of numpy and matplotlib to maintain compatability.

For older Macs or Linux, it should work to just remove `/osx-64` and `python.app` from the above

* Activate the environment, e.g. `conda activate lesspayne`
 
* Download and install this branch:
```
git clone https://github.com/alexji/LESSPayne.git 
cd LESSPayne
pip install -e .
```
Right now we install in editable/development mode, which will be fixed someday.

* Try running the GUI:
```
cd LESSPayne/smh/gui
pythonw __main__.py
```

Once this works, it is recommended you update your shell configuration to launch the GUI, e.g.
```
alias runsmhr='conda activate lesspayne; cd ~/LESSPayne/LESSPayne/smh/gui; pythonw __main__.py'
```

* Compile moog17scat (see below) and add it to your path. For example, edit your `.bash_profile` with this line:
```
export PATH=/path/to/moog17scat:$PATH
```
Then when you type `which MOOGSILENT` it should return the path to the relevant location.

* Some installation notes for Linux/Debian: the same process works but it takes a very long time to install pyside2 (hours?) so be patient. Thanks to Shivani Shah and Terese Hansen for this information.

MOOG
----
It is currently recommended that you use this version of MOOG: https://github.com/alexji/moog17scat

Follow the usual MOOG installation instructions. When you compile MOOG, make sure that you have not activated any anaconda environments, because it can mess up the gfortran flags.
Note that SMHR requires you to have an executable called `MOOGSILENT` callable from your `$PATH` environment variable. Specifically, it uses the version of MOOG that you get from `which MOOGSILENT`.

This version is modified from the 2017 February version of MOOG from Chris Sneden's website. It includes Jennifer Sobeck's scattering routines (turned on and off with the flag `scat`, which is not true in the default MOOG 2017) and the fixes to the Barklem damping that were implemented in the 2014 MOOG refactoring.
There is now a 2019 November version of MOOG, but it did not add anything different unless you use the HF molecule or work on combined spectra of globular clusters. It did also start forcing MOOG to read everything as loggf from linelists, rather than logging things if all the loggfs were positive. But in SMHR we add a fake line whenever this is detected, so it does impact anything here.

Note for M1+ macs: the default shell is now `zsh`, so if you did not change the default shell on a new computer, check to make sure you edited `.zshrc` instead of the usual `.bashrc`

Usage
-----
```
conda activate lesspayne
python ~/lib/LESSPayne/LESSPayne/cli/run.py -a mycfg.yaml # example in cli directory
```
