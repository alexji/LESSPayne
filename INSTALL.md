
Installation
============

0.  Download [git]

1.  Install [Anaconda Python](https://www.continuum.io/downloads). For now, select Python 2.7 until some [PySide issues](https://github.com/andycasey/smhr/issues/14) are resolved.

2.  Clone SMH from the [GitHub repository](https://github.com/andycasey/smhr) using this terminal command:

    ``git clone git@github.com:andycasey/smhr.git``
    
    This will create a directory called `smhr`, where everything lives.
  
3.  Finally, you will need `MOOGSILENT` installed. If you don't have it already installed, first make sure you have a FORTRAN compiler installed (e.g., [gfortran](https://gcc.gnu.org/wiki/GFortran)). In OSX you can install `gfortran` and other command line tools with the terminal command `xcode-select --install`. Then to install `MOOGSILENT`:

  ``pip install moogsilent``
  
brief other problems:
4. conda install pyside
5. remove pyside in setup.py
6. python setup.py install --with-models
7. cp default sessions file

That's it. Currently to run SMH, `cd` to the `smhr` folder then type:

````python
ipython
run -i smh/gui/__main__.py
````

(Eventually this will be packaged into a `smh` command line tool or packaged into a clickable application.)