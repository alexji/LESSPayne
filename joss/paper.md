---
title: 'LESSPayne: Labeling Echelle Spectra with SMHR and Payne'
tags:
  - Python
  - astronomy
authors:
  - name: Alexander P. Ji
    orcid: 0000-0002-4863-8842
    equal-contrib: true
    affiliation: "1, 2, 3, 4"
    corresponding: true
  - name: Andrew R. Casey
    orcid: 0000-0003-0174-0564
    equal-contrib: true
    affiliation: "5, 6"
  - name: Yuan-Sen Ting (丁源森)
    orcid: 0000-0001-5082-9536
    equal-contrib: true
    affiliation: "7, 8"
  - name: Erika M. Holmbeck
    orcid: 0000-0002-5463-6800
    affilation: "4, 9"
  - name: Anna Frebel
    orcid: 0000-0002-2139-7145
    affiliation: "4, 10"
  - name: Rana Ezzeddine
    orcid: 0000-0002-8504-8470
    affiliation: "4, 11"
affiliations:
  - name: Department of Astronomy & Astrophysics, University of Chicago, 5640 South Ellis Avenue, Chicago, IL 60637, USA
    index: 1
  - name: Kavli Institute for Cosmological Physics, University of Chicago, Chicago, IL 60637, USA
    index: 2
  - name: NSF-Simons AI Institute for the Sky (SkAI), 172 E. Chestnut St., Chicago, IL 60611, USA
    index: 3
  - name: Joint Institute for Nuclear Astrophysics—Center for Evolution of the Elements, USA
    index: 4
  - name: School of Physics & Astronomy, Monash University, Wellington Road, Clayton, VIC 3800, Australia
    index: 5
  - name: ARC Centre of Excellence for All Sky Astrophysics in 3 Dimensions (ASTRO 3D), Canberra, ACT 2611, Australia
    index: 6
  - name: Department of Astronomy, The Ohio State University, Columbus, OH 43210, USA
    index: 7
  - name: Center for Cosmology and AstroParticle Physics (CCAPP), The Ohio State University, Columbus, OH 43210, USA
    index: 8
  - name: Lawrence Livermore National Laboratory, 7000 East Avenue, Livermore, CA 94550, USA
    index: 9
  - name: Department of Physics & Kavli Institute for Astrophysics and Space Research, Massachusetts Institute of Technology, Cambridge, MA 02139, USA
    index: 10
  - name: Department of Astronomy, University of Florida, 211 Bryant Space Sciences Center, Gainesville, Florida, 32611, USA
    index: 11

date: 14 January 2025
bibliography: paper.bib

[//]: # # Optional fields if submitting to a AAS journal too, see this blog post:
[//]: # # https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
[//]: # # aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
[//]: # # aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

High-resolution stellar spectra can be used to determine chemical abundances.

LESSPayne performs a semiautomatic classical analysis.
Radial velocity measurement,
order normalization and stitching,
equivalent width fitting,
model atmosphere interpolation,
synthetic spectrum fitting.

Ingredients:
- PayneEchelle
- SMHR
- ATLAS model atmospheres
- MOOG radiative transfer

The goal is to be able to rapidly analyze high-resolution spectra.
The fully automatic results can run "on-the-fly" during an observing run
to produce inital stellar parameter and abundance estimates.
However the final results need to be visually inspected and adjusted
for quality before they are publication-worthy.

In principle, an interface is present to change the model atmosphere library
and radiative transfer code.
It is also possible to change the PayneEchelle emulator to include
more parameters or do NLTE fits (e.g. Li+in prep).
For now none of these are implemented.

## Step 1: PayneEchelle Fit
## Step 2: Order normalization and stitching
## Step 3: Equivalent width fits
## Step 4: Stellar parameter specification
## Step 5: Synthetic spectrum fits
## Step 6: Refit synthetic spectra (TODO)
## Step 7: Uncertainty Propagation
## Step 8: Abundance Summary

# Statement of need

Chemical abundances from highres stellar spectroscopy.
Often done fully manually, which is slow and heterogeneous.
Fully automatic results are more consistent but hard to get all the elements
and less reliable results.
We want something that splits the difference:
analyze spectra automatically, but with a graphical user interface (GUI)
that allows the user to inspect and manually change individual line fits as needed.
The expected scale is between 10s to 1000s of stars.

This package was written in response to two large surveys:
the R-Process Alliance Snapshot survey, and
the SDSS-V Low Alpha Metal Poor Star followup survey.
These used medium-high resolution (R~30,000) optical echelle spectra
with moderate S/N ~ 30 per pixel, spanning at least 3500A-8000A.


[//]: # `Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
[//]: # enables wrapping low-level languages (e.g., C) for speed without losing
[//]: # flexibility or ease-of-use in the user-interface. The API for `Gala` was
[//]: # designed to provide a class-based and user-friendly interface to fast (C or
[//]: # Cython-optimized) implementations of common operations such as gravitational
[//]: # potential and force evaluation, orbit integration, dynamical transformations,
[//]: # and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
[//]: # interfaces well with the implementations of physical units and astronomical
[//]: # coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
[//]: # `astropy.coordinates`).

[//]: # `Gala` was designed to be used by both astronomical researchers and by
[//]: # students in courses on gravitational dynamics or astronomy. It has already been
[//]: # used in a number of scientific publications [@Pearson:2017] and has also been
[//]: # used in graduate courses on Galactic dynamics to, e.g., provide interactive
[//]: # visualizations of textbook material [@Binney:2008]. The combination of speed,
[//]: # design, and support for Astropy functionality in `Gala` will enable exciting
[//]: # scientific explorations of forthcoming data releases from the *Gaia* mission
[//]: # [@gaia] by students and experts alike.
[//]: # 
[//]: # # Mathematics
[//]: # 
[//]: # Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$
[//]: # 
[//]: # Double dollars make self-standing equations:
[//]: # 
[//]: # $$\Theta(x) = \left\{\begin{array}{l}
[//]: # 0\textrm{ if } x < 0\cr
[//]: # 1\textrm{ else}
[//]: # \end{array}\right.$$
[//]: # 
[//]: # You can also use plain \LaTeX for equations
[//]: # \begin{equation}\label{eq:fourier}
[//]: # \hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
[//]: # \end{equation}
[//]: # and refer to \autoref{eq:fourier} from text.

[//]: # # Citations
[//]: # 
[//]: # Citations to entries in paper.bib should be in
[//]: # [rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
[//]: # format.
[//]: # 
[//]: # If you want to cite a software repository URL (e.g. something on GitHub without a preferred
[//]: # citation) then you can do it with the example BibTeX entry below for @fidgit.
[//]: # 
[//]: # For a quick reference, the following citation commands can be used:
[//]: # - `@author:2001`  ->  "Author et al. (2001)"
[//]: # - `[@author:2001]` -> "(Author et al., 2001)"
[//]: # - `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

[//]: # # Figures
[//]: # 
[//]: # Figures can be included like this:
[//]: # ![Caption for example figure.\label{fig:example}](figure.png)
[//]: # and referenced from text using \autoref{fig:example}.
[//]: # 
[//]: # Figure sizes can be customized by adding an optional second parameter:
[//]: # ![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from ...
APJ acknowledges support from NSF AST-2206264.

# References
