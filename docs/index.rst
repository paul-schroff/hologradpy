.. HoloGradPy documentation master file, created by
   sphinx-quickstart on Thu Jun  1 15:17:54 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HoloGradPy Documentation
########################

HoloGradPy provides functionality to holographically generate light potentials of
arbitrary shape using a phase-modulating SLM
(see `our publication <https://doi.org/10.1038/s41598-023-30296-6>`_).

Author
******
This package was created by Paul Schroff during his PhD at the University of
Strathclyde in the research group of Stefan Kuhr.

.. note::
   For questions or suggestions, email paul.schroff@strath.ac.uk.

Features
********
To calculate the SLM phase pattern for a given target light potential, we
implemented a phase-retrieval algorithm based on
`conjugate gradient minimisation <https://doi.org/10.1364/OE.22.026548>`_.
The gradient used in the conjugate gradient minimisation is calculated using PyTorch's
automatic differentiation capabilities.

Functions to characterise the beam profile and the constant phase at the SLM are
provided. These measurements are crucial for accurate experimental results.

We employed a `feedback algorithm <https://dx.doi.org/10.1088/0953-4075/48/11/115303>`_
and model pixel crosstalk on the SLM to further reduce experimental errors in the light
potentials.


.. note::
   This package works best with a Nvidia GPU to run the phase retrieval algorithm.

.. warning::
   This documentation is work in progress - refer to the :ref:`example scripts <Examples>` to get started.


.. toctree::
   :maxdepth: 1
   :caption: User Guide

   install
   auto_examples/index

.. toctree::
   :hidden:
   :caption: Links

   ↪ Publication <https://doi.org/10.1038/s41598-023-30296-6>
   ↪ GitHub <https://github.com/paul-schroff/hologradpy>

Cite as
*******
If you are using this code, please cite our publication:

P. Schroff, A. La Rooij, E. Haller, S. Kuhr,
Accurate holographic light potentials using pixel crosstalk modelling.
*Sci Rep* **13**, 3252 (2023). https://doi.org/10.1038/s41598-023-30296-6

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
