.. HoloGradPy documentation master file, created by
   sphinx-quickstart on Thu Jun  1 15:17:54 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HoloGradPy Documentation
========================

HoloGradPy provides functionality to holographically generate light potentials of
arbitrary shape using a phase-modulating SLM
(see `our publication <https://doi.org/10.1038/s41598-023-30296-6>`_).

To calculate the SLM phase pattern for a given target light potential, we
implemented a phase-retrieval algorithm based on
`conjugate gradient minimisation <https://doi.org/10.1364/OE.22.026548>`_.
The gradient for the conjugate gradient minimisation is calculated using PyTorch's
automatic differentiation capabilities.

Functions to characterise the beam profile and the constant phase at the SLM are
provided. These measurements are cruicial for accurate experimental results.

We employed a `feedback algorithm <https://dx.doi.org/10.1088/0953-4075/48/11/115303>`_ and model pixel
crosstalk on the SLM to further reduce experimental errors in the light potentials.

.. note::
   This package works best with a Nvidia GPU to run the phase retrieval algorithm.

.. warning::
   This documentation is work in progress - refer to the example scripts and the
   programmer reference to get started.



User Guide
==========

.. toctree::
   :maxdepth: 2

   install
   auto_examples/index

Programmer Reference
--------------------

.. toctree::
   :maxdepth: 2

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
