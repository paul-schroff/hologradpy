HoloGradPy Documentation
########################

HoloGradPy provides functionality for generating light potentials using a 
phase-modulating SLM (see
`our publication <https://doi.org/10.1038/s41598-023-30296-6>`_).
SLM calibration methods using random speckle intensity patterns based on
`more recent work <https://doi.org/10.1364/OE.539548>`_ are included as well.

.. note::
   The algorithms in this package work best with CUDA acceleration, especially the
   speckle calibration algorithms.

.. warning::
   This documentation is work in progress. Refer to the
   :ref:`example scripts <Examples>` to get started.

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- docs-start -->

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   install
   auto_examples/index
   API Reference <autoapi/hologradpy/index>

.. toctree::
   :hidden:
   :caption: Links

   Sci. Rep. 13, 3252 (2023) <https://doi.org/10.1038/s41598-023-30296-6>
   Opt. Express 32, 48957 (2024) <https://doi.org/10.1364/OE.539548>
   GitHub <https://github.com/paul-schroff/hologradpy>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
