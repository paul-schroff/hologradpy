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

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: sd-mt-4

   .. grid-item-card:: :octicon:`download` Installation
      :link: install
      :link-type: doc

      What to install, and how to get CUDA acceleration working.

   .. grid-item-card:: :octicon:`beaker` Example scripts
      :link: auto_examples/index
      :link-type: doc

      Worked examples, from phase retrieval and vortex annihilation to SLM
      calibration against a camera.

   .. grid-item-card:: :octicon:`book` API reference
      :link: autoapi/hologradpy/index
      :link-type: doc

      Every module, class and function, generated from the source.

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- docs-start -->

.. toctree::
   :hidden:
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
