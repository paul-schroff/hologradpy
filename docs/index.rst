HoloGradPy Documentation
########################

.. div:: sd-d-flex-row sd-align-major-start sd-gap-2 sd-mb-4

   .. button-link:: https://github.com/paul-schroff/hologradpy
      :color: primary
      :outline:

      :octicon:`mark-github` Source code

   .. button-link:: https://www.gnu.org/licenses/lgpl-3.0.en.html
      :color: primary
      :outline:

      :octicon:`law` LGPL-3.0

   .. button-link:: #about
      :color: primary
      :outline:

      :octicon:`info` About

   .. button-link:: #citing-hologradpy
      :color: primary
      :outline:

      :octicon:`book` Publications

.. warning::
   This package and its documentation are work in progress and might change without
   notice.

.. sidebar:: Top-hat beam
   :class: animation

   .. image:: /_static/top_hat_beam_shaping.gif
      :alt: A Gaussian focal spot being shaped into a top hat
      :target: auto_examples/phase_retrieval/top_hat_beam_shaping.html

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- intro-start -->
   :end-before: <!-- intro-end -->

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: sd-mt-4

   .. grid-item-card:: :octicon:`download` Installation
      :link: install
      :link-type: doc

      What to install, and how to get CUDA acceleration working.

   .. grid-item-card:: :octicon:`beaker` Example scripts
      :link: examples
      :link-type: doc

      Worked examples, from phase retrieval and vortex annihilation to SLM
      calibration against a camera.

   .. grid-item-card:: :octicon:`book` API reference
      :link: autoapi/hologradpy/index
      :link-type: doc

      Every module, class and function, generated from the source.

.. hint::
   The algorithms in this package work best with CUDA acceleration, especially when
   modelling pixel crosstalk on the SLM.

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- docs-start -->
   :end-before: <!-- contact-start -->

.. card:: :octicon:`mail` Contact
   :link: mailto:paul.schroff@strath.ac.uk
   :link-type: url

   Contact Paul Schroff at paul.schroff@strath.ac.uk for questions or suggestions.

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- contact-end -->

.. toctree::
   :hidden:
   :caption: User Guide

   install
   examples
   API Reference <autoapi/hologradpy/index>

.. toctree::
   :hidden:
   :caption: Links

   Sci. Rep. 13, 3252 (2023) <https://doi.org/10.1038/s41598-023-30296-6>
   Opt. Express 32, 48957 (2024) <https://doi.org/10.1364/OE.539548>
   GitHub <https://github.com/paul-schroff/hologradpy>

