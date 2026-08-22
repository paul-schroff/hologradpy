# HoloGradPy
<a href="https://www.gnu.org/licenses/lgpl-3.0.en.html"><img alt="License: LGPL v3" src="https://img.shields.io/badge/License-LGPL_v3-blue.svg"></a>

Accurate SLM holography from a calibrated, differentiable model of the optical setup.

For reference, have a look at the [documentation](https://hologradpy.readthedocs.io/en/latest/). The results in our publications, listed below, were generated using the 
algorithms in this package.

<!-- docs-start -->

## Features

- **Phase retrieval**: holograms for arbitrary target light potentials by conjugate
  gradient minimisation <a href="#ref1">[1]</a> of flexible cost functions, with gradients from automatic
  differentiation.
- **Camera feedback**: a feedback algorithm <a href="#ref2">[2]</a> measures the light potential on the
  camera and corrects residual experimental errors.
- **SLM calibration**: wavefront calibration by an interferometric raster scan or by
  rapid stochastic speckle fitting, camera mapping, and pixel crosstalk calibration
  with the kernel models of our publications.
- **Differentiable optical models**: the SLM-to-camera optical path is built from
  PyTorch modules, so simulated camera images are differentiable with respect to the
  SLM phase and every model parameter.
- **Hardware interface**: a small native SLM and camera layer with fully simulated
  devices for development without hardware, and adapters for
  [slmsuite](https://slmsuite.readthedocs.io/en/latest/), which covers most
  commercially available SLMs and many camera models.

## About

I (Paul Schroff) wrote this package during my PhD in Prof. Stefan Kuhr's lab at the
[Ultracold Matter and Quantum Technology](https://umqt.phys.strath.ac.uk/) group of the
University of Strathclyde, where we investigated SLM-generated potentials for quantum
gas microscopes.

I now maintain it in Prof. Jonathan Pritchard's
[SQuAre](https://umqt.phys.strath.ac.uk/ryd-projects/scalable-qubit-arrays/) lab, where
its algorithms are used for top-hat beam shaping. 
Many thanks to Daniel Walker, who turned some of my PhD spaghetti code into something
modular and easy to use for our atom array experiments. 
Much of what I learned from him has fed back into this package.

For questions or suggestions, email [paul.schroff@strath.ac.uk](mailto:paul.schroff@strath.ac.uk).

## Publications

This package accompanies the following work. If you use it, please cite whichever is
relevant to what you used.

**Phase retrieval with camera feedback using pixel crosstalk modelling**

> P. Schroff, A. La Rooij, E. Haller and S. Kuhr,
> *Accurate holographic light potentials using pixel crosstalk modelling*,
> Sci. Rep. **13**, 3252 (2023).
> [https://doi.org/10.1038/s41598-023-30296-6](https://doi.org/10.1038/s41598-023-30296-6)

**Speckle calibration and crosstalk optimisation**

> P. Schroff, E. Haller, S. Kuhr and A. La Rooij,
> *Rapid stochastic spatial light modulator calibration and pixel crosstalk optimization*,
> Opt. Express **32**, 48957 (2024).
> [https://doi.org/10.1364/OE.539548](https://doi.org/10.1364/OE.539548)

## References

1. <a id="ref1"></a>T. Harte, G. D. Bruce, J. Keeling and D. Cassettari,
   *Conjugate gradient minimisation approach to generating holographic traps for
   ultracold atoms*,
   Opt. Express **22**, 26548 (2014).
   <https://doi.org/10.1364/OE.22.026548>

2. <a id="ref2"></a>G. D. Bruce, M. Y. H. Johnson, E. Cormack, D. A. W. Richards, J. 
   Mayoh and D. Cassettari,
   *Feedback-enhanced algorithm for aberration correction of holographic atom traps*,
   J. Phys. B: At. Mol. Opt. Phys. **48**, 115303 (2015).
   <https://doi.org/10.1088/0953-4075/48/11/115303>
