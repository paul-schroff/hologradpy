.. _Examples:

Example scripts
===============

Worked examples, from driving the hardware to computational holography, calibrating the
optical setup, and running camera feedback. The gallery is published on its own, and its
scripts are in the `hologradpy-examples <https://github.com/paul-schroff/hologradpy-examples>`_
repository.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`gear` Hardware interface
      :link: https://hologradpy.readthedocs.io/projects/examples/en/latest/auto_examples/hardware_interface/index.html
      :link-type: url

      Talking to cameras and SLMs through the native device interface and setting up
      simulated hardware for development and testing.

   .. grid-item-card:: :octicon:`cpu` Computational holography
      :link: https://hologradpy.readthedocs.io/projects/examples/en/latest/auto_examples/phase_retrieval/index.html
      :link-type: url

      Optimizing the SLM phase pattern for a target intensity profile in the Fourier
      plane.

   .. grid-item-card:: :octicon:`sync` Camera feedback
      :link: https://hologradpy.readthedocs.io/projects/examples/en/latest/auto_examples/camera_feedback/index.html
      :link-type: url

      Measuring the intensity profile on the camera and correcting for residual errors
      the model cannot predict.

   .. grid-item-card:: :octicon:`tools` Calibration
      :link: https://hologradpy.readthedocs.io/projects/examples/en/latest/auto_examples/calibration/index.html
      :link-type: url

      Camera mapping, wavefront calibration and pixel crosstalk.
