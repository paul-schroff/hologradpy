.. _Examples:

Example scripts
===============

Worked examples, from driving the hardware to computational holography, calibrating the
optical setup, and running camera feedback.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`gear` Hardware interface
      :link: auto_examples/hardware_interface/index
      :link-type: doc

      Talking to cameras and SLMs through the native device interface and setting up
      simulated hardware for development and testing.

   .. grid-item-card:: :octicon:`cpu` Computational holography
      :link: auto_examples/phase_retrieval/index
      :link-type: doc

      Optimising the SLM phase pattern for a target intensity profile in the Fourier
      plane.

   .. grid-item-card:: :octicon:`sync` Camera feedback
      :link: auto_examples/camera_feedback/index
      :link-type: doc

      Measuring the intensity profile on the camera and correcting for residual errors
      the model cannot predict.

   .. grid-item-card:: :octicon:`tools` Calibration
      :link: auto_examples/calibration/index
      :link-type: doc

      Camera mapping, wavefront calibration and pixel crosstalk.

.. toctree::
   :hidden:
   :maxdepth: 2

   auto_examples/hardware_interface/index
   auto_examples/phase_retrieval/index
   auto_examples/camera_feedback/index
   auto_examples/calibration/index
