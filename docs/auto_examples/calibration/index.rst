:orphan:

.. _Calibration:

Calibration
===========

Calibrating the optical setup, so the model matches the experiment better.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>


Camera mapping
--------------

Mapping the camera coordinates relative to the SLM's Fourier plane.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Displaying a checkerboard and detecting its corners, fitting an affine transform  between the image plane and the sensor.">

.. only:: html

  .. image:: /auto_examples/calibration/camera_mapping/images/thumb/sphx_glr_checkerboard_camera_mapping_thumb.png
    :alt:

  :doc:`/auto_examples/calibration/camera_mapping/checkerboard_camera_mapping`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Camera mapping from a checkerboard</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sequentially displays linear phases on the SLM and detects the position of the resulting focal spots on the camera, fitting an partial affine transform (translation, scale, and rotation) between the camera coordinates and the coordinates of the simulated output plane. Robust to aberrations and works with the zeroth order on or off the sensor. Figures out any flips or rotations of the camera relative to the SLM.">

.. only:: html

  .. image:: /auto_examples/calibration/camera_mapping/images/thumb/sphx_glr_coarse_camera_mapping_thumb.png
    :alt:

  :doc:`/auto_examples/calibration/camera_mapping/coarse_camera_mapping`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Coarse camera mapping</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Generates a random array of spots on the camera by calculating an SLM phase pattern via superposition of linear phases, then detects the positions of the resulting focal spots  on the camera, fitting an affine transform between the camera coordinates and the coordinates of the simulated output plane. Needs a coarse mapping first to know where to place the spots on the camera.">

.. only:: html

  .. image:: /auto_examples/calibration/camera_mapping/images/thumb/sphx_glr_spot_array_camera_mapping_thumb.png
    :alt:

  :doc:`/auto_examples/calibration/camera_mapping/spot_array_camera_mapping`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Camera mapping from an array of spots</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


Wavefront calibration
---------------------

Measuring the laser intensity profile illuminating the SLM and its phase.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Fitting a parametrized model of the complex field illuminating the SLM by minimizing the difference between camera images of speckle intensity patterns and their simulated counterparts. The phase patterns displayed on the SLM are smooth to reduce effects of pixel crosstalk, which are not included in the model.">

.. only:: html

  .. image:: /auto_examples/calibration/wavefront_calibration/images/thumb/sphx_glr_wavefront_calibration_speckle_thumb.png
    :alt:

  :doc:`/auto_examples/calibration/wavefront_calibration/wavefront_calibration_speckle`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Wavefront calibration from speckle</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Subdivides the SLM into a grid of superpixels, and sequentially fills each of them with a linear phase, measureing the power of each diffracted spot on the camera to  reconstruct the intensity profile of the beam incident onto the SLM. Repeating this with a fixed reference superpixel and extracting the phases from the resulting interference  fringes recovers the phase.">

.. only:: html

  .. image:: /auto_examples/calibration/wavefront_calibration/images/thumb/sphx_glr_wavefront_calibration_raster_thumb.png
    :alt:

  :doc:`/auto_examples/calibration/wavefront_calibration/wavefront_calibration_raster`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Wavefront calibration by rastering</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


Pixel crosstalk calibration
---------------------------

Recovering the crosstalk between neighbouring liquid-crystal pixels.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Recovering the fringing field between neighboring liquid-crystal pixels on the SLM by  fitting a crosstalk model to the speckle that random SLM patterns produce.">

.. only:: html

  .. image:: /auto_examples/calibration/pixel_crosstalk_calibration/images/thumb/sphx_glr_pixel_crosstalk_calibration_speckle_thumb.png
    :alt:

  :doc:`/auto_examples/calibration/pixel_crosstalk_calibration/pixel_crosstalk_calibration_speckle`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Pixel crosstalk calibration from speckle</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /auto_examples/calibration/camera_mapping/index.rst
   /auto_examples/calibration/wavefront_calibration/index.rst
   /auto_examples/calibration/pixel_crosstalk_calibration/index.rst


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: calibration_python.zip </auto_examples/calibration/calibration_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: calibration_jupyter.zip </auto_examples/calibration/calibration_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
