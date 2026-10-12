.. _Conventions:

Conventions
===========

HoloGradPy describes a plane in two frames. Positions, displacements and angles in its
interfaces are physical. Arrays are stored and indexed in the pixel frame. Only
:mod:`hologradpy.grids` converts between the two.

The physical frame
------------------

- Coordinates are ``(x, y)`` in metres.
- x points right, and y points up on a screen that draws row 0 at the top.
- The origin of a plane of ``height`` by ``width`` pixels is the pixel
  ``(width // 2, height // 2)``.
- An angle turns +x towards +y. A positive angle therefore turns counterclockwise on
  the screen.

Column ``i`` sits at ``x = (i - width // 2) * pitch_x``, and row ``i`` sits at
``y = (height // 2 - i) * pitch_y``. Row 0 holds the largest y.

The grid of an axis of even length is asymmetric. x runs from ``-n/2`` to ``n/2 - 1``
pitches, and y runs from ``n/2`` down to ``-(n/2 - 1)``.

The pixel frame
---------------

- Arrays are indexed ``image[row, column]``.
- Sizes are ``(height, width)``, and pitches are ``(y, x)``.
- Pixel positions and pixel-frame displacements are ``(column, row)``, in the order of
  ``(x, y)``.

The row index grows down the screen, opposite to y. So a pixel position is named
``(column, row)``, never ``(x, y)``.

Some examples:

- :class:`~hologradpy.roi.ROI` holds ``(top_row, left_column, height, width)``.
  :meth:`~hologradpy.roi.ROI.centered` takes its centre as ``(column, row)`` and its
  size as ``(height, width)``.
- :class:`~hologradpy.calibration.camera_mapping.mapping.CameraMapping` keeps
  ``zeroth_order_position`` as ``(column, row)`` on the sensor.
- :func:`~hologradpy.calibration.spot_detection.detect_spot` returns the
  ``(column, row)`` of a spot.
- :func:`~hologradpy.profiles.masks.disc_mask` takes its centre as ``(column, row)``.
- :func:`~hologradpy.fourier_transforms.translate.fft_translate` takes its shift as
  ``(columns, rows)``.
- :class:`~hologradpy.calibration.wavefront.raster_calibration.raster_calibrator.RasterCalibrator`
  takes ``number_of_superpixels`` as ``(rows, columns)`` and ``superpixel_size`` as
  ``(height, width)``.

An index is not a position. :func:`~hologradpy.grids.coordinates_to_indices` returns
``(row, column)`` pairs, as ``image[row, column]`` takes them. A camera's
``excluded_pixels`` are ``(row, column)`` pairs too.

Converting between the frames
-----------------------------

The helpers of :mod:`hologradpy.grids` hold the only conversion.

- :func:`~hologradpy.grids.get_spatial_grid` gives the ``(x, y)`` of every pixel in
  metres, and :func:`~hologradpy.grids.get_pixel_grid` gives it in pixels.
- :func:`~hologradpy.grids.plane_center` gives the pixel of the origin, as
  ``(column, row)``.
- :func:`~hologradpy.grids.metres_to_pixel` and
  :func:`~hologradpy.grids.pixel_to_metres` convert positions.
- :func:`~hologradpy.grids.pixel_frame_vector` converts displacements, signed pitches
  and shifts in pixel units. It negates the second component.
- :func:`~hologradpy.grids.pixel_frame_angle` converts angles. It negates them.

Both frame helpers are their own inverse. A conversion negates about the centre row
and never reverses an axis, so the FFT, the chirp-z transform and the NUFFT stay
centred.

Lenses and camera mappings
--------------------------

The parameters of the lenses and of the camera mappings are physical.

- ``camera_angle`` turns from +x towards +y, and ``camera_shift`` is ``(x, y)``.
- The angle and the shift of
  :class:`~hologradpy.optics.modules.learnable_partial_affine.LearnablePartialAffine`
  and of :class:`~hologradpy.optics.modules.geometric_transforms.GeometricWarp` are
  physical. The shift is in output pixels along x and y.
- ``CameraMapping.rotation_degrees`` turns from +x towards +y.

``CameraMapping.transform`` maps a camera pixel to a model pixel, both as
``(column, row)``. The model is the one without its focal-plane partial affine.
``CameraMapping.image_plane_to_sensor`` takes ``(x, y)`` metres and returns sensor
``(column, row)`` pixels. ``CameraMapping.sensor_to_image_plane`` does the reverse.
``CameraMapping.sensor_crop_to_model`` resamples a crop of the sensor onto model
pixels, about the crop's centre pixel.

Camera orientation
------------------

:class:`~hologradpy.hardware.camera.abstract.CameraOrientation` says how a sensor is
mounted, with the flags a device takes: ``rot``, ``fliplr`` and ``flipud``. The
rotation comes before the flips, as in slmsuite.

A rotation of ``"90"`` is one :func:`numpy.rot90`. It turns the displayed image a
quarter turn counterclockwise. These turns and flips act on the pixel array, so they
belong to the pixel frame.

A camera turned counterclockwise by an angle against the model plane reports minus
that angle as ``CameraMapping.rotation_degrees``. The mapping runs from the camera back
to the model.

Saved records
-------------

Records are saved as ASDF files. Each record type has a name and a version, which
:func:`~hologradpy.serialization.record_type` sets.

A change that old files need help with raises the version. The class then gains a
``_migrate`` class method, which converts a file saved at an older version as it
loads. ASDF converts each tagged node on its own, so a record nested in another record
is converted too.

The conversions so far:

- ``wavefront_calibration`` version 2: y points up. The y of the beam centre and the
  y of the raster tilts change sign.
- ``wavefront_calibration`` version 3: the fitted beam moves from the metadata into
  ``beam``.
- ``raster_visualization`` version 2: the lattice drift along y changes sign.
- ``camera_mapping`` version 2: ``zeroth_order_position`` is ``(column, row)``.
- ``target_placement`` version 2: the zeroth order, the target centre and the
  addressable corners are ``(column, row)``.
- ``vortex_annihilation`` version 2: the vortex positions are ``(column, row)``.

A model checkpoint (``.pt``) is not a record, so it is not converted.
