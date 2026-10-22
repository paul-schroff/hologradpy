Coordinate frames
=================

HoloGradPy describes a plane in two frames. Positions, displacements and angles in its
interfaces are physical. Arrays are stored and indexed in the pixel frame. Only
:mod:`hologradpy.grids` converts between the two.

.. plot:: figures/coordinate_frames.py
   :alt: The pixel frame and the physical frame on a plane of 4 by 6 pixels

   A plane of 4 by 6 pixels in the two frames. The pixel frame counts columns to the
   right and rows down from the top-left pixel. The physical frame has its origin at
   the pixel ``(width // 2, height // 2)``, here ``(3, 2)``, with x to the right and y
   up. The highlighted pixel, ``(column, row) = (1, 3)``, lies at ``(x, y) = (-2, -1)``
   pitches. The positions are those of :func:`~hologradpy.grids.get_spatial_grid`.

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
- :class:`~hologradpy.calibration.camera_mapping.mapping.CameraMapping` derives
  ``zeroth_order_position`` from the similarity fitted to its point pairs, as
  ``(column, row)`` on the sensor. The rows of its ``sensor_pixels_per_metre`` are the
  sensor ``(column, row)``, and its columns are the change of each along x and y.
- :func:`~hologradpy.calibration.spot_detection.detect_spot` returns the
  ``(column, row)`` of a spot, and
  :func:`~hologradpy.calibration.spot_detection.brightest_pixel` that of the brightest
  pixel.
- :func:`~hologradpy.profiles.masks.disc_mask` takes its centre as ``(column, row)``.
- :func:`~hologradpy.fourier_transforms.translate.fft_translate`,
  :func:`~hologradpy.fourier_transforms.translate.translate_real` and
  :func:`~hologradpy.fourier_transforms.translate.translate_intensity` take their
  shift as ``(columns, rows)``.
  :func:`~hologradpy.fourier_transforms.translate.cross_correlation_peak` returns its
  shift in the same order. :func:`~hologradpy.analysis.fitting.fit_translation`
  returns a :class:`~hologradpy.analysis.fitting.TranslationFit` with its ``shift`` as
  ``(columns, rows)``.
- The ``pattern_offsets`` and ``translations`` that the camera feedback correctors
  keep in the metadata of their records are ``(columns, rows)``.
- :class:`~hologradpy.calibration.wavefront.raster_calibration.raster_calibrator.RasterCalibrator`
  takes ``number_of_superpixels`` as ``(rows, columns)`` and ``superpixel_size`` as
  ``(height, width)``. The helpers of
  :class:`~hologradpy.calibration.wavefront.raster_calibration.superpixel_slicer.SuperpixelSlicer`
  take and return the same orders, and ``SuperpixelSlicer.center`` gives the
  ``(column, row)`` of a superpixel.

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
- :func:`~hologradpy.grids.metres_to_pixels` converts a shift in metres to a shift in
  pixels along x and y. It changes the units within the physical frame and negates
  nothing.
- :func:`~hologradpy.grids.pixel_frame_vector` converts displacements, signed pitches
  and shifts in pixel units. It negates the second component.
- :func:`~hologradpy.grids.pixel_frame_angle` converts angles. It negates them.

Both frame helpers are their own inverse. A conversion negates about the centre row
and never reverses an axis, so the FFT, the chirp-z transform and the NUFFT stay
centred.
