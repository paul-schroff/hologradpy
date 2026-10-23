Cameras and lenses
==================

.. _Lenses and camera mappings:

Lenses and camera mappings
--------------------------

The focal plane holds positions ``(x, y)`` in metres from the zeroth order, as the
Fourier lens forms them before the focal-plane partial affine of a model. The tilt of
:meth:`~hologradpy.optics.systems.abstract.SLMFourierLensModel.steering_phase`, the
``target_position`` of the camera feedback and the ``initial_tilt`` of the coarse
mapper are focal-plane positions. The camera mappings call the same plane the image
plane, in ``image_plane_to_sensor``, ``sensor_to_image_plane`` and
``sensor_pixels_per_metre``. The output plane is the grid a model outputs, the camera
grid of the camera-aware optical systems. The camera plane holds positions ``(x, y)``
in metres from the sensor's centre pixel, :func:`~hologradpy.grids.plane_center` of the
whole sensor.

The camera-aware optical systems take the camera as two numbers, and a camera mapping
reports the same two, with the same meaning and the same sign, together with the zoom.

- ``camera_angle`` is the turn of the focal plane as the camera shows it, in degrees
  from +x towards +y. A positive angle turns the displayed image counterclockwise.
- ``camera_shift`` is where the zeroth order lands on the camera, as ``(x, y)`` metres
  from the centre pixel of the output plane. For a mapping, the output plane is the
  whole sensor.
- ``camera_zoom`` is the zoom of the focal plane as the camera shows it, in
  camera-plane metres per focal-plane metre. The systems hold it as the
  ``scale_factor`` of their partial affine, and take no seed for it.

A camera simulated with ``camera_angle=15`` and ``camera_shift=(600e-6, -300e-6)`` maps
back to a ``camera_angle`` of 15 and a ``camera_shift`` of ``(600e-6, -300e-6)``,
against an :class:`~hologradpy.optics.systems.slm_czt.SLMCZT` on the camera's grid or
against an :class:`~hologradpy.optics.systems.slm_fft.SLMFFT`. The three numbers come
from the similarity that
:meth:`~hologradpy.optics.systems.abstract.SLMFourierLensModel.calibrate_from_mapping`
inverts into a model, so a model on the whole sensor calibrated from the mapping holds
them. A similarity holds no mirror, and it describes the camera in metres on square
pixels only, so a mirrored mapping, or one measured on pixels that are not square,
gives none of the three.

For a camera with an anisotropic zoom, RANSAC can fit the similarity to a subset of
the point pairs, as few as two of the four pairs of a coarse mapping. The zoom can then
be the zoom along one axis, and the shift moves by the anisotropy times the distance
from those pairs to the zeroth order. A model calibrated from the mapping holds the
same numbers.

Each transform between the focal plane and the camera runs in one direction. The table
gives it, with the frame, the units and the origin.

.. list-table::
   :header-rows: 1
   :widths: 30 18 30 22

   * - Name
     - Runs
     - Frame and units
     - Origin
   * - ``camera_angle`` of SLMCZT, SLMNUFFT and SLMFFTAffine
     - focal plane -> camera
     - physical, degrees from +x towards +y
     - a turn about the centre pixel of the output plane
   * - ``camera_shift`` of the same systems
     - focal plane -> camera
     - physical ``(x, y)``, metres at ``camera_pixel_size``
     - the centre pixel of the output plane
   * - ``angle``, ``scale_factor`` and ``shift`` of ``focal_plane_partial_affine``
     - focal plane -> output plane
     - physical, degrees, a zoom along ``(x, y)``, and output pixels along ``(x, y)``
     - the centre pixel of the output plane
   * - ``CameraMapping.camera_angle``, ``camera_shift`` and ``camera_zoom``
     - focal plane -> camera
     - physical, degrees, ``(x, y)`` metres at the sensor's pitch, and camera-plane
       metres per focal-plane metre
     - the sensor's centre pixel
   * - ``CameraMapping.transform``, ``affine`` (the full fit) and ``partial_affine``
       (the similarity)
     - camera -> model
     - pixel frame, ``(column, row)`` camera pixels to ``(column, row)`` model pixels.
       ``rotation_degrees`` turns +column towards +row
     - pixel ``(0, 0)`` of each plane
   * - ``CameraMapping.zeroth_order_position``
     - focal plane -> camera
     - pixel frame, ``(column, row)`` camera pixels
     - pixel ``(0, 0)``
   * - ``CameraMapping.image_plane_to_sensor`` and ``sensor_pixels_per_metre``
     - focal plane -> camera
     - ``(x, y)`` metres to ``(column, row)`` camera pixels
     - the zeroth order, and pixel ``(0, 0)``
   * - ``CameraMapping.sensor_to_image_plane``, ``tilt_to_sensor_center`` and
       ``sensor_crop_to_model``
     - camera -> focal plane
     - ``(column, row)`` camera pixels to ``(x, y)`` metres, and the crop as a pixel
       array
     - pixel ``(0, 0)``, and the zeroth order
   * - ``SLMFourierLensModel.focal_plane_positions``
     - output plane -> focal plane, through the model's current partial affine
     - physical ``(x, y)``, metres
     - the centre pixel of the output plane, and the zeroth order
   * - ``steering_phase(tilt)``, ``target_position``, ``initial_tilt`` and
       ``linear_phase_tilt``
     - in the focal plane
     - physical ``(x, y)``, metres
     - the zeroth order
   * - ``get_diffraction_spot_position``
     - on the camera
     - physical ``(x, y)`` camera-plane metres, or ``(column, row)`` camera pixels
       truncated to integers
     - the sensor's centre pixel, or pixel ``(0, 0)``
   * - ``FocalSpotFit.waist``
     - on the camera
     - camera-plane metres
     - none
   * - ``OrientationSuggestion.residual_transform`` and
       ``MappingFit.reprojection_errors``
     - camera -> model
     - pixel frame, model pixels
     - pixel ``(0, 0)``
   * - ``CameraOrientation`` (``rot``, ``fliplr``, ``flipud``) and
       ``OrientationSuggestion.suggested``
     - raw frame -> displayed frame
     - the pixel array
     - ``rot="90"`` is one :func:`numpy.rot90`, before the flips

For a mapping without a mirror, ``partial_affine.rotation_degrees`` holds the same
number as ``camera_angle``, since the inverse and the change of frame each negate the
angle. ``affine.rotation_degrees`` holds about the same number, from the full fit.

A partial affine turns the focal plane by its angle about the centre pixel of the
output plane, then zooms it by ``scale_factor`` along the output x and y, and then
moves it by ``shift``. The window of the focal plane that a lens samples therefore
moves the opposite way. The chirp-z and NUFFT lenses, the field warp and the target
warp of optimal transport follow this order.

- The lenses
  (:class:`~hologradpy.optics.modules.propagators.partial_affine_fourier_lens.PartialAffineFourierLens`)
  turn the window of the focal plane in radians per sample of the SLM grid. Their turn
  acts in metres only for square SLM pixels.
- The field warp, ``GeometricWarp``, turns positions in metres. The input pitch takes
  the input pixels to metres, and the output pitch takes the metres to output pixels.
- :class:`~hologradpy.holography.phase_retrieval.optimal_transport.OptimalTransportPhaseRetriever`
  resamples its target through the inverse of the turn and the zoom, with the turn in
  metres. The shift moves every point of the image alike, so it enters the transport
  map, as the image-plane position of the output grid's centre.

``CameraMapping.image_plane_to_sensor``, ``sensor_to_image_plane``,
``sensor_crop_to_model`` and ``zeroth_order_position`` run through
``CameraMapping.partial_affine``, the similarity that a model is calibrated from, so
the mapping places a point where the calibrated model puts it. A similarity holds no
mirror, so a mirrored mapping places points through its full fit,
``CameraMapping.affine``, and calibrates no model. The zeroth-order masks follow
``zeroth_order_position``.

:class:`~hologradpy.optics.modules.slm_fields.psf.PSFSLMField` maps its image-plane
kernel to the SLM grid through an inverse chirp-z transform, a chirp-z with a negative
magnification. The Fourier lens then images the kernel upright, in the model's
orientation.
:class:`~hologradpy.calibration.wavefront.speckle_calibration.speckle_calibrator.psf_calibrator.PSFSpeckleCalibrator`
turns a captured focal spot into the model's orientation with
``CameraMapping.sensor_crop_to_model``, and the turned spot seeds the kernel.

Camera orientation
------------------

:class:`~hologradpy.hardware.camera.abstract.CameraOrientation` says how a sensor is
mounted, with the flags a device takes, ``rot``, ``fliplr`` and ``flipud``. The flags
carry the raw frame to the displayed frame, and the rotation comes before the flips,
as in slmsuite.

A rotation of ``"90"`` is one :func:`numpy.rot90`. It turns the displayed image a
quarter turn counterclockwise, as a ``camera_angle`` of 90 degrees does. A camera that
shows the focal plane unturned in its raw frame therefore reads a ``camera_angle`` of
90 once it is mounted with ``rot="90"``. These turns and flips act on the pixel array,
so they belong to the pixel frame. They reverse the array about its middle,
``((width - 1) / 2, (height - 1) / 2)``, and the centre pixel,
:func:`~hologradpy.grids.plane_center`, lies half a pixel past the middle along an
axis of even length. The pixel at the centre of the raw frame therefore lands one pixel
from the centre pixel of the displayed frame along each reversed axis of even length.

:meth:`~hologradpy.calibration.camera_mapping.coarse_mapping.coarse_mapper.CoarseMapper.map_camera`
with ``find_camera_orientation=True`` suggests a mounting.
``OrientationSuggestion.suggested`` is the whole mounting to pass to
:meth:`~hologradpy.hardware.camera.abstract.Camera.set_orientation`. Mounted so, the
camera shows the focal plane unmirrored and turned by at most 45 degrees.

The coarse mapper and the checkerboard mapper start their search at the middle of the
pixel array, ``((width - 1) / 2, (height - 1) / 2)``. Camera-plane positions in metres
are measured from the centre pixel, :func:`~hologradpy.grids.plane_center`,
focal-plane positions from the zeroth order, and pixel positions ``(column, row)`` from
pixel ``(0, 0)``.
