Saved records
=============

Records are saved as `ASDF <https://www.asdf-format.org/>`_ files. Each record type
has a name and a version, which
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
- ``camera_mapping`` version 2: the zeroth order is derived from the fitted
  similarity, as ``(column, row)``. A ``zeroth_order_position`` stored at version 1 or
  2 is dropped as the file loads.
- ``target_placement`` version 2: the zeroth order, the target centre and the
  addressable corners are ``(column, row)``.
- ``vortex_annihilation`` version 2: the vortex positions are ``(column, row)``.

The ``camera_angle``, ``camera_shift`` and ``camera_zoom`` of a ``camera_mapping`` are
derived from its stored fields whenever it is read, so every saved mapping that records
its camera and its output plane gives them.

The loader keeps only the keys that a record type declares, so the stored zeroth order
of a ``camera_mapping`` is dropped without a ``_migrate``.

Records saved on the ``feature/robust_camera_feedback`` branch hold some metadata in
the conventions of that branch. The ``frame_translations`` in the metadata of a
:class:`~hologradpy.calibration.wavefront.abstract.WavefrontCalibrationData` have y
pointing down. The ``pattern_offsets`` and ``translations`` in the metadata of a
:class:`~hologradpy.holography.camera_feedback.abstract.CameraFeedbackData` are
``(rows, columns)``. These keys are not converted as the file loads, since HoloGradPy
reads none of them back.

A model checkpoint (``.pt``) is not a record, so it is not converted.
