SLM phase and the beam
======================

The SLM phase
-------------

A phase displayed on the SLM multiplies the field by ``exp(1j * phase)``, both in
:class:`~hologradpy.optics.modules.virtual_slms.abstract.VirtualSLM` and through
:meth:`SLM.set_phase <hologradpy.hardware.slm.abstract.SLM.set_phase>`. A positive
phase is a phase delay, as in slmsuite, since a plane wave propagated downstream by
``z`` gains ``exp(1j * k_z * z)``
(:class:`~hologradpy.optics.modules.propagators.angular_spectrum_method.AngularSpectrumMethod`)
and a converging lens has the phase ``-k * r**2 / (2 * f)``
(:func:`~hologradpy.profiles.phase.lens_phase`).

- A phase rising along +x steers the light towards +x in the focal plane, and a phase
  rising along +y steers it towards +y. :func:`~hologradpy.profiles.phase.linear_phase`
  and :meth:`~hologradpy.optics.systems.abstract.SLMFourierLensModel.steering_phase`
  give this ramp for a displacement ``(x, y)`` of the spot.
- The gray level falls as the phase rises, since most liquid-crystal SLMs shorten the
  phase delay as the voltage rises. Zero phase is level 0, and a small positive phase
  is one of the highest levels.
- slmsuite negates the phase in the same way
  (:meth:`slmsuite.hardware.slms.slm.SLM.set_phase`). It shows each phase one level
  lower, so its zero phase is the highest level. The two differ by a constant phase.
- A :class:`~hologradpy.phase_levels.PhaseResponse` holds this mapping for a device,
  and ``full_scale_cycles`` is the phase range from level 0 to full scale, in cycles.
  :meth:`~hologradpy.phase_levels.LookupResponse.from_phase_delays` takes a curve
  rising with level, as a vendor table or a measurement gives it, and stores its
  negative.

The slmsuite adapter,
:class:`~hologradpy.hardware.slmsuite.adapter.SLMSuiteSLMAdapter`, converts the phase to
levels in HoloGradPy and passes whole levels to slmsuite. slmsuite displays integer data
as given, without negating it.

Along the beam
--------------

A positive ``defocus`` in :func:`~hologradpy.profiles.phase.defocus_phase` moves the
focus away from the SLM, along the beam, so the camera's plane lies ``defocus`` before
the focus.
:class:`~hologradpy.optics.modules.propagators.angular_spectrum_method.AngularSpectrumMethod`
propagates a field downstream for a positive ``propagation_distance``. With a defocus
``d`` on the SLM, the image plane therefore holds the in-focus field propagated by
``-d``.

The ``defocus_distances`` that
:class:`~hologradpy.holography.camera_feedback.model_learning_feedback.ModelLearningFeedbackCorrector`
takes in
:meth:`~hologradpy.holography.camera_feedback.model_learning_feedback.ModelLearningFeedbackCorrector.run`
have the sign of ``defocus``.
