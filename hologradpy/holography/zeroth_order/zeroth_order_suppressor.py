"""Holograms changed to cancel the zeroth order the model predicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ...calibration.model_fit import field_on_frames
from ...optics.modules.zeroth_order import ZEROTH_ORDER_LAYER, ZerothOrderField
from ...optics.systems import SLMFourierLensModel
from ...optimization import LevenbergMarquardt
from ...serialization import SaveableRecord, record_type
from ...utils import ProgressBar, gpu_to_numpy

# The parameter the steps vary, the levels of the virtual SLM.
_LEVELS = "virtual_slm.levels"


@record_type("zeroth_order_suppression")
@dataclass
class ZerothOrderSuppressionData(SaveableRecord):
    """A hologram changed to cancel the zeroth order the model predicts.

    Attributes:
        timestamp: When the steps finished.
        phase: The SLM phase in radians.
        region: True on the pixels of the output plane where the zeroth order was
            cancelled.
        region_power: The power the model puts in the region, before the first step
            and after every step.
        step_costs: The cost before every Levenberg-Marquardt step, half the sum of
            squares of its residual.
        metadata: The settings of the steps.
    """

    timestamp: datetime
    phase: NDArray
    region: NDArray[np.bool_]
    region_power: list[float] = field(default_factory=list)
    step_costs: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ZerothOrderSuppressor:
    """Change the hologram on display to cancel the zeroth order the model predicts.

    The model carries a
    :class:`~hologradpy.optics.modules.zeroth_order.ZerothOrderField` after its virtual
    SLM. Levenberg-Marquardt steps on the levels
    (:class:`~hologradpy.optimization.LevenbergMarquardt`) change the hologram the
    model shows, such as one a phase retriever designed. The residual of a step is the
    model's field in the region and the change of the field over the rest of the output
    plane from the hologram the step starts from. The steps cancel the zeroth order and
    keep the rest of the light where it was. The light at the wraps is renewed before
    every step (:meth:`~.ZerothOrderField.update_wraps`), and the steps run without
    quantization, since a rounded level has no gradient.
    """

    def __init__(
        self,
        slm_camera_model: SLMFourierLensModel,
        zeroth_order_field: ZerothOrderField,
        region: Tensor,
    ) -> None:
        """
        Args:
            slm_camera_model: The model, showing the hologram to change.
            zeroth_order_field: The zeroth order the model predicts, such as
                :meth:`~.ZerothOrderField.from_calibration_data` gives it. It is
                inserted into the model after the virtual SLM, as
                :data:`~hologradpy.optics.modules.zeroth_order.ZEROTH_ORDER_LAYER`,
                unless the model carries it there already.
            region: True on the pixels of the output plane where the zeroth order is
                cancelled, ``(H, W)``. It holds none of the target's light, such as the
                calibration's disc less the target's spots.

        Raises:
            ValueError: The model carries another zeroth-order field.
        """
        self.slm_camera_model: SLMFourierLensModel = slm_camera_model
        self.zeroth_order_field: ZerothOrderField = zeroth_order_field
        layers = slm_camera_model.layers()
        if ZEROTH_ORDER_LAYER not in layers:
            if slm_camera_model.has(ZerothOrderField):
                raise ValueError(
                    "The model carries a ZerothOrderField under another name. Pass "
                    "that field, or a model without one."
                )
            slm_camera_model.insert_after(
                "virtual_slm", ZEROTH_ORDER_LAYER, zeroth_order_field
            )
        elif layers[ZEROTH_ORDER_LAYER] is not zeroth_order_field:
            raise ValueError(
                f"The model carries another ZerothOrderField as {ZEROTH_ORDER_LAYER}. "
                "Pass that field, or a model without one."
            )
        self.region: Tensor = torch.as_tensor(region, dtype=torch.bool).to(
            slm_camera_model.device
        )

    def _field(self) -> Tensor:
        """The model's field on the output plane, ``(H, W)``, with the light at the
        wraps of the hologram on display.
        """
        model = self.slm_camera_model
        self.zeroth_order_field.update_wraps(model)
        with torch.no_grad():
            return field_on_frames(model())[0]

    def suppress_zeroth_order(
        self, steps: int = 8, verbose: bool = True
    ) -> ZerothOrderSuppressionData:
        """Take the steps, and leave the changed hologram on the model.

        Args:
            steps: The Levenberg-Marquardt steps. On a simulated SLM, eight steps from
                a CG hologram of a 2 x 2 spot array left 0.014 % of its zeroth order,
                at 0.5 % of the spots' power.
            verbose: Show the progress of the steps.

        Returns:
            ZerothOrderSuppressionData: The hologram and the power left in the region.
        """
        model = self.slm_camera_model
        virtual_slm = model.virtual_slm
        levels = dict(model.named_parameters())[_LEVELS]
        region = self.region
        rest = ~region
        held_quantize = virtual_slm.quantize
        held_requirement = levels.requires_grad
        virtual_slm.quantize = False
        levels.requires_grad_(True)
        start = self._field()
        region_power = [float(start[region].abs().square().sum())]
        costs: list[float] = []
        try:
            optimizer = LevenbergMarquardt([levels])
            with ProgressBar(
                total=steps, description="Zeroth-order steps", verbose=verbose
            ) as progress:
                for _ in range(steps):
                    # A new closure for every step, so the optimizer evaluates the
                    # residual from the hologram the step starts from, with its light
                    # at the wraps.
                    def closure(
                        chunk: int,
                        values: list[Tensor] | None,
                        start: Tensor = start,
                    ) -> Tensor:
                        trial = levels if values is None else values[0]
                        output = torch.func.functional_call(model, {_LEVELS: trial}, ())
                        field_out = field_on_frames(output)[0]
                        residual = torch.cat(
                            [field_out[region], (field_out - start)[rest]]
                        )
                        return torch.view_as_real(residual).flatten()

                    costs.append(optimizer.step(closure))
                    start = self._field()
                    region_power.append(float(start[region].abs().square().sum()))
                    progress.update(cost=costs[-1])
        finally:
            virtual_slm.quantize = held_quantize
            levels.requires_grad_(held_requirement)
        return ZerothOrderSuppressionData(
            timestamp=datetime.now(),
            phase=gpu_to_numpy(virtual_slm.get_phase()),
            region=gpu_to_numpy(region),
            region_power=region_power,
            step_costs=costs,
            metadata={"steps": steps},
        )
