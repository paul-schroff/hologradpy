"""Setup shared by the whole suite.

Anything every test file would otherwise repeat belongs here. The backend in particular
has to be chosen before any module imports ``matplotlib.pyplot``, which is why it used
to sit above the imports of a dozen files and force a ``# noqa: E402`` on every import
line beneath it.
"""

from __future__ import annotations

import matplotlib

# Headless, once for the suite. A test that opens a window blocks a CI run forever, and
# a dozen of these draw figures.
matplotlib.use("Agg")

# The float64 fixture stays in test_optics_adjoint.py and test_optics_gradcheck.py
# rather than moving here. It is autouse in both, and autouse in conftest would put the
# whole suite in float64: slower everywhere, and a silent change to what the other
# tests measure.
