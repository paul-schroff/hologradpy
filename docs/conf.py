# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import pathlib
import re
import sys
import warnings

from sphinx_gallery.sorting import ExplicitOrder

# ---- Project information -------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "HoloGradPy"
copyright = "2026, Paul Schroff, Department of Physics, University of Strathclyde"
author = "Paul Schroff"
release = "0.2.0.dev0"


paths = [".."]
for path in paths:
    sys.path.insert(0, os.path.abspath(path))

# ---- General configuration -----------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

# The README is included into index.rst below an RST title, so its headings start at
# H2 by design. MyST flags that on any fragment it parses, which is a false positive
# here.
suppress_warnings = ["myst.header"]

# Configuring relative paths for warning messages
_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _relative_warning(message, category, filename, lineno, line=None):
    path = pathlib.Path(filename)
    try:
        shown = path.resolve().relative_to(_REPOSITORY_ROOT).as_posix()
    except ValueError:
        shown = path.name
    return f"{shown}:{lineno}: {category.__name__}: {message}\n"


warnings.formatwarning = _relative_warning

default_role = "literal"

# ---- Docstrings ----------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_rtype = True

# ---- Type hints ----------------------------------------------------------------------
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_type_aliases = {}

# ---- AutoAPI -------------------------------------------------------------------------
autoapi_dirs = ["../hologradpy"]

# The default drops the __init__ docstring, which is where nearly every constructor
# in this package documents its arguments.
autoapi_python_class_content = "both"

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_member_order = "groupwise"

# Land on the package page, with its summary tables, in place of the generated
# "API Reference" index: a flat wall of every dotted module path.
autoapi_add_toctree_entry = False

# Keep the generated rst after building
autoapi_keep_files = True

# ---- Cross-references ----------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "slmsuite": ("https://slmsuite.readthedocs.io/en/latest/", None),
}

# ---- Example gallery -----------------------------------------------------------------
# Examples run only when asked. Several take minutes and need a GPU, so a Read the
# Docs build leaves them unexecuted and shows the source alone. Set
# HOLOGRADPY_RUN_EXAMPLES=1 to run them and capture their figures.
RUN_EXAMPLES = os.environ.get("HOLOGRADPY_RUN_EXAMPLES") == "1"

# Re-run only selected examples for faster iteration.
#
#   HOLOGRADPY_RUN_EXAMPLES=1 HOLOGRADPY_EXAMPLES=hardware_interface
#   HOLOGRADPY_RUN_EXAMPLES=1 HOLOGRADPY_EXAMPLES='camera_mapping|top_hat'
EXAMPLE_FILTER = os.environ.get("HOLOGRADPY_EXAMPLES", "")

if RUN_EXAMPLES and EXAMPLE_FILTER:
    # Editing library code leaves every example's md5 untouched.
    _gallery_root = pathlib.Path(__file__).parent / "auto_examples"
    for _stamp in _gallery_root.glob("**/*.py.md5"):
        if re.search(EXAMPLE_FILTER, _stamp.as_posix()):
            _stamp.unlink()

sphinx_gallery_conf = {
    "examples_dirs": [
        "../examples/hardware_interface",
        "../examples/phase_retrieval",
        "../examples/camera_feedback",
        "../examples/calibration",
    ],
    "gallery_dirs": [
        "auto_examples/hardware_interface",
        "auto_examples/phase_retrieval",
        "auto_examples/camera_feedback",
        "auto_examples/calibration",
    ],
    "reference_url": {"hologradpy": None},
    "filename_pattern": (EXAMPLE_FILTER or r".*") if RUN_EXAMPLES else "(?!.*)",
    "ignore_pattern": r"(__init__|.*[\\/]dev_scripts[\\/].*)\.py",
    "plot_gallery": RUN_EXAMPLES,
    "only_warn_on_example_error": True,
    "image_srcset": ["2x"],
    "subsection_order": ExplicitOrder(
        [
            "../examples/calibration/camera_mapping",
            "../examples/calibration/wavefront_calibration",
            "../examples/calibration/pixel_crosstalk_calibration",
        ]
    ),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---- LaTeX ---------------------------------------------------------------------------
latex_engine = "xelatex"

latex_elements = {
    "extraclassoptions": "openany,oneside",
}

latex_table_style = ["booktabs"]

latex_use_xindy = False

# ---- Options for HTML output ---------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": False,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
