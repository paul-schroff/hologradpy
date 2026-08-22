# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

from sphinx_gallery.sorting import ExplicitOrder

# ---- Project information -------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "HoloGradPy"
copyright = "2026, Paul Schroff, Department of Physics, University of Strathclyde"
author = "Paul Schroff"
release = "1.0"


paths = [".."]
for path in paths:
    sys.path.insert(0, os.path.abspath(path))

# ---- General configuration -----------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Skip regenerating the example gallery, for faster iteration on everything else.
# The rst that sphinx-gallery wrote on the last normal build persists in
# auto_examples/, so those pages still build as ordinary sources and every link into
# the gallery keeps resolving. Needs one normal build first to exist.
SKIP_EXAMPLES = os.environ.get("HOLOGRADPY_SKIP_EXAMPLES") == "1"
if SKIP_EXAMPLES and not os.path.exists(
    os.path.join(os.path.dirname(__file__), "auto_examples", "index.rst")
):
    raise RuntimeError(
        "HOLOGRADPY_SKIP_EXAMPLES=1 needs a previously generated gallery. Run one "
        "build without it first."
    )

extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_design",
]
if not SKIP_EXAMPLES:
    extensions.append("sphinx_gallery.gen_gallery")

# The README is included into index.rst below an RST title, so its headings start at
# H2 by design. MyST flags that on any fragment it parses, which is a false positive
# here.
suppress_warnings = ["myst.header"]

# Single backticks mean code, so `like_this` in a docstring is not italicised.
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
# Examples run only when asked. Several take minutes and want a GPU, so a Read the
# Docs build leaves them unexecuted and shows the source alone. Set
# HOLOGRADPY_RUN_EXAMPLES=1 to run them and capture their figures.
RUN_EXAMPLES = os.environ.get("HOLOGRADPY_RUN_EXAMPLES") == "1"

sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "reference_url": {"hologradpy": None},
    "filename_pattern": r".*" if RUN_EXAMPLES else "(?!.*)",
    "ignore_pattern": r"(__init__|.*[\\/]dev_scripts[\\/].*)\.py",
    "plot_gallery": RUN_EXAMPLES,
    "only_warn_on_example_error": True,
    "subsection_order": ExplicitOrder(
        [
            "../examples/hardware_interface",
            "../examples/phase_retrieval",
            "../examples/camera_mapping",
            "../examples/wavefront_calibration",
            "../examples/pixel_crosstalk_calibration",
            "../examples/camera_feedback",
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
