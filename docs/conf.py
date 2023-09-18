# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_rtd_theme
import sys
import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HoloGradPy'
copyright = '2023, Paul Schroff, Department of Physics, University of Strathclyde'
author = 'Paul Schroff'
release = '1.0'

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('../hologradpy/'))
paths = ['../..', '../../hologradpy', ]
for path in paths:
    sys.path.insert(0, os.path.abspath(path))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx_gallery.gen_gallery', 'sphinx_codeautolink']

# ToDo: Complete list of mock imports
autodoc_mock_imports = ['mpl_toolkits.axes_grid1' 'scipy', 'matplotlib', 'cv2', 'time', 'torchmin', 'checkerboard']
autodoc_member_order = 'bysource'
autoclass_content = 'both'

sphinx_gallery_conf = {
    'examples_dirs': '../examples',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {

}

