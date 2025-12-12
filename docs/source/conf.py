# Configuration file for the Sphinx documentation builder.

# Ensure Sphinx can find the 'clawdia' package
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'CLAWDIA'
copyright = '2025, Miquel Lluís Llorens Monteagudo'
author = 'Miquel Lluís Llorens Monteagudo'

from clawdia import __version__ as version
release = version


# -- General configuration ---------------------------------------------------

extensions = [
    'numpydoc',            # Parses NumPy-style docstrings
    'sphinx.ext.autodoc',  # Auto-generates documentation from docstrings
    'sphinx.ext.viewcode', # Adds links to highlighted source code
    'sphinx.ext.mathjax',  # Renders math equations
    'sphinx.ext.autosummary',  # Generate summary tables
    'sphinx.ext.intersphinx',  # Allows references to external libraries
]

autosummary_generate = True
autodoc_default_options = {
    "special-members": "__init__",
    "inherited-members": True,
    "show-inheritance": True,
}

# Add explicit anchors to methods and class members
numpydoc_class_members_toctree = False  # Ensures class methods are linked in the "On this page" section

# To Enable Cross-Referencing with Sphinx (Intersphinx)
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}


# -- Options for HTML output -------------------------------------------------

html_static_path = ['_static']
templates_path = ['_templates']
html_theme = 'pydata_sphinx_theme'
html_css_files = [
    'custom.css',
]

html_logo = "_static/clawdia-logo.svg"

html_theme_options = {
    # HEADER ------------------------------------------------------------------
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/MiquelLluis/clawdia",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
   ],


   # SECONDARY SIDEBAR (RIGHT) ------------------------------------------------

   "secondary_sidebar_items": ["page-toc"],
    "show_nav_level": 0,
    "show_toc_level": 2,
    "navigation_depth": 3,
}

