# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CLAWDIA'
copyright = '2025, Miquel Lluís Llorens Monteagudo'
author = 'Miquel Lluís Llorens Monteagudo'
release = '0.4.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'numpydoc',            # Parses NumPy-style docstrings
    'sphinx.ext.autodoc',  # Auto-generates documentation from docstrings
    'sphinx.ext.viewcode', # Adds links to highlighted source code
    'sphinx.ext.mathjax',  # Renders math equations
]

# Optional: Numpydoc settings (tweak as needed)
# numpydoc_show_class_members = False  # Avoid showing all class members automatically
# numpydoc_class_members_toctree = False  # Avoid creating a toctree for class members

templates_path = ['_templates']
exclude_patterns = []

# Ensure Sphinx can find the 'clawdia' package
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# # Optional: Customize the theme with some common settings
# html_theme_options = {
#     "logo": {
#         "text": "CLAWDIA",  # Displayed text with the logo
#     },
#     "navbar_start": ["navbar-logo", "navbar-title"],  # Controls the navigation bar
#     "navbar_end": ["search-field", "navbar-icon-links"],  # Controls the right-hand section of the navbar
#     "icon_links": [
#         {
#             "name": "GitHub",
#             "url": "https://github.com/yourusername/clawdia",
#             "icon": "fab fa-github",  # Requires FontAwesome, which this theme supports
#         },
#     ],
# }
