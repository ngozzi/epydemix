import os
import sys

project = 'epydemix'
copyright = '2025, Nicolò Gozzi'
author = 'Nicolò Gozzi'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",       # Extracts documentation from docstrings
    "sphinx.ext.napoleon",      # Supports Google and NumPy-style docstrings
    "sphinx.ext.viewcode",      # Adds links to view source code
    "sphinx.ext.autosummary",   # Automatically generates summaries
]

# Enable autosummary
autosummary_generate = True  # Generate stub `.rst` files for API documentation

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

sys.path.insert(0, os.path.abspath('../../epydemix')) 
print("HERE")
print(sys.path)
