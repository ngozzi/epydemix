import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../..'))  

project = 'epydemix'
copyright = '2025, Nicolò Gozzi'
author = 'Nicolò Gozzi'
release = '1.0.0'

extensions = [
    "sphinx.ext.autodoc",      
    "sphinx.ext.napoleon",      
    "sphinx.ext.viewcode",      
    "sphinx.ext.autosummary",  
]

autosummary_generate = True  
templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
