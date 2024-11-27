# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
from pathlib import Path

sys.path.append("../")
sys.path.append("../../")
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(1, os.path.abspath("../../"))

sys.path.insert(0, str(Path("..", "src").resolve()))

project = "Cómo encontrar el mejor jugador para tu Equipo de Fútbol"
copyright = "2024, Tomás Glauberman - Ignacio Pardo - Juan Ignacio Silvestri"
author = "Tomás Glauberman - Ignacio Pardo - Juan Ignacio Silvestri"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx_markdown_builder"]

# Elimina autodoc_mock_imports si no es necesario
# autodoc_mock_imports = ["src"]

templates_path = ["_templates"]
exclude_patterns = ["build/*"]

language = "es"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
