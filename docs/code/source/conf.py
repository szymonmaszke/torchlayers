# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import pytorch_sphinx_theme


def get_version():
    namespace = {}

    exec(open("../../../torchlayers/_version.py").read(), namespace)  # get version
    return namespace["__version__"]


sys.path.insert(0, os.path.abspath("../../.."))


# -- Project information -----------------------------------------------------

project = "torchlayers"
copyright = "2019, Szymon Maszke"
author = "Szymon Maszke"
version = get_version()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.katex",
    "javasphinx",
]

autosectionlabel_prefix_document = True

source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pytorch": ("https://pytorch.org/docs/stable/", None),
}

katex_prerender = True

napoleon_use_ivar = True

# # Called automatically by Sphinx, making this `conf.py` an "extension".
def setup(app):
    # NOTE: in Sphinx 1.8+ `html_css_files` is an official configuration value
    # and can be moved outside of this function (and the setup(app) function
    # can be deleted).
    html_css_files = [
        "https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/katex.min.css"
    ]

    # In Sphinx 1.8 it was renamed to `add_css_file`, 1.7 and prior it is
    # `add_stylesheet` (deprecated in 1.8).
    add_css = getattr(app, "add_css_file", app.add_stylesheet)
    for css_file in html_css_files:
        add_css(css_file)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

html_theme_options = {
    "related": "https://szymonmaszke.github.io/torchlayers/related.html",
    "roadmap": "https://github.com/szymonmaszke/torchlayers/blob/master/ROADMAP.md",
    "github_issues": "https://github.com/szymonmaszke/torchlayers/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc",
    "home": "https://szymonmaszke.github.io/torchlayers",
    "installation": "https://szymonmaszke.github.io/torchlayers/#installation",
    "github": "https://github.com/szymonmaszke/torchlayers",
    "docs": "https://szymonmaszke.github.io/torchlayers/#torchlayers",
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": False,
    "canonical_url": "https://szymonmaszke.github.io/torchlayers/",
}

# Other settings

default_role = "py:obj"  # Reference to Python by default
