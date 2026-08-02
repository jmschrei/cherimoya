# Sphinx documentation build configuration file

import os
import sys

# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'Cherimoya'
copyright = '2026, Jacob Schreiber'
author = 'Jacob Schreiber'
release = '0.2.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
    '.ipynb_checkpoints', '**/.ipynb_checkpoints',
    '.ipynb_checkpoints/**', '**/.ipynb_checkpoints/**']

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstrings = False
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Mock heavy imports so docs can build without GPU/torch/triton
autodoc_mock_imports = [
    'torch', 'triton', 'numpy', 'scipy', 'pandas', 'h5py',
    'tqdm', 'seaborn', 'tangermeme', 'bpnetlite', 'modisco',
    'macs3', 'bam2bw', 'sklearn', 'joblib',
]

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_title = 'Cherimoya'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#5a9a68",
        "color-brand-content": "#4a8a58",
        "color-admonition-background": "rgba(90, 154, 104, 0.08)",
        "color-link": "#5a9a68",
        "color-link--hover": "#3d7a4a",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7cc08a",
        "color-brand-content": "#8dd09a",
        "color-admonition-background": "rgba(124, 192, 138, 0.1)",
        "color-link": "#7cc08a",
        "color-link--hover": "#a0daa8",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/jmschrei/cherimoya",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
    ],
}

# Logo
html_logo = '../imgs/cherimoya.png'
html_favicon = '../imgs/cherimoya.png'
