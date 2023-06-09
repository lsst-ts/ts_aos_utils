"""Sphinx configuration file for an LSST stack package.

This configuration only affects single-package Sphinx documentation builds.
For more information, see:
https://developer.lsst.io/stack/building-single-package-docs.html
"""

from documenteer.conf.pipelinespkg import *  # noqa

project = "ts_aos_utils"
html_theme_options["logotext"] = project  # noqa
html_title = project
html_short_title = project
