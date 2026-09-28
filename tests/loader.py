"""Loads the addon from a checkout whose folder name is not a Python identifier."""

from __future__ import annotations

import importlib.util
import os
import sys


def load_addon(path: str, name: str = "vangers_world"):
    """Import the addon's ``__init__.py`` as a package called ``name``."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
