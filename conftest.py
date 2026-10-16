"""Shared test helpers.

Chapter folders have hyphens in their names, so their scripts are not
importable packages. The ``load_script`` fixture loads a script that sits next
to the test file, the way a reader would find it.
"""

import importlib.util

import pytest


@pytest.fixture
def load_script(request):
    def load(name):
        path = request.path.parent / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load
