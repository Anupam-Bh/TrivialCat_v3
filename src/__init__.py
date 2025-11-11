# src/__init__.py
import pkgutil
import importlib
import inspect

__all__ = []  # the list of names to export on `from src import *`

# Iterate every module in this package
for finder, module_name, is_pkg in pkgutil.iter_modules(__path__):
    full_name = f"{__name__}.{module_name}"
    module   = importlib.import_module(full_name)

    # For each attribute in the module:
    for name, obj in inspect.getmembers(module, 
                                        lambda o: inspect.isclass(o) or inspect.isfunction(o)):
        if not name.startswith('_'):
            globals()[name] = obj   # inject into src namespace
            __all__.append(name)
