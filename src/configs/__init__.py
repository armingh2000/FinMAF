import importlib
import pkgutil

for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    # Load the module
    module = importlib.import_module("." + module_name, package=__name__)

    # Iterate through its attributes and add them to the global scope
    for attr_name in dir(module):
        # Optionally, filter out module-private names or unwanted attributes
        if not attr_name.startswith("_"):
            globals()[attr_name] = getattr(module, attr_name)
