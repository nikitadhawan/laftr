from pathlib import Path
from os.path import dirname, basename, isfile, join
import importlib
import os

module_names = []

base_path = Path(dirname(__file__)).resolve()

for dirpath, dirnames, filenames in os.walk(dirname(__file__)):
    relative_path = str(Path(dirpath).resolve().relative_to(base_path)).replace(
        "/", ".")
    relative_path = relative_path if relative_path.startswith(
        ".") else "." + relative_path

    for name in filenames:
        if name.endswith(".py") and name != "__init__.py":
            name = name[:-3]
            if not relative_path.endswith("."):
                mod_name = relative_path + "." + name
            else:
                mod_name = relative_path + name
            module_names.append(mod_name)

for module in module_names:
    sub_mod = importlib.import_module(module, __name__)
