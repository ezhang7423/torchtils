# type: ignore[attr-defined]
"""eds torch stuff"""

import sys
from functools import partial
from importlib import import_module
from importlib import metadata as importlib_metadata
from inspect import isclass
from pathlib import Path
from pkgutil import iter_modules

from blocks import *
from einops import rearrange, reduce

# import often used modules
from torch import einsum, nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from utils import *


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()

# # iterate through the modules in the current package
# package_dir = Path(__file__).resolve().parent
# for (_, module_name, _) in iter_modules([package_dir]):

#     # import the module and iterate through its attributes
#     module = import_module(f"{__name__}.{module_name}")
#     for attribute_name in dir(module):
#         attribute = getattr(module, attribute_name)

#         if isclass(attribute):
#             # Add the class to this package's variables
#             globals()[attribute_name] = attribute
