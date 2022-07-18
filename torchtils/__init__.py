# type: ignore[attr-defined]
"""eds torch stuff"""

import sys
from functools import partial
from importlib import import_module
from importlib import metadata as importlib_metadata
from inspect import isclass
from pathlib import Path
from pkgutil import iter_modules

from .utils import *
from .blocks import *

# import often used modules
from torch import einsum, nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from einops import rearrange, reduce
from torchvision.utils import save_image

def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()

