import math
from collections import namedtuple
from inspect import isfunction


# constants

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])

# helpers functions

def register_buffer(self, name, val): # register as same type as weights for lightning modules
    self.register_buffer(name, val.type(self.dtype))

def exists(x):
    """Check that x is not None"""
    return x is not None


def default(val, d):
    """If val exists, return it. Otherwise, return d"""
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        yield from dl


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

from .progress import *
from .arrays import *
from .model_wrappers import *