import os
import sys

import numpy as np

gpu_enable = True

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False

from pychu import Variable  # noqa


def get_array_module(x):
    """xがnumpyかcupyかを返す

    Args:
        x (Variable, ndarray(cupy or numpy)): input

    Returns:
        xp (module): numpy or cupy module
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    """Numpy配列に変換する

    Args:
        x (Variable, ndarray(cupy or numpy)): input

    Returns:
        np.ndarray: Numpy配列
    """
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x):
    """Cupy配列に変換する

    Args:
        x (Variable, ndarray(cupy or numpy)): input

    Raises:
        Exception: CuPyがインストールされていない場合

    Returns:
        cupy.ndarray: CuPy配列
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception("CuPy cannot be loaded. Install CuPy!")

    return cp.asarray(x)
