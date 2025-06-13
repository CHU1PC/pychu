import cupy as cp
import numpy as np

x = np.array([1, 2, 3])
xp = cp.get_array_module(x)
print(xp)

x = cp.array([1, 2, 3])
xp = cp.get_array_module(x)
print(xp)
