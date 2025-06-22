import sys
import os

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pychu.layers as L  # noqa

x = np.random.randn(2, 3, 4).astype(np.float32)


layer = L.TimeLinear(out_size=5)

y = layer.forward(x)

print(x.shape)
print("-" * 50)
print(y.shape)
print("-" * 50)
print(y)
