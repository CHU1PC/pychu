import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pychu  # noqa
import pychu.functions as F  # noqa
from pychu import Variable  # noqa

x1 = np.random.rand(1, 3, 7, 7)
col1 = F.im2col(x1, filter=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)

x2 = np.random.rand(10, 3, 7, 7)
filter_size = (5, 5)
stride = (1, 1)
pad = (0, 0)
col2 = F.im2col(x2, filter=filter_size, stride=stride, pad=pad, to_matrix=True)
print(col2.shape)

N, C, H, W = 1, 5, 15, 15
OC, (KH, KW) = 8, (3, 3)

x = Variable(np.random.rand(N, C, H, W))
W = np.random.rand(OC, C, KH, KW)  # type: ignore
y = F.conv2d(x, W, b=None, stride=1, pad=1)
y.backward()

print(y.shape)
print(x.grad.shape)
