import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pychu.functions as F

x1 = np.random.rand(1, 3, 7, 7)
col1 = F.im2col(x1, filter=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)

x2 = np.random.rand(10, 3, 7, 7)
filter_size = (5, 5)
stride = (1, 1)
pad = (0, 0)
col2 = F.im2col(x2, filter=filter_size, stride=stride, pad=pad, to_matrix=True)
print(col2.shape)
