import sys

import numpy as np

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")
import pychu.functions as F  # noqa
from pychu import Variable  # noqa

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = x.T
print(y)
y.backward(retain_grad=True)
print(x.grad)
