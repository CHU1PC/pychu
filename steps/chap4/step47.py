import sys

import numpy as np

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")
import pychu.functions as F  # noqa
from pychu import Variable, as_variable  # noqa
from pychu.models import MLP  # noqa


def softmax1d(x):
    x = as_variable
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


model = MLP((10, 3))

x = np.array([[0.2, -0.4]])
y = model(x)
p = F.softmax(y)
print(y)
print(p)
