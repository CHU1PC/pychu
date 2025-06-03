import sys

import numpy as np

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")
import pychu.functions as F  # noqa
from pychu import Variable  # noqa

x = Variable(np.array([1.0, 2.0]))
v = Variable(np.array([4.0, 5.0]))


def f(x):
    t = x**2
    y = F.sum(t)
    return y


y = f(x)
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()

z = F.matmul(v, gx)
z.backward()
print(x.grad)
