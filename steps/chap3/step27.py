import math
import sys

import numpy as np

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")
from pychu import Function, Variable  # noqa
from pychu.utils import plot_dot_graph  # noqa


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x = self.inputs[0].data
        return gy*np.cos(x)


def my_sin(x, threshold=1e-2):
    y = 0
    for n in range(100000):
        t = x**(2*n + 1) * (-1)**(n) / math.factorial(2*n + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


def sin(x):
    f = Sin()
    return f(x)


x0 = Variable(np.pi/4)
y0 = sin(x0)
y0.backward()

print(y0.data)
print(x0.grad)

x1 = Variable(np.pi/4)
y1 = my_sin(x1)
y1.backward()

x1.name = "x1"
y1.name = "y1"

plot_dot_graph(y1, verbose=False, to_file='my_sin.png')