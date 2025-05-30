import sys

import numpy as np

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")

from pychu import Variable


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2) ** 2 + (x0 - 1) ** 2
    return y


x0 = Variable(0.0)
x1 = Variable(2.0)

y = rosenbrock(x0, x1)
y.backward()
print(x0.grad, x1.grad)
