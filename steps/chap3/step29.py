import sys

import numpy as np  # noqa

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")

from pychu import Variable  # noqa


###############################################################################
# ニュートン法
def f(x):
    y = x**4 - 2 * x**2
    return y


def gx2(x):
    return 12 * x**2 - 4
###############################################################################


x = Variable(2.0)
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)
