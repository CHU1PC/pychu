import sys

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")

from pychu import Variable  # noqa


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2) ** 2 + (x0 - 1) ** 2
    return y


x0 = Variable(0.0)
x1 = Variable(2.0)

###############################################################################
lr = 1e-3

iters = 1000

for i in range(iters):
    print(x0, x1)
    y = rosenbrock(x0, x1)
    x0.cleargrad()
    x1.cleargrad()
    y.backward()
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad
###############################################################################
