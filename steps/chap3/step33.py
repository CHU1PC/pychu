import sys

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")
from pychu import Variable  # noqa


def f(x):
    y = x**4 - 2 * x**2
    return y


x = Variable(2.0)
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad  # f'(x)
    x.cleargrad()
    gx.backward()
    gx2 = x.grad  # f''(x)

    # x = x - f'(x) / f''(x)
    x.data -= gx.data / gx2.data
