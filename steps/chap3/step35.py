import sys

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")

import pychu.functions as F  # noqa
from pychu import Variable  # noqa
from pychu.utils import plot_dot_graph  # noqa

x = Variable(1.0)
y = F.tanh(x)
x.name = "x"
y.name = "y"
y.backward(create_graph=True)

iters = 9
for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = "gx" + str(iters+1)
plot_dot_graph(gx, verbose=False,
               sto_file=r"D:\program\programming\study\ゼロから作るdeeplearning"
               r"\ゼロから作る3\from_zero_3_github\steps\chap3"
               r"\tanh.png")
