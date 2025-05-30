import sys

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning\ゼロから作る3\from_zero_3_github")

import numpy as np  # noqa
from pychu import Variable  # noqa
from pychu.utils import get_dot_graph  # noqa

x0 = Variable(1.0)
x1 = Variable(1.0)
y = x0 + x1

x0.name = "x0"
x1.name = "x1"
y.name = "y"

txt = get_dot_graph(y, verbose=False)
with open("sample.dot", "w") as o:
    o.write(txt)
