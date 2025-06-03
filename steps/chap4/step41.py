import sys

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")
import pychu.functions as F  # noqa
from pychu import Variable  # noqa

x = Variable([[1, 2], [3, 4], [5, 6]])
W = Variable([[5, 6, 7], [7, 8, 9]])
y = F.matmul(x, W)
y.backward()
print(y)
print(x.grad.shape)
print(W.grad.shape)
