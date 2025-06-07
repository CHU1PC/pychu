import sys

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning\ゼロから作る3\from_zero_3_github")
import pychu.functions as F
from pychu import Variable

x = Variable([[1, 1, 1],
             [1, 1, 1]])
W = Variable([[1],
              [1],
              [1]])
y = F.linear(x, W)
print(y.shape)
y1 = F.matmul(x, W)
print(y1.shape)
