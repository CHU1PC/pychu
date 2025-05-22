import os
import sys

sys.path.append(os.path.dirname(
    r"D:\program\programming\study\ゼロから作るdeeplearning\ゼロから作る3\chap1"
    ))
import numpy as np
from step1 import Variable


class Function:
    # callは
    # f = Function()
    # f(input) <-- のここで実行さえる
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


# pythonではclassの宣言のなかにclassを書くことで継承できる
class Square(Function):
    def forward(self, x):
        return x**2


if __name__ == "__main__":
    x = Variable(np.array(10))
    f = Square()
    y = f(x)

    print(type(y))
    print(y.data)
