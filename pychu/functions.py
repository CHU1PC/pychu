import sys

import numpy as np

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")
from pychu.core import Function  # noqa


# sin関数
class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x, = self.inputs
        return gy * cos(x)


def sin(x):
    return Sin()(x)


# cos関数
class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        x, = self.inputs
        return -gy * sin(x)


def cos(x):
    return Cos()(x)


# tanh関数
class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, gy):
        # tanhの微分は1 - f(x)**2のためtanhの出力結果が必要となる
        # またoutputsはweekref(弱参照)のため()がいる
        y = self.outputs[0]()
        return gy * (1 - y**2)  # type: ignore


def tanh(x):
    return Tanh()(x)
