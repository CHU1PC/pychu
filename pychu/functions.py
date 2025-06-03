import sys

import numpy as np

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")
from pychu.core import Function, as_variable  # noqa


# reshape関数
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


# broadcastto関数
class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)

    def backward(self, gy):
        # broadcastはデータの数を拡張しているため単純に勾配の値が増やした軸への和になる
        return sum_to(gy, self.x_shape)


def broadcast_to(x, shape):
    """xをしたいshapeに拡張する

    Args:
        x (ndarray): 変換したいndarray入力
        shape (tuple, list): 変換したい形(行列)

    Returns:
        ndarray: 変換した後のndarrayを返す
    """
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


# sumto関数
class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        import pychu.utils as utils
        return utils.sum_to(x, self.shape)


def sum_to(x, shape):
    """xを指定したshapeになるように和をとって変形させる

    Args:
        x(ndarray): ndarrayの入力
        shape(tuple, list): 変換したい形(行列)

    Returns:
        ndarray: 変換したあとのndarrayを返す
    """
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


# transpose関数
class Transpose(Function):
    def forward(self, x):
        return np.transpose(x)

    def backward(self, gy):
        gx = transpose(gy)
        return gx


def transpose(x):
    return Transpose()(x)


# sum関数
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        return np.sum(x, axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


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
