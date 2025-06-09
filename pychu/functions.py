import sys

import numpy as np

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")
import pychu.utils as utils  # noqa
from pychu.core import Function, as_variable  # noqa

"""
このファイルではcore.pyで作られvariableにもっと機能を増やしていくための
関数(function)を追加していく
"""


###############################################################################
# tensor用関数(tensor function)
###############################################################################


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
        Variable: 変換した後のVariableを返す
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
        Variable: 変換したあとのVariableを返す
    """
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


###############################################################################
# Variable用の数学関数(math function for Variable)
###############################################################################


# matmul関数
class MatMul(Function):
    def forward(self, x, W):
        return np.matmul(x, W)

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    """行列積を計算するもの

    Args:
        x (ndarray): テンソル
        W (nadarray): x・WのW

    Returns:
        Variable : 行列積をした後の行列を返す
    """
    return MatMul()(x, W)


# Linear関数
class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y = y + b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


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


# exp**x関数
class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


# sigmoid関数
class Sigmoid(Function):
    def forward(self, x):
        y = np.tanh(x * 0.5) * 0.5 + 0.5
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)  # type: ignore
        return gx


def sigmoid(x):
    return Sigmoid()(x)


# softmax関数
class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y = y / y.sum(axis=self.axis, keepdims=True)
        return y

    def backeward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


###############################################################################
# 損失関数(loss function)
###############################################################################


# 平均2乗誤差
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        return (diff**2).sum() / len(diff)

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gy = broadcast_to(gy, diff.shape)
        gx0 = gy * diff*(2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


# SoftmaxCrossEntropy関数
class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)
