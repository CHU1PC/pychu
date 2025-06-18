import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pychu  # noqa
import pychu.utils as utils  # noqa
from pychu.utils import pair, get_conv_outsize, get_deconv_outsize  # noqa
from pychu import cuda  # noqa
from pychu.core import Function, Variable, as_array, as_variable  # noqa

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
        xp = pychu.cuda.get_array_module(x)
        return xp.broadcast_to(x, self.shape)

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
# 画像変換用の関数(img translate function)
###############################################################################


class Im2col(Function):
    def __init__(self, filter, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = None
        self.filter = filter
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.filter, self.stride, self.pad,
                         self.to_matrix)
        return y

    def backward(self, gy):
        gx = col2im(gy, self.input_shape, self.filter, self.stride,
                    self.pad, self.to_matrix)
        return gx


def im2col(x, filter, stride=1, pad=0, to_matrix=True):
    """Extract patches from an image based on the filter.

    Args:
        x (`dezero.Variable` or `ndarray`): Input variable of shape
            `(N, C, H, W)`
        filter (int or (int, int)): Size of filter.
        stride (int or (int, int)): Stride of kernel.
        pad (int or (int, int)): Spatial padding width for input arrays.
        to_matrix (bool): If True the `col` will be reshaped to 2d array whose
            shape is `(N*OH*OW, C*KH*KW)`

    Returns:
        `dezero.Variable`: Output variable. If the `to_matrix` is False, the
            output shape is `(N, C, KH, KW, OH, OW)`, otherwise
            `(N*OH*OW, C*KH*KW)`.

    Notation:
    - `N` is the batch size.
    - `C` is the number of the input channels.
    - `H` and `W` are the height and width of the input image, respectively.
    - `KH` and `KW` are the height and width of the filters, respectively.
    - `SH` and `SW` are the strides of the filter.
    - `PH` and `PW` are the spatial padding sizes.
    - `OH` and `OW` are the the height and width of the output, respectively.
    """
    y = Im2col(filter, stride, pad, to_matrix)(x)
    return y


class Col2im(Function):
    def __init__(self, input_shape, filter, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = input_shape
        self.filter = filter
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = col2im_array(x, self.input_shape, self.filter, self.stride,
                         self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = im2col(gy, self.filter, self.stride, self.pad,
                    self.to_matrix)
        return gx


def col2im(x, input_shape, filter, stride=1, pad=0, to_matrix=True):
    return Col2im(input_shape, filter, stride, pad, to_matrix)(x)


def im2col_array(img, filter, stride, pad, to_matrix=True):
    batch_size, channel_size, height, width = \
        img.shape
    filter_height, filter_width = pair(filter)
    stride_height, stride_width = pair(stride)
    pad_height, pad_width = pair(pad)
    output_height = \
        get_conv_outsize(height, filter_height, stride_height, pad_height)
    output_width = \
        get_conv_outsize(width, filter_width, stride_width, pad_width)

    xp = cuda.get_array_module(img)
    if xp != np:
        col = _im2col_gpu(img, filter, stride, pad)
    else:
        img = np.pad(img,
                     ((0, 0), (0, 0),
                      (pad_height, pad_height + stride_height - 1),
                      (pad_width, pad_width + stride_width - 1)),
                     mode="constant", constant_values=(0, ))
        col = np.ndarray((batch_size,
                          channel_size,
                          filter_height, filter_width,
                          output_height, output_width), dtype=img.dtype)

        for j in range(filter_height):
            j_lim = j + stride_height * output_height
            for i in range(filter_width):
                i_lim = i + stride_width * output_width
                col[:, :, i, i, :, :] = \
                    img[:, :, j:j_lim:stride_height, i:i_lim:stride_width]
    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).\
            reshape((batch_size * output_height * output_width, - 1))

    return col


def col2im_array(col, img_shape, filter, stride, pad, to_matrix=True):
    batch_size, channel_size, height, width = img_shape
    filter_height, filter_width = pair(filter)
    stride_height, stride_width = pair(stride)
    pad_height, pad_width = pair(pad)
    output_height = \
        get_conv_outsize(height, filter_height, stride_height, pad_height)
    output_width = \
        get_conv_outsize(width, filter_width, stride_width, pad_width)

    if to_matrix:
        col = col.reshape(batch_size,
                          output_height, output_width,
                          channel_size,
                          filter_height, filter_width).transpose(
                              0, 3, 4, 5, 1, 2
                          )
    xp = cuda.get_array_module(col)
    if xp != np:
        img = _col2im_gpu(col, stride_height, stride_width, pad_height,
                          pad_width, height, width)
        return img
    else:
        img = np.zeros((batch_size, channel_size,
                        height + 2 * pad_height + stride_height - 1,
                        width + 2 * pad_width + stride_width - 1),
                       dtype=col.dtype)
        for j in range(filter_height):
            j_lim = j + stride_height * output_height
            for i in range(filter_width):
                i_lim = i + stride_width * output_width
                img[:, :, j:j_lim:stride_height, i:i_lim:stride_width] += \
                    col[:, :, j, i, :, :]
        return img[:, :,
                   pad_height:height + pad_height,
                   pad_width:width + pad_width]


def _im2col_gpu(img, filter, stride, pad):
    n, c, h, w = img.shape
    kh, kw = pair(filter)
    sy, sx = pair(stride)
    ph, pw = pair(pad)
    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    return col


def _col2im_gpu(col, sy, sx, ph, pw, h, w):
    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img = cuda.cupy.empty((n, c, h, w), dtype=col.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img


class Conv2d(Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x, W, b):
        xp = cuda.get_array_module(x)

        filter_height, filter_width = W.shape[2:]
        col = im2col_array(x, (filter_height, filter_width),
                           self.stride, self.pad, to_matrix=False)

        y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        y = xp.rollaxis(y, 3, 1)
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad,
                      outsize=(x.shape[2], x.shape[3]))
        gW = Conv2DGradW(self)(x, gy)

        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))

        return gx, gW, gb


def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2d(stride, pad)(x, W, b)


class Deconv2d(Function):
    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize

    def forward(self, x, W, b):
        xp = cuda.get_array_module(x)

        Weight = W
        stride_height, stride_width = self.stride
        pad_height, pad_width = self.pad
        channe_size, output_channel_size, filter_height, filter_width = \
            Weight.shape

        batch_size, channe_size, height, width = x.shape

        if self.outsize is None:
            output_height = \
                get_deconv_outsize(height, filter_height,
                                   stride_height, pad_height)
            output_width = \
                get_deconv_outsize(width, filter_width,
                                   stride_width, pad_width)
        else:
            output_height, output_width = pair(self.outsize)

        img_shape = (batch_size, output_channel_size,
                     output_height, output_width)

        gcol = xp.tensordot(Weight, x, (0, 1))
        gcol = xp.rollaxis(gcol, 3)

        y = col2im_array(gcol, img_shape, (filter_height, filter_width),
                         self.stride, self.pad, to_matrix=False)

        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        return y

    def backward(self, gy):
        x, W, b = self.inputs

        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)

        f = Conv2DGradW(self)
        gW = f(gy, x)

        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    return Deconv2d(stride, pad, outsize)(x, W, b)


class Conv2DGradW(Function):
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        filter_height, filter_width = W.shape[2:]
        self.filter = (filter_height, filter_width)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        xp = cuda.get_array_module(x)

        col = im2col_array(x, self.filter, self.stride, self.pad,
                           to_matrix=False)

        gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad,
                      outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy


class Pooling(Function):
    def __init__(self, filter, stride=1, pad=0):
        super().__init__()
        self.filter = filter
        self.stride = stride
        self.pad = pad

    def forward(self, x):

        col = im2col_array(x, self.filter, self.stride, self.pad,
                           to_matrix=False)
        # N: batch size
        # C: channel size
        # FH: filter height
        # FW: filter width
        # OH: output height
        # OW: output width
        N, C, FH, FW, OH, OW = col.shape

        col = col.reshape(N, C, FH * FW, OH, OW)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.filter = mpool2d.filter
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy):
        xp = cuda.get_array_module(gy)

        # N: batch size
        # C: channel size
        # OH: output height
        # OW: output width
        N, C, OH, OW = gy.shape

        # H: height
        # W: width
        N, C, H, W = self.input_shape

        # FH: filter height
        # FW: filter width
        FH, FW = pair(self.filter)

        gcol = xp.zeros((N * C * OH * OW * FH * FW), dtype=self.dtype)

        indexes = (self.indexes.ravel() +
                   xp.arange(0, self.indexes.size * FH * FW, FH * FW))

        gcol[indexes] = gy.ravel()
        gcol = gcol.resahpe(N, C, OH, OW, FH, FW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)

        gx = col2im_array(gcol, (N, C, H, W), self.filter, self.stride,
                          self.pad, to_matrix=False)

        return gx

    def backward(self, ggx):
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)


class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d):
        self.filter = mpool2d.filter
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        col = im2col_array(x, self.filter, self.stride, self.pad,
                           to_matrix=False)
        # N: batch size
        # C: channel size
        # FH: filter height
        # FW: filter width
        # OH: output height
        # OW: output width
        N, C, FH, FW, OH, OW = col.shape
        col = col.reshape(N, C, FH * FW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, FH * FW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


def pooling(x, filter, stride=1, pad=0):
    return Pooling(filter, stride, pad)(x)


###############################################################################
# Variable用の数学関数(math function for Variable)
###############################################################################


# matmul関数
class MatMul(Function):
    def forward(self, x, W):
        return x.dot(W)

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
            y += b
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
        xp = cuda.get_array_module(x)
        return xp.sin(x)

    def backward(self, gy):
        x, = self.inputs
        return gy * cos(x)


def sin(x):
    return Sin()(x)


# cos関数
class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.cos(x)

    def backward(self, gy):
        x, = self.inputs
        return -gy * sin(x)


def cos(x):
    return Cos()(x)


# tanh関数
class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.tanh(x)

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
        xp = cuda.get_array_module(x)
        return xp.exp(x)

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


# sigmoid関数
class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x * 0.5) * 0.5 + 0.5
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
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backeward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


# ReLU関数
class ReLU(Function):
    def forward(self, x):
        return np.maximum(x, 0.0)

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)


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
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


###############################################################################
# accuracy/ dropout/
###############################################################################


def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))


def dropout(x, dropout_ratio=0.1):
    x = as_variable(x)

    if pychu.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x
