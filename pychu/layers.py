import os
import weakref

import numpy as np
import pychu.functions as F  # noqa
from pychu import cuda  # noqa
from pychu.core import Parameter  # noqa
from pychu.utils import pair
"""
パラメータの自動化を行うためのファイル
"""


class Layer:
    def __init__(self):
        """
        self._paramsにはLayerのインスタンスが持つParameter, Layerクラスを持つ
        """
        self._params = set()
        self.training = True

    def __setattr__(self, name, value):
        """インスタンス変数を宣言する際に呼び出されるメソッド

        Args:
            name (str): 名前を保持するための変数
            value (int, float, double)): 値を保持するための変数
        """
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError

    def params(self):
        """
        もしobjがLayerならLayerのなかのパラメータを取り出し
        もしobjがParameterならばそのまま返す
        """
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                # obj.params()のなかのparamsをyieldで取り出す
                yield from obj.params()
            else:
                yield obj

    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            # objがLayerであればparent_keyは最初にnameそこからname/nameとどんどん増えていく
            key = parent_key + "/" + name if parent_key else name

            if isinstance(obj, Layer):
                # objがLayerなら再帰でLayerの中身を調べる
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def train(self):
        """すべてのParameterをtraining = Falseに変える"""
        self.training = True
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                obj.train()

    def eval(self):
        """すべてのParameterをtraining = Trueに変える"""
        self.training = False
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                obj.eval()

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def save_weights(self, path):
        # GPU上にある時cupyを利用しているため, numpyを使うためにCPUへ
        self.to_cpu()

        params_dict: dict = {}
        # params_dictに_flatten_paramsでLayerの構成を格納する
        self._flatten_params(params_dict)

        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}

        # 途中でエラーが発生したときに未完成のファイルの残さないためにremoveする
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt):
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict: dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

    def _init_W(self, xp=np):
        """W(重さ)を初期化するメソッド

        Args:
            xp (_type_, optional): _description_. Defaults to np.
        """
        In, Out = self.in_size, self.out_size
        W_data = xp.random.randn(In, Out).astype(self.dtype) * np.sqrt(1 / In)
        self.W.data = W_data

    def forward(self, x):
        # in_sizeがNoneでself.WがNoneで初期化されていた場合forwardが呼び出されるときに入力と同じサイズだけ作る
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.dropout_ratio = p

    def forward(self, x):
        if self.training:
            xp = cuda.get_array_module(x)
            mask = xp.random.rand(*x.shape) > self.dropout_ratio
            scale = xp.array(1.0 - self.dropout_ratio).astype(x.dtype)
            y = x * mask / scale
            return y
        else:
            return x


class Conv2d(Layer):
    def __init__(self, out_channels, filter, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter = filter
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

    def _init_W(self, xp=np):
        """W(重さ)を初期化するメソッド

        Args:
            xp (cupy or numpy): Defaults to np.

        Notation:
            C: input channels
            OC: output channels
            FH: filter height
            FW: filter width
        """
        C, OC = self.in_channels, self.out_channels
        FH, FW = pair(self.filter)

        # Xavierの初期化
        scale = np.sqrt(C * FH * FW)
        W_data = xp.random.randn(OC, C, FH, FW).astype(self.dtype) / scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y
