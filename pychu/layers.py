import weakref

import numpy as np
import pychu.functions as F  # noqa
from pychu import cuda  # noqa
from pychu.core import Parameter  # noqa

"""
パラメータの自動化を行うためのファイル
"""


class Layer:
    def __init__(self):
        """
        self._paramsにはLayerのインスタンスが持つパラメータを保持するもの
        """
        self._params = set()

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

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()


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
