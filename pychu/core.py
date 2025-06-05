import contextlib
import weakref

import numpy as np

"""
このファイルはVariableの演算子(+-*/%)などの設定
variable変数が使えるメソッド(T, shape, size)などの設定を行う
"""


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True


# 関数をwith文で使えるようにするもの
# yieldの前
# try:
#    (     )   <- ここだwith文内の処理
#    yield
#    (     )   <- ここがwith文が終わった後の処理

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


# =============================================================================
# Variable / Function
# =============================================================================
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


class Variable:
    # これはnpなどのほかの変数と__array_priority__(標準では0.0)を比較して大きいほうの演算子を使う
    __array_priority__ = 1

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                data = np.array(data)

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        self.name = name

    def set_creator(self, func):
        self.creator = func

        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        """
        retain_grad: 途中の微分を覚えておくかどうか、デフォルトでは覚えない(False)
        create_graph: 2階微分以上を行うか行わないか、デフォルトでは行わない(False)
        """
        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            # こうすることで今までndarrayで作られていたものではなくつながりを持った計算になる
            # つながりがあればそれに対してもまたそいつが何によって作られたのかなどがわかる
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]

            # gxsを作る前にusing_configを使うことでFunction内の
            # Config.back_propが変化しつながりを作るか作らないかを決めれる
            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs, )

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        # これがx.grad += gxだとx.gradが上書きされるため
                        # つながりがなくなってしまう
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None

    def reshape(self, *shape_):
        if len(shape_) == 1 and isinstance(shape_[0], (tuple, list)):
            shape = shape_[0]
        from pychu import functions
        return functions.reshape(self, shape)

    def transpose(self):
        from pychu import functions
        return functions.transpose(self)

    @property
    def T(self):
        from pychu import functions
        return functions.transpose(self)

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        """これはprint(x)などでVariableが呼ばれたときになんと返すかを決めれる処理"""
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

    def __neg__(self):
        return neg(self)

    def sum(self, axis=None, keepdims=False):
        import pychu.functions as F
        return F.sum(self, axis, keepdims)


class Function:
    # 変数と関数のつながりが作られる
    # __call__はFunctionを継承するクラスを使うとき
    # 例 f = Add()
    #    y = f(x0, x1) <-ここで呼ばれる
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]  # type: ignore

        xs = [x.data for x in inputs]

        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Variable(y) for y in ys]

        # Configクラスのenable_backpropを呼び出しているだけ
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


# =============================================================================
# 演算子クラスと関数
# =============================================================================

def _setup_variable_operators():
    ops = [
        ('__add__', 'add'),
        ('__radd__', 'add'),
        ('__sub__', 'sub'),
        ('__rsub__', 'sub'),
        ('__mul__', 'mul'),
        ('__rmul__', 'mul'),
        ('__truediv__', 'div'),
        ('__rtruediv__', 'div'),
        ('__floordiv__', 'floordiv'),
        ('__rfloordiv__', 'floordiv'),
        ('__pow__', 'pow'),
        ('__mod__', 'mod'),
        ('__rmod__', 'mod'),
    ]

    for method, func in ops:
        if method.startswith('__r'):
            # setattrはVariableにmethod(__add__や__sub__, __mul__など)という名前で
            setattr(Variable, method,
                    lambda self, other, f=func: globals()[f](other, self))
        else:

            setattr(Variable, method,
                    lambda self, other, f=func: globals()[f](self, other))

    # インプレース演算子
    iops = [
        ('__iadd__', '__add__'),
        ('__isub__', '__sub__'),
        ('__imul__', '__mul__'),
        ('__itruediv__', '__truediv__'),
        ('__ifloordiv__', '__floordiv__'),
        ('__ipow__', '__pow__'),
        ('__imod__', '__mod__'),
    ]

    def _inplace_op(self, other, method):
        result = getattr(self, method)(other)
        self.data = result.data
        return self

    Variable._inplace_op = _inplace_op  # type: ignore

    for imethod, method in iops:
        setattr(Variable, imethod,
                lambda self, other, m=method: self._inplace_op(other, m))


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, -gy


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * (1 / x1), gy * (-x0 / x1 ** 2)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x0):
        return x0 ** self.c

    def backward(self, gy):
        x0 = self.inputs[0]
        return gy * (self.c * x0 ** (self.c - 1))


class FloorDiv(Function):
    def forward(self, x0, x1):
        return x0 // x1

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * (1 // x1), gy * (-x0 // x1 ** 2)


class Mod(Function):
    def forward(self, x0, x1):
        return x0 % x1

    def backward(self, gy):
        x0, x1 = self.inputs
        q = x0 // x1
        return gy * 1, gy * (-q)


def neg(x):
    return Neg()(x)


def add(x0, x1):
    return Add()(x0, x1)


def sub(x0, x1):
    return Sub()(x0, x1)


def mul(x0, x1):

    return Mul()(x0, x1)


def div(x0, x1):
    return Div()(x0, x1)


def floordiv(x0, x1):
    return FloorDiv()(x0, x1)


def pow(x, c):
    return Pow(c)(x)


def mod(x, c):
    return Mod()(x, c)
