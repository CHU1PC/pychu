import contextlib
import weakref

import numpy as np


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Variable:
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

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

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
            if f is not None:
                gys = [output().grad for output in f.outputs]
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs, )

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad += gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None

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
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

###############################################################################
# i〇〇〇系は自身のデータを変更させるからself.data = の形で自分自身を変化させる
    def __neg__(self):
        return neg(self)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __iadd__(self, other):
        self.data = (self+other).data
        return self

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __isub__(self, other):
        self.data = (self - other).data
        return self.data

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __imul__(self, other):
        self.data = (self * other).data
        return self

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __itruediv__(self, other):
        self.data = (self / other).data
        return self.data

    def __floordiv__(self, other):
        return floordiv(self, other)

    def __rfloordiv__(self, other):
        return floordiv(other, self)

    def __pow__(self, other):
        return pow(self, other)

    def __ipow__(self, other):
        self.data = (self ** other).data
        return self

    def __mod__(self, other):
        return mod(self, other)

    def __rmod__(self, other):
        return mod(other, self)

    def __imod__(self, other):
        self.data = (self % other).data
        return self
###############################################################################


class Config:
    enable_backprop = True


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


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]  # type: ignore

        xs = [x.data for x in inputs]

        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Variable(y) for y in ys]

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
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * (1 / x1), gy * (-x0 / x1 ** 2)


class Square(Function):
    def forward(self, x):
        y = x**2
        return np.array(y)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x0):
        return x0 ** self.c

    def backward(self, gy):
        x0 = self.inputs[0].data
        return gy * (self.c * x0 ** (self.c - 1))


class FloorDiv(Function):
    def forward(self, x0, x1):
        return x0 // x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * (1 // x1), gy * (-x0 // x1 ** 2)


class Mod(Function):
    def forward(self, x0, x1):
        return x0 % x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        q = x0 // x1
        return gy * 1, gy * (-q)


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


###############################################################################
def neg(x):
    f = Neg()
    return f(x)


def add(x0, x1):
    f = Add()
    return f(x0, x1)


def sub(x0, x1):
    f = Sub()
    return f(x0, x1)


def mul(x0, x1):
    f = Mul()
    return f(x0, x1)


def div(x0, x1):
    f = Div()
    return f(x0, x1)


def floordiv(x0, x1):
    f = FloorDiv()
    return f(x0, x1)


def square(x):
    f = Square()
    return f(x)


def pow(x, c):
    f = Pow(c)
    return f(x)


def mod(x, c):
    f = Mod()
    return f(x, c)


def exp(x):
    f = Exp()
    return f(x)
###############################################################################


if __name__ == "__main__":
    x = Variable(np.array(5.0))
    y = x % 3
    print(y)
