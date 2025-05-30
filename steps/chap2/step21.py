import contextlib
import weakref

import numpy as np


###############################################################################
def as_variable(obj):
    # objがVariableじゃない時にVariableに代入してから返す
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)
###############################################################################


class Variable:
    ###########################################################################
    # np.ndarrayとVariableの足し算の際にnp.ndarrayの__add__を使うかVariableの
    # __add__を使うかを決める、この時に__array_priority__の値が大きいほうが使われる
    __array_priority__ = 1
    ###########################################################################

    def __init__(self, data, name=None):
        # 本書でのas_array()はここで実装されておりVariableの型がnp.ndarrayでない時に
        # 変換するようにしている
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

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)


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
        #######################################################################
        # inputsにas_variableを適応してすべてをVariableのインスタンスにする
        inputs = [as_variable(x) for x in inputs]  # type: ignore
        #######################################################################

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


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


class Square(Function):
    def forward(self, x):
        y = x**2
        return np.array(y)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


def add(x0, x1):
    f = Add()
    return f(x0, x1)


def mul(x0, x1):
    f = Mul()
    return f(x0, x1)


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


if __name__ == "__main__":
    # x = Variable(np.array(2.0))
    # y = 3.0 * x + 1.0
    # print(y)
    # y.backward()
    # print(x.grad)

    x = Variable(np.array([1.0]))
    y = np.array([2.0]) + x
    print(y)
