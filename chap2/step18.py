import contextlib
import weakref

import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                data = np.array(data)

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func

        self.generation = func.generation + 1

    # def backward(self):
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
###############################################################################
            if not retain_grad:
                for y in f.outputs:
                    # yはoutputsのなかの変数のためweakred(弱参照)である
                    # これによって一番最初の入力のgradだけが残る
                    y().grad = None
###############################################################################

    def cleargrad(self):
        self.grad = None


###############################################################################
class Config:
    # Trueならば学習モード, Falseならば推論モード
    enable_backprop = True


# contextlib.contextmanagerによりyieldの後ろ部分が実行されるのが
# with文内の処理が終わった後になる
@contextlib.contextmanager
def using_config(name, value):
    # getattrやsetattrは属性名をstr型で受取るためnameにはstr型が入る
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        # yieldは一時停止するキーワードで一度この関数が呼び出された場所に戻って
        # その呼び出された所の処理をしてから次の処理に進む
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    # with no_grad():で使えるようになる
    return using_config("enable_backprop", False)
###############################################################################


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]

        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Variable(y) for y in ys]

###############################################################################
        # ここのif文を追加することによって推論or学習に切り替えれる
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
###############################################################################

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


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


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


def add(x0, x1):
    f = Add()
    return f(x0, x1)


if __name__ == "__main__":
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(a, a)
    y.backward()
    print("a.grad:", a.grad)
    print("x.grad:", x.grad)

    x0, x1 = Variable(np.array(1.0)), Variable(np.array(1.0))
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()
    print("y.grad:", y.grad, ", t.grad:", t.grad)
    print("x0.grad:", x0.grad, ", x1.grad:", x1.grad)
