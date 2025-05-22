import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                data = np.array(data)

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            if f is not None:
                # x, y = f.input, f.output
                # x.grad = f.backward(y.grad)

                # gxsに微分したものが入っている
                gys = [output.grad for output in f.outputs]
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs, )

                # x.gradにgxs内にある微分したtupleを格納していく
                for x, gx in zip(f.inputs, gxs):
                    x.grad = gx

                    if x.creator is not None:
                        funcs.append(x.creator)


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]

        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Variable(y) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
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
        # x = self.inputs.data -> x = self.inputs[0].data
        # tupleのためそれの第一要素だけを取り出す
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
    x0, x1 = Variable(np.array(2.0)), Variable(np.array(3.0))
    y = add(square(x0), square(x1))
    y.backward()
    print(y.data)
    print(x0.grad)
    print(x1.grad)
