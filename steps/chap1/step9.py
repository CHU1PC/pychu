import numpy as np
from step7 import Function as FunctionBase
from step8 import Variable as VariableBase


class Variable(VariableBase):
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")

        self.data = data
        self.grad = None
        self.creator = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            if f is not None:
                x, y = f.input, f.output
                if y.grad is None:
                    y.grad = np.ones_like(y.data)
                x.grad = f.backward(y.grad)

                if x.creator is not None:
                    funcs.append(x.creator)


class Function(FunctionBase):
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output  # type: ignore
        return output


class Square(Function):
    def forward(self, x):
        y = x**2
        return np.array(y)

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return np.array(gx)


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return np.array(y)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return np.array(gx)


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


if __name__ == "__main__":
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))

    y.backward()
    print(x.grad)
