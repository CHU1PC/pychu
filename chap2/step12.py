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
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            if f is not None:
                x, y = f.input, f.output
                x.grad = f.backward(y.grad)

                if x.creator is not None:
                    funcs.append(x.creator)


class Function:
    # inputs -> *inputsにすることで可変長引数にできる
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]

        # self.forward(xs) -> self.forward(*xs)
        # こうすることで展開して渡してくれる
        ys = self.forward(*xs)

        # tupleじゃないときにtupleにする
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


# class Square(Function):
#     def forward(self, x):
#         y = x**2
#         return np.array(y)

#     def backward(self, gy):
#         x = self.input.data
#         gx = 2 * x * gy
#         return gx


# class Exp(Function):
#     def forward(self, x):
#         y = np.exp(x)
#         return y

#     def backward(self, gy):
#         x = self.input.data
#         gx = np.exp(x) * gy
#         return gx


class Add(Function):
    # forward(self, xs) -> forward(self, x0, x1)
    # これはFunctionのforward関数で仮引数を
    # forward(self, xs) -> forward(self, *xs)
    # に変えたことで実現できている
    def forward(self, x0, x1):
        y = x0 + x1
        return y


# def square(x):
#     f = Square()
#     return f(x)


# def exp(x):
#     f = Exp()
#     return f(x)


def add(x0, x1):
    f = Add()
    return f(x0, x1)


if __name__ == "__main__":
    x0, x1 = Variable(np.array(2)), Variable(np.array(3))
    # f = Add()がなくなった
    # y = ys[0]がなくなった
    y = add(x0, x1)
    print(y.data)
