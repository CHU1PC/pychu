import numpy as np
from step1 import Variable
from step3 import Exp, Square


def numerical_diff(f, x, eps=1e-4):
    # Variableでデータをobject化
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    # fには任意のfunctionが入る
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


if __name__ == "__main__":

    def f(x):
        A = Square()
        B = Exp()
        C = Square()
        return C(B(A(x)))

    x = Variable(np.array(0.5))
    dy = numerical_diff(f, x)
    print(dy)
