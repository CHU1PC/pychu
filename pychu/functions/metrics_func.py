import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pychu import as_variable, as_array  # noqa
from pychu.core import Variable  # noqa


def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))
