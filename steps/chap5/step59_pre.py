import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pychu.layers as L  # noqa
import pychu.utils as utils  # noqa

rnn = L.RNN(10)
x = np.random.rand(1, 1)
h = rnn(x)
print(h.shape)
# y = rnn(np.random.rand(1, 1))
utils.plot_dot_graph(h)
