import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pychu.config as config  # noqa
from pychu import Layer  # noqa
from pychu import utils  # noqa


class Model(Layer):
    def plot(self, *inputs,
             to_file=os.path.join(config.DATA_PATH, "model.png")):

        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)
