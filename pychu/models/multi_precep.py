import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pychu  # noqa
from pychu.models import Model  # noqa
import pychu.functions as F  # noqa
import pychu.layers as L  # noqa


class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            # self.l(str(i)) = layer
            setattr(self, "l" + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
