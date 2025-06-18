import os
import sys
import numpy as np  # noqa

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from pychu import Model  # noqa
import pychu.layers as L  # noqa
import pychu.functions as F  # noqa
from pychu import optimizers  # noqa
from pychu import datasets  # noqa
from pychu import DataLoader  # noqa


class MOD(Model):
    def __init__(self, hidden_size, activation=F.sigmoid, dropout_ratio=0.5):
        super().__init__()
        self.activation = activation
        self.hidden = hidden_size
        self.layers = []

        for i, out_size in enumerate(hidden_size[:-1]):
            layer = L.Linear(out_size)
            setattr(self, f"l{i}", layer)
            self.layers.append(layer)
            # Dropoutを中間層に追加
            dropout = L.Dropout(dropout_ratio)
            setattr(self, f"dropout{i}", dropout)
            self.layers.append(dropout)

        # 最後の出力層（Dropoutはかけない）
        out_size = hidden_size[-1]
        layer = L.Linear(out_size)
        setattr(self, f"l{len(hidden_size)-1}", layer)
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


max_epoch = 3
batch_size = 100

train_set  = datasets.MNIST(train=True)  # noqa
train_loader = DataLoader(train_set, batch_size)

model = MOD((100, 10))
optim = optimizers.SGD().setup(model)

if os.path.exists("my_mlp.npz"):
    model.load_weights("my_mlp.npz")

for epoch in range(max_epoch):
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optim.update()
        sum_loss += float(loss.data) * len(t)  # type: ignore

    print(f"epoch: {epoch + 1}, loss: {(sum_loss / len(train_set)):.4f}")

model.save_weights("my_mlp.npz")
