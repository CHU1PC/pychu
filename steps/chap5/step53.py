import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pychu.functions as F  # noqa
from pychu import DataLoader, datasets, optimizers  # noqa
from pychu.models import MLP  # noqa

max_epoch = 3
batch_size = 100

train_set  = datasets.MNIST(train=True)  # noqa
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
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
