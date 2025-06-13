import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import pychu  # noqa
import pychu.datasets as datasets  # noqa
import pychu.functions as F  # noqa
import pychu.optimizers as optimizers  # noqa
from pychu import DataLoader  # noqa
from pychu.models import MLP  # noqa

max_epoch = 5
batch_size = 100
hidden_size = 1000


train_set = datasets.MNIST(train=True)
test_set = datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)


model = MLP((hidden_size, 10), activation=F.relu)
optim = optimizers.SGD().setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optim.update()

        sum_loss += float(loss.data) * len(t)  # type: ignore
        sum_acc += float(acc.data) * len(t)  # type: ignore
    print(f"epoch: {(epoch + 1)}")
    print(f"train loss: {(sum_loss / len(train_set)):.4f}, "
          f"accuracy: {(sum_acc / len(train_set)):.4f}")

    sum_loss, sum_acc = 0, 0
    with pychu.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)  # type: ignore
            sum_acc += float(acc.data) * len(t)  # type: ignore
    print(f"test loss: {(sum_loss / len(test_set)):.4f}, "
          f"tets accuracy: {(sum_acc / len(test_set)):.4f}")
