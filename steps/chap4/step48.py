import math
import sys

import numpy as np

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")

import pychu  # noqa
import pychu.datasets as datasets  # noqa
import pychu.functions as F  # noqa
from pychu import optimizers  # noqa
from pychu.models import MLP  # noqa

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0
x, t = datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)  # type: ignore

    avg_loss = sum_loss / data_size
    print(f"epoch {epoch + 1}, loss {avg_loss:.2f}")
