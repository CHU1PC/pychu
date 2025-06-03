import sys
import time

import matplotlib.pyplot as plt
import numpy as np  # noqa

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")
import pychu.functions as F  # noqa
from pychu import Variable  # noqa

start_time = time.time()
x = Variable(np.linspace(-7, 7, 10000))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data.flatten()]

for i in range(3):
    logs.append(x.grad.data.flatten())
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

print(f"end time: {time.time() - start_time}")
labels = ["y=sin(x)", "y'=cos(x)", "y''=-sin(x)", "y'''=-cos(x)"]
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc="lower right")
plt.show()
