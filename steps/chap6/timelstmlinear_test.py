import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pychu.layers as L  # noqa

x = np.random.rand(2, 3, 4).astype(np.float32)  # (バッチサイズ, 時系列長, 入力次元)

lstm = L.TimeLSTM(hidden_size=5)
linear = L.TimeLinear(out_size=7)

y_lstm = lstm(x)
y = linear(y_lstm)

print("input:", x.shape)
print("lstm out:", y_lstm.shape)
print("linear out:", y.shape)
print("-" * 50)
print(y)
