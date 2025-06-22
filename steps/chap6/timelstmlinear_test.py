import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pychu.layers as L  # noqa


def test_timelstmlinear_shapes():
    x = np.random.rand(2, 3, 4).astype(np.float32)  # (バッチサイズ, 時系列長, 入力次元)

    lstm = L.TimeLSTM(hidden_size=5, in_size=4)
    linear = L.TimeLinear(out_size=7, in_size=5)

    y_lstm = lstm(x)
    y = linear(y_lstm)

    # LSTM出力shape: (2, 3, 5)
    assert y_lstm.shape == (2, 3, 5) or \
        getattr(y_lstm, "shape", None) == (2, 3, 5)
    # Linear出力shape: (2, 3, 7)
    assert y.shape == (2, 3, 7) or getattr(y, "shape", None) == (2, 3, 7)


if __name__ == "__main__":
    pytest.main([__file__])
