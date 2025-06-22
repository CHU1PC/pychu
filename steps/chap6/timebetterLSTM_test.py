import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from pychu.models import BetterLSTM  # noqa


def test_better_lstm_forward_shape():
    vocab_size = 10
    hidden_size = 8   # ★ここをwordvec_sizeと同じにする
    dropout_ratio = 0.0

    model = BetterLSTM(hidden_size=hidden_size, vocab_size=vocab_size,
                       dropout_ratio=dropout_ratio)

    batch_size = 4
    time_size = 5
    xs = np.random.randint(0, vocab_size, (batch_size, time_size))

    out = model.forward(xs)
    assert out.shape == (batch_size, time_size, vocab_size)


def test_better_lstm_weight_tying():
    vocab_size = 7
    hidden_size = 5

    model = BetterLSTM(hidden_size=hidden_size, vocab_size=vocab_size,
                       dropout_ratio=0.0)

    embed_W = model.embed.W
    # TimeLinearの重みは model.fc.linear.W
    fc_W = model.fc.linear.W
    assert np.allclose(embed_W.T, fc_W.data)


if __name__ == "__main__":
    pytest.main([__file__])
