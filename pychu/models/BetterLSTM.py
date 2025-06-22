import numpy as np

import pychu.layers as L
from pychu.models import Model


class BetterLSTM(Model):
    def __init__(self, hidden_size, vocab_size,
                 dropout_ratio=0.5):
        """BetterLSTMの初期化

        Args:
            out_size (int): 出力ベクトルの次元数
            hidden_size (int): LSTMの中間層の次元数, hidden
            vocab_size (None or int): 語句数. Defaults to None.
            wordvec_size (int): 埋め込みベクトルの次元数. Defaults to 1000.
            dropout_ratio (float): Defaults to 0.5.
        """
        super().__init__()
        V, D, H = vocab_size, hidden_size, hidden_size  # noqa

        # 埋め込みベクトルの初期化
        embed_W = np.random.rand(V, D).astype(np.float32) * 0.01

        self.embed = L.TimeEmbedding(embed_W)

        # TimeLSTMは第1引数にoutsize, 第2引数にinsize, 返ってくるのは(N, T, hidden_size)
        self.lstm1 = L.TimeLSTM(hidden_size=hidden_size, in_size=D,
                                stateful=True)
        self.lstm2 = L.TimeLSTM(hidden_size=hidden_size, in_size=hidden_size,
                                stateful=True)

        self.fc = L.TimeLinear(out_size=V, in_size=H, W=embed_W.T)

        self.dropout = L.TimeDropout(dropout_ratio)

    def forward(self, xs):
        """
        Args:
            xs (ndarray): 入力ID列。shapeは (N, T)、Nはバッチサイズ、Tは時系列長
        """
        y = self.embed(xs)
        y = self.dropout(y)
        y = self.lstm1(y)
        y = self.dropout(y)
        y = self.lstm2(y)
        y = self.dropout(y)
        y = self.fc(y)
        return y
