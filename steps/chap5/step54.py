import os
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pychu.functions as F  # noqa
from pychu import test_mode  # noqa

x = np.ones(5)
print(x)

# 学習
y = F.dropout(x)
print(y)

# テスト時
with test_mode():
    y = F.dropout(x)
    print(y)
