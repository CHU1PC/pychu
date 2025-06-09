import math
import sys

import numpy as np

sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning"
                r"\ゼロから作る3\from_zero_3_github")

import pychu
import pychu.functions as F
from pychu import optimizers
from pychu.models import MLP

max_epoch = 300
batch_size= 30
hidden_size = 10
lr = 1.0
x, t = pychu.data