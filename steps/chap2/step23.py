if "__file__" in globals():
    import sys
    sys.path.append(r"D:\program\programming\study\ゼロから作るdeeplearning\ゼロから作る3\from_zero_3_github")

import numpy as np
from pychu import Variable

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)
