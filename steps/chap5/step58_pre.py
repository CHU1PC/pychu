import sys
import os
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pychu.datasets  # noqa
import pychu  # noqa
import pychu.config as config  # noqa
from pychu.models import VGG16  # noqa

img_path = os.path.join(config.DATA_PATH, "zebra.jpg")

img = Image.open(img_path)
x = VGG16.preprocess(img)
x = x[np.newaxis]

model = VGG16(pretrained=True)
with pychu.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

model.plot(x, to_file=os.path.join(config.DATA_PATH, "vgg.pdf"))
labels = pychu.datasets.ImageNet.labels()
print(labels[predict_id])
