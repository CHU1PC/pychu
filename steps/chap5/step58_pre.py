import sys
import os
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pychu.datasets  # noqa
import pychu  # noqa
import pychu.utils  # noqa
import pychu.config as config  # noqa
from pychu.models import VGG16  # noqa

url = "https://github.com/oreilly-japan/deep-learning-from-scratch-3/"\
    "raw/images/zebra.jpg"
img_path = pychu.utils.get_file(url)
img = Image.open(img_path)

x = VGG16.preprocess(img)
x = x[np.newaxis]

model = VGG16(pretrained=True)
model.eval()
y = model(x)
predict_id = np.argmax(y.data)

model.plot(x, to_file=os.path.join(config.DATA_PATH, "vgg.pdf"))
labels = pychu.datasets.ImageNet.labels()
print(labels[predict_id])
