import os
import numpy as np
import pychu  # noqa
from pychu.models import Model  # noqa
import pychu.layers as L  # noqa
import pychu.functions as F  # noqa
from pychu import config  # noqa


class VGG16(Model):

    # VGGの重みは公開されているためもし重みがあれば使う
    def __init__(self, pretrained=False):
        super().__init__()

        self.conv1_1 = L.Conv2d(64, filter_size=3, stride=1, pad=1)
        self.conv1_2 = L.Conv2d(64, filter_size=3, stride=1, pad=1)
        self.conv2_1 = L.Conv2d(128, filter_size=3, stride=1, pad=1)
        self.conv2_2 = L.Conv2d(128, filter_size=3, stride=1, pad=1)
        self.conv3_1 = L.Conv2d(256, filter_size=3, stride=1, pad=1)
        self.conv3_2 = L.Conv2d(256, filter_size=3, stride=1, pad=1)
        self.conv3_3 = L.Conv2d(256, filter_size=3, stride=1, pad=1)
        self.conv4_1 = L.Conv2d(512, filter_size=3, stride=1, pad=1)
        self.conv4_2 = L.Conv2d(512, filter_size=3, stride=1, pad=1)
        self.conv4_3 = L.Conv2d(512, filter_size=3, stride=1, pad=1)
        self.conv5_1 = L.Conv2d(512, filter_size=3, stride=1, pad=1)
        self.conv5_2 = L.Conv2d(512, filter_size=3, stride=1, pad=1)
        self.conv5_3 = L.Conv2d(512, filter_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(1000)

        self.dropout = L.Dropout(p=0.2)

        if pretrained:
            vgg16_weight = os.path.join(config.WEIGHTS_PATH, "vgg16.npz")
            self.load_weights(vgg16_weight)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.pooling(x, 2, 2)
        x = F.reshape(x, (x.shape[0], -1))  # 2次元テンソルに変換
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

    @staticmethod
    def preprocess(image, size=(224, 224), dtype=np.float32):
        image = image.convert('RGB')
        if size:
            image = image.resize(size)
        image = np.asarray(image, dtype=dtype)
        image = image[:, :, ::-1]
        image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
        image = image.transpose((2, 0, 1))
        return image
