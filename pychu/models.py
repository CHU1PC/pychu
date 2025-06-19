import os
import numpy as np
import pychu.config as config  # noqa
import pychu.functions as F  # noqa
import pychu.layers as L  # noqa
from pychu import Layer  # noqa
from pychu import utils  # noqa


class Model(Layer):
    def plot(self, *inputs,
             to_file=os.path.join(config.DATA_PATH, "model.png")):

        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            # self.l(str(i)) = layer
            setattr(self, "l" + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class VGG16(Model):
    WEIGHTS_PATH = \
        os.path.join(os.path.dirname(__file__), "weight", "vgg16.npz")

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

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
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
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
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
