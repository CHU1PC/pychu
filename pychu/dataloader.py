import math

import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        """datasetを受取りbatch_sizeに対してiterを作る

        Args:
            dataset (Dataset): Datasetのインスタンス
            batch_size (int): 1回のイテレーションで取得するデータ数
            shuffle (bool): データをシャッフルして取得するかどうか.デフォルトはTrue
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)

        self.reset()

    def reset(self):
        """
        インスタンス変数のiterationを0にリセットする
        """
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batchsize = self.iteration, self.batch_size
        batch_index = self.index[i * batchsize:(i + 1) * batchsize]
        batch = [self.dataset[i] for i in batch_index]
        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()
