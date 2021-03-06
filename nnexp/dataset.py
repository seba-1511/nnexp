 #!/usr/bin/env python

import numpy as np
import torch as th
from torchvision import datasets, transforms
from random import choice


class DataSplitter(object):

    def __init__(self, data, train=0.7, valid=0.2, test=0.1):
        self.data = data
        self.train = train
        self.valid = valid
        self.test = test
        data_len = len(data)
        train_len = int(train * data_len)
        valid_len = int(valid * data_len)
        test_len = int(test * data_len)
        indexes = [x for x in range(0, data_len)]
        test_idx = []
        valid_idx = []
        train_idx = []

        for _ in range(test_len):
            n = choice(indexes)
            test_idx.append(n)
            indexes.remove(n)

        for _ in range(valid_len):
            n = choice(indexes)
            valid_idx.append(n)
            indexes.remove(n)

        train_idx = indexes

        self.sets = [train_idx, valid_idx, test_idx]
        self.lengths = [len(train_idx), len(valid_idx), len(test_idx)]
        self.use = 0

    def __len__(self):
        return self.lengths[self.use]

    def __getitem__(self, index):
        data_idx = self.sets[self.use][index]
        return self.data[data_idx]

    def use_train(self):
        self.use = 0

    def use_valid(self):
        self.use = 1

    def use_test(self):
        self.use = 2


class CSVDataset(object):
    def __init__(self, path, in_cols, out_cols, headers=False):
        data = np.genfromtxt(path, delimiter=',')
        if headers:
            data = data[1:, :]
        self.data = data
        self.X = data[:, :in_cols]
        self.y = data[:, in_cols:in_cols+out_cols]
        

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        X, y = th.from_numpy(self.X[index]), th.from_numpy(self.y[index])
        X = X.float()
        y = y.float()
        return X, y


def split_dataset(dataset, train=0.7, valid=0.2, test=0.1):
    return DataSplitter(dataset, train, valid, test)
