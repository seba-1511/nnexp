 #!/usr/bin/env python

import torch as th
from torchvision import datasets, transforms


class DataSplitter(object):

    def __init__(self, data, train=0.7, valid=0.2, test=0.1):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def use_train(self):
        pass

    def use_valid(self):
        pass

    def use_test(self):
        pass


class CSVDataset(object):
    def __init__(self, path, train_cols, test_cols):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


def split_dataset(dataset, train=0.7, valid=0.2, test=0.1):
    train = DataSplitter(dataset, train, valid, test)
    train.use_train()
    valid = DataSplitter(dataset, train, valid, test)
    valid.use_valid()
    test = DataSplitter(dataset, train, valid, test)
    test.use_test()
    return train, valid, test
