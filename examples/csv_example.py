#!/usr/bin/env python

import numpy as np
import torch as th
from nnexp import learn, Network
from nnexp.dataset import CSVDataset

if __name__ == '__main__':
    dataset = CSVDataset('./data/spindle/trainingDataRnHsinD.csv', in_cols=5, out_cols=2, headers=True)
    model = [th.nn.Linear(5, 16), th.nn.Tanh(), th.nn.Linear(16, 2)]
    model = Network(model)
    opt = th.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss = th.nn.SmoothL1Loss()
    learn('csv_example', dataset, model=model, optimizer=opt, loss=loss, num_epochs=20)
