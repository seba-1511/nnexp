#!/usr/bin/env python

import torch as th

def get_model(in_size, out_size, sizes=[64, 64]):
    params = [th.nn.Linear(in_size, sizes[0])]
    for i in range(1, len(sizes)):
        params.append(th.nn.Sigmoid())
        params.append(th.nn.Linear(sizes[i-1], sizes[i]))
    params.append(th.nn.Linear(sizes[-1], out_size))
    return params

def get_optimizer(model, lr=0.01):
    return th.optim.SGD(model.parameters(), lr=lr, momentum=0.95)

def get_loss():
    return th.nn.MSELoss()
