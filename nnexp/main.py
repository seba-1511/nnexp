#!/usr/bin/env python

from __future__ import print_function

import os
import numpy as np
import randopt as ro
import argparse

import torch as th
from torch.autograd import Variable
from torch.nn import Module

from tqdm import tqdm
from time import sleep

from plot import Plot, Container
from dataset import split_dataset
from default import get_model, get_optimizer, get_loss



parser = argparse.ArgumentParser(
    description='nnexp argument parser')
parser.add_argument(
    '--cuda', action='store_true', default=False, help='Train on GPU')
args = parser.parse_args()
args.cuda = False


class Network(Module):

    def __init__(self, layers):
        super(Network, self).__init__()
        self.layers = layers
        for i, l in enumerate(layers):
            setattr(self, 'l' + str(i), l)

    def forward(self, x):
        if len(x.size()) > 2:
            x = x.view(-1, 784)
        for layer in self.layers:
            x = layer(x)
        return x



def train(data, model, loss, optimizer):
    model.train()
    total_error = 0.0

    for X, y in tqdm(data, leave=False):
        if args.cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X).float(), Variable(y).float()
        optimizer.zero_grad()
        error = loss(model.forward(X), y)
        error.backward()
        optimizer.step()
        total_error += error
    return total_error.cpu().data.numpy()[0] / len(data)


def test(data, model, loss):
    model.eval()
    error = 0.0
    for X, y in tqdm(data, leave=False):
        if args.cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X).float(), Variable(y).float()
        error += loss(model(X), y)
    return error.cpu().data.numpy()[0] / len(data)


def learn(exp_name, dataset, model=None, optimizer=None, loss=None,
          rng_seed=1234, num_epochs=10, split=(0.7, 0.2, 0.1), bsz=64):

    if model is None:
        in_size = dataset[0][0].numel()
        if isinstance(dataset[0][1], (int, long, float, complex)):
            out_size = 1
        else:
            out_size = dataset[0][1].numel()
        model = get_model(in_size, out_size)
        model = Network(model)

    if loss is None:
        if isinstance(dataset[0][1], (int, long, float, complex)):
            reg = True
        else:
            reg = False
        loss = get_loss(regression=reg)

    if optimizer is None:
        optimizer = get_optimizer(model)

    opt_hyperparams = optimizer.param_groups[0]
    opt_hyperparams = {k: opt_hyperparams[k] for k in opt_hyperparams if not k == 'params'}

    exp = ro.Experiment(exp_name, {
        'model': str(model),
        'optimizer': str(optimizer),
        'opt_hyperparams': opt_hyperparams, 
        'loss': str(loss),
        'rng_seed': rng_seed,
        'num_epochs': num_epochs,
        'bsz': bsz,
        'split': split,
    })

    th.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    if args.cuda:
        th.cuda.manual_seed(rng_seed)
        model.cuda()

    print('Splitting dataset in ' + str(split[0]) + ' train, ' + str(split[1]) + ' Validation, ' + str(split[2]) + ' Test')
    dataset = split_dataset(dataset, split[0], split[1], split[2])
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = th.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True, **kwargs)
    dataset.use_valid()
    valid_loader = th.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True, **kwargs)
    dataset.use_test()
    test_loader = th.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True, **kwargs)

    train_errors = []
    valid_errors = []

    # Start training
    for epoch in range(num_epochs):
        print('\n\n', '-' * 20, ' Epoch ', epoch, ' ', '_' * 20)
        dataset.use_train()
        train_errors.append(train(train_loader, model, loss, optimizer))
        print('Training error: ', train_errors[-1])
        dataset.use_valid()
        valid_errors.append(test(valid_loader, model, loss))
        print('Validation error: ', valid_errors[-1])

    # Benchmark on Test
    dataset.use_test()
    test_error = test(test_loader, model, loss)
    print('Final Test Error: ', test_error)

    # Save experiment result
    exp.add_result(test_error, {
        'train_errors': train_errors,
        'valid_errors': valid_errors,
    })

    # Plot Results
    if not os.path.exists('./results'):
        os.mkdir('./results')

    p = Plot('Convergence')
    x = np.arange(0, len(train_errors), 1)
    p.plot(x, np.array(train_errors), label='Train')
    p.plot(x, np.array(valid_errors), label='Validation')
    p.set_axis('Epoch', 'Loss')
    
    b = Plot('Final Error')
    b.bar(x=[train_errors[-1], valid_errors[-1], test_error],
          labels=['Train', 'Validation', 'Test'])
    
    cont = Container(1, 2, title=exp_name)
    cont.set_plot(0, 0, p)
    cont.set_plot(0, 1, b)
    cont.save('./results/' + exp_name + '.pdf')
