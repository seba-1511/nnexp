#!/usr/bin/env python

import numpy as np
import torch as th
from torchvision import datasets, transforms
from nnexp import learn

if __name__ == '__main__':
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()) 
    learn('mnist_simple', dataset, split=(0.98, 0.0015, 0.0005))
