#!/usr/bin/env python

import numpy as np
import torch as th
from torchvision import datasets, transforms
from nnexp import learn

if __name__ == '__main__':
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()) 
    learn('mnist_simple', dataset, split=(0.90, 0.0005, 0.0915))
