
.PHONY: all

all: mnist

install:
	pip install -r requirements.txt
	python setup.py develop

mnist:
	python examples/mnist_simple.py

custom:
	python examples/mnist_custom.py
