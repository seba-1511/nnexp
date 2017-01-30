
.PHONY: all

all: custom

install:
	pip install -r requirements.txt
	python setup.py develop

mnist:
	python examples/mnist_simple.py 

custom:
	python examples/csv_example.py
