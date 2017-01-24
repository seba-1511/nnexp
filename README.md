# nnexp
A template for neural network experiments

## Goals
* Define your PyTorch Dataset, your NN, and your randopt variables. Call `train(...)`, and off you go.
* Automatic splitting (if desired) into train/valid/test data.
* Choose to use CUDA (automatically distributed) or not.
* Use sensible - but overrideable - defaults. (for optimizer, data pre-processing, hyper-params, etc...)
* Provide Mnist example.
* train function should only do 1 epoch.
* Sensible command-line arguments support.
