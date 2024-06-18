# ChebyKAN-Keras

## Overview
ChebyKAN-Keras is a Keras implementation of the Chebyshev-based Kolmogorov-Arnold Network (ChebyKAN), inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials. This project demonstrates the application of ChebyKAN on the MNIST dataset for digit classification.

## Features

* Custom Keras layer implementation using Chebyshev polynomials
* Model architecture designed and tested on the MNIST dataset
* Training and evaluation scripts included
* Calculation of performance metrics such as accuracy, precision, recall, and F1 score


## Requirements

* Python 3.10 or higher
* TensorFlow 2.16.1 or higher

## Installation
1. Clone the repository

```bash
git clone https://github.com/BAW2501/ChebyKAN-Keras.git
cd ChebyKAN-Keras
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the example script

```bash
python Example.py
```
# Usage

## Load and preprocess the MNIST dataset

The script automatically loads and preprocesses the MNIST dataset, reshaping and normalizing the images.

## Define the model

The model is defined using custom ChebyKAN layers and compiled with appropriate loss functions and metrics.

## Train the model

```python
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```




