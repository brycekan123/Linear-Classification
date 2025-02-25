# Binary Classification with Linear Models

## Overview
This project implements a binary classification model using linear classification techniques. The model is trained to distinguish between two classes using a dataset loaded from external sources. It utilizes stochastic gradient descent (SGD) to optimize the model parameters, ensuring numerical stability and efficient learning.

## Mathematical Foundation
### Hypothesis Function
The model uses a linear function to estimate the probability of an instance belonging to class 1:

```
ŷ = σ(w · x + b)
```

where:
- `x` is the input feature vector,
- `w` is the weight vector,
- `b` is the bias term,
- `σ(z)` is the sigmoid function, defined as:

```
σ(z) = 1 / (1 + exp(-z))
```

This ensures that the output is in the range `(0,1)`, making it interpretable as a probability.

### Loss Function
The model is trained using the **binary cross-entropy loss**, given by:

```
L(y, ŷ) = -y * log(ŷ) - (1 - y) * log(1 - ŷ)
```

This loss function measures how well the predicted probability matches the actual class labels.

### Gradient Descent Update
To optimize the parameters `w` and `b`, stochastic gradient descent (SGD) is used. The updates for each parameter are computed as follows:

```
w ← w - η * ∇w L
b ← b - η * ∇b L
```

where `η` is the learning rate, and the gradients are given by:

```
∇w L = (ŷ - y) * x
∇b L = (ŷ - y)
```

This ensures the model iteratively improves its predictions by minimizing the loss.

## Implementation Details
### Data Loading (`data_loader.py`)
- Contains functions for loading and preprocessing datasets.
- Implements data parsing, normalization, and batching for training.
- Ensures proper handling of input data for compatibility with the classifier.

### Model Training (`classification.py`)
- Defines the `BinaryClassifier` class, which implements logistic regression.
- Contains the `train` function, which performs SGD optimization.
- Includes the forward pass computation using the sigmoid function and loss evaluation.
- Implements numerical stability improvements when computing exponentials.

### Binary Classification Execution (`bm_classify.py`)
- Serves as the entry point for training and testing the model.
- Initializes the `BinaryClassifier` and loads the dataset.
- Calls `train` from `classification.py` and evaluates the model's performance.
- Outputs training results and test accuracy.
