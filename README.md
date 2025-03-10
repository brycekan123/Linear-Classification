# Binary and Multiclass Classification with Linear Models

## Overview
This project implements both binary and multiclass classification models using linear classification techniques. Stochastic Gradient Descent (SGD) and Gradient Descent (GD) are used to optimize model parameters, ensuring numerical stability and efficient learning. I only implemented bm_classify.py.

## Mathematical Foundation

### Binary Classification

#### Hypothesis Function
The binary classification model estimates the probability of an instance belonging to class 1 using the function:

$$
ŷ = \sigma(w \cdot x + b)
$$

where:
- $x$ is the input feature vector,
- $w$ is the weight vector,
- $b$ is the bias term,
- $\sigma(z)$ is the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + \exp(-z)}
$$

This ensures the output is in the range $(0,1)$, making it interpretable as a probability.

#### Loss Function
Binary classification is trained using binary cross-entropy loss:

$$
L(y, ŷ) = -y \cdot \log(ŷ) - (1 - y) \cdot \log(1 - ŷ)
$$

This loss function measures how well the predicted probability matches the actual class labels.

#### Gradient Descent Update
To optimize $w$ and $b$, SGD is used with the updates:

$$
w \leftarrow w - \eta \cdot \nabla_w L
$$  
$$
b \leftarrow b - \eta \cdot \nabla_b L
$$

where $\eta$ is the learning rate, and the gradients are computed as:

$$
\nabla_w L = (ŷ - y) \cdot x
$$  
$$
\nabla_b L = (ŷ - y)
$$

### Multiclass Classification

#### Hypothesis Function
The multiclass classification model extends binary classification using the softmax function:

$$
ŷ = \text{softmax}(W \cdot X + b)
$$

where:
- $W$ is the weight matrix $(C \times D)$, where $C$ is the number of classes,
- $X$ is the input feature matrix,
- $b$ is the bias vector,
- softmax is defined as:

$$
\text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

For numerical stability, we compute:

$$
\text{softmax}(z_i) = \frac{\exp(z_i - \max(z))}{\sum_j \exp(z_j - \max(z))}
$$

#### Loss Function
The loss function for multiclass classification is categorical cross-entropy:

$$
L(y, ŷ) = -\sum_c y_c \cdot \log(ŷ_c)
$$

where $y_c$ is the one-hot encoded label.

#### Gradient Descent Update
SGD and GD are used to update $W$ and $b$:

$$
W \leftarrow W - \eta \cdot \nabla_W L
$$  
$$
b \leftarrow b - \eta \cdot \nabla_b L
$$

where the gradients are computed as:

$$
\nabla_W L = (\hat{ŷ} - y) \cdot X
$$  
$$
\nabla_b L = (\hat{ŷ} - y)
$$

## Implementation Details

### Data Loading (`data_loader.py`)
- Loads and preprocesses datasets.
- Implements data parsing, normalization, and batching.
- Ensures proper handling of input data for classifier compatibility.

### Model Training (`classification.py`)
- **BinaryClassifier**: Implements logistic regression for binary classification.
- **MulticlassClassifier**: Implements multinomial logistic regression for multiclass classification.
- **Training functions**: Perform SGD and GD optimization.
- Includes forward pass computation using the sigmoid/softmax function and loss evaluation.
- Implements numerical stability improvements.

### Classification Execution (`bm_classify.py`)
- Entry point for training and testing both binary and multiclass models.
- Initializes the classifiers and loads the dataset.
- Calls `train` functions from `classification.py` and evaluates performance.
- Outputs training results and test accuracy.
<img width="388" alt="Screenshot 2025-03-09 at 10 13 05 PM" src="https://github.com/user-attachments/assets/6ac5fa90-236c-4cc6-99f6-eca7f3978a96" />

