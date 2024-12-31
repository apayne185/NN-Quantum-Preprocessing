# Quantum Neural Networks: A Comparative Study on MNIST Dataset
This repository accompanies the research paper "Quantum Neural Networks: A Comparative Study of Quantum and Classical Neural Networks on the MNIST Dataset" by Shahaf Brenner, Anna Payne, Trinidad Roca, Juan Diego Fernandez, Sergio Verdugo, Noah Valderrama, and Pedro Torrado.

## Overview
This project explores the performance of Quantum Neural Networks (QNNs) and Classical Neural Networks (NNs) on a subset of the MNIST dataset. It evaluates:

* Training speed, accuracy, and loss differences.
* Quantum advantages such as superposition and entanglement.
* QNNs’ performance under resource-constrained and real-world conditions.

By leveraging TensorFlow Quantum, the research highlights the potential of QNNs for machine learning tasks and discusses their scalability and current limitations in a local setting.

## Project Structure
* data_preprocess.py: Scripts for MNIST data cleaning and resizing (28x28 → 4x4).
* qnn.py: Implementation of Quantum Neural Network (QNN) using Cirq and TensorFlow Quantum.
* nn.py: Implementation of a classical neural network for comparison.
* qnn_train.py: Training the QNN model. 
* nn_qnn_compare.py: Scripts for running and evaluating experiments across epochs and models.
* QNN_VS_Classic_NN.ipynb: Jupyter notebook for visualizing entire model performance, including accuracy and loss graphs.


## Key Features
### Data Preprocessing

* Resizing MNIST images to 4x4 for quantum hardware compatibility.
* Binarization of pixel values for qubit mapping.

### Model Architectures

* Classical Neural Network: Simple feed-forward NN with 37 parameters.
* Quantum Neural Network: Quantum circuit with 32 parameters leveraging quantum gates for feature representation.

### Comparison Metrics

* Accuracy, loss, and scalability.
* Testing scenarios with varying epochs (3, 5, and 10).

## Setup and Installation
### Dependencies
* Python 3.9+
* TensorFlow Quantum
* Cirq
* TensorFlow
* NumPy
* Matplotlib

### How to Run
* python data_preproces.py
* python qnn.py
* python nn.py
* python nn_qnn_compare.py


## Results Summary
* Testing Accuracy:
    * Full QNN: 90.6%
    * Fair NN: 82.7%
* Loss Comparison:
    * Full QNN: 0.345
    * Fair NN: 0.264

QNNs demonstrated potential advantages in noisy and high-dimensional data scenarios, while classical models excelled in low-noise, computationally simple tasks.


## Future Work
Future research will explore:

* Leveraging more advanced quantum hardware with higher qubit counts.
* Expanding to complex and noisy datasets.
* Integrating hybrid quantum-classical models for feature extraction and classification.
* Integrating IBM's Quantum Experience platform


## Citation
If you use this repository or reference our paper, please cite:
"Quantum Neural Networks: A Comparative Study of Quantum and Classical Neural Networks on the MNIST Dataset"


