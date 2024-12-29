import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import collections

# visualization tools
# %matplotlib inline
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
from IPython.display import display




(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescaling images from [0,255] to [0.0,1.0] range:
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

print("MNIST Dataset has been loaded using Keras")
print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))


def filter_36(x, y):
    keep = (y == 3) | (y == 6)
    x, y = x[keep], y[keep]
    y = y == 3
    return x,y

x_train, y_train = filter_36(x_train, y_train)
x_test, y_test = filter_36(x_test, y_test)

print("\nMNIST Dataset reduced to 3, 6 samples and the label (y) is converted to bool")
print("Number of reduced training examples:", len(x_train))
print("Number of reduced test examples:", len(x_test))
print(y_train[0])
plt.imshow(x_train[0, :, :, 0])
plt.colorbar()


x_train_small = tf.image.resize(x_train, (4,4)).numpy()
x_test_small = tf.image.resize(x_test, (4,4)).numpy()
print("MNIST Dataset examples downsized from 28x28 to 4x4 pixels")
print(y_train[0])
plt.imshow(x_train_small[0,:,:,0], vmin=0, vmax=1)
plt.colorbar()



def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each sample:
    for x,y in zip(xs,ys):
       orig_x[tuple(x.flatten())] = x
       mapping[tuple(x.flatten())].add(y)

    new_x = []
    new_y = []
    for flatten_x in mapping:
      x = orig_x[flatten_x]
      labels = mapping[flatten_x]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(next(iter(labels)))
      else:
          # Remove images that match more than one label.
          pass

    num_uniq_3 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
    num_uniq_6 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)

    print("Number of unique samples = ", len(mapping.values()))
    print("Number of samples with lable 3s = ", num_uniq_3)
    print("Number of samples with lable 6s = ", num_uniq_6)
    print("Number of samples with contradicting labels (both 3 and 6) = ", num_uniq_both, "\n")
    print("Initial number of samples = ", len(xs))
    print("Number of Non-contradicting samples = ", len(new_x))

    return np.array(new_x), np.array(new_y)

print("Faulty samples removed from dataset")
x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)




'''
Preprocessing for QNN
'''

THRESHOLD = 0.5

x_train_bin = np.array(x_train_nocon > THRESHOLD, dtype=np.float32)
x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.float32)
_ = remove_contradicting(x_train_bin, y_train_nocon)



def convert_to_circuit(image):
    """Encode binary values of the classical image into a quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


x_train_circ = [convert_to_circuit(x) for x in x_train_bin]
x_test_circ = [convert_to_circuit(x) for x in x_test_bin]

print("Cirq Circuit visualization of 1st Data Sample in preprocessed dataset")
display(SVGCircuit(x_train_circ[0]))


bin_img = x_train_bin[0,:,:,0]
indices = np.array(np.where(bin_img)).T
print("Code visualization of 1st Data Sample")
print(indices)


x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
print("Circuits converted to tensors for QNN. Preprocessing Done.")