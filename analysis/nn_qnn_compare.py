from model.qnn_train import qnn_history, short_qnn_history, qnn_results, short_qnn_results
from model.nn import fair_nn_history, fair_nn_results
import seaborn as sns
import matplotlib.pyplot as plt


# Curves come from the Keras History objects of the runs above, never from
# hand-copied numbers. 'val_*' keys are measured on the held-out test set.
qnn_test_acc = qnn_history.history['val_hinge_accuracy']
short_qnn_test_acc = short_qnn_history.history['val_hinge_accuracy']
fair_nn_test_acc = fair_nn_history.history['val_binary_accuracy']
epochs = range(1, len(qnn_test_acc) + 1)


'''
Full QNN vs Short QNN
'''

print("Full QNN vs Short QNN")

plt.figure()
sns.barplot(x=["Full QNN", "Short QNN"],
            y=[qnn_results[1], short_qnn_results[1]])
plt.title('Final test accuracy: Full QNN vs Short QNN')
plt.ylabel('Test accuracy')
plt.show()

plt.figure()
plt.plot(epochs, qnn_test_acc, label='Full QNN')
plt.plot(epochs, short_qnn_test_acc, label='Short QNN')
plt.title('Test accuracy per epoch: Full QNN vs Short QNN')
plt.xlabel('Epoch')
plt.ylabel('Test accuracy')
plt.legend()
plt.grid(True)
plt.show()


'''
Full QNN vs Fair NN
'''

print("Full QNN vs Fair NN")

plt.figure()
sns.barplot(x=["Full QNN", "Fair NN"],
            y=[qnn_results[1], fair_nn_results[1]])
plt.title('Final test accuracy: Full QNN vs Fair NN')
plt.ylabel('Test accuracy')
plt.show()

# The QNN minimises hinge loss and the NN minimises binary cross-entropy, so
# their loss values are on different scales and are plotted side by side
# rather than on a shared axis.
fig, (ax_qnn, ax_nn) = plt.subplots(1, 2, figsize=(10, 4))
ax_qnn.plot(epochs, qnn_history.history['val_loss'])
ax_qnn.set_title('Full QNN test loss (hinge)')
ax_nn.plot(epochs, fair_nn_history.history['val_loss'])
ax_nn.set_title('Fair NN test loss (binary cross-entropy)')
for ax in (ax_qnn, ax_nn):
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)
plt.tight_layout()
plt.show()


'''
Full QNN vs Fair NN vs Short QNN
'''

print("Full QNN vs Fair NN vs Short QNN")

plt.figure()
plt.plot(epochs, qnn_test_acc, label='Full QNN')
plt.plot(epochs, short_qnn_test_acc, label='Short QNN')
plt.plot(epochs, fair_nn_test_acc, label='Fair NN')
plt.title('Test accuracy per epoch: all models')
plt.xlabel('Epoch')
plt.ylabel('Test accuracy')
plt.legend()
plt.grid(True)
plt.show()
