# import ../model/qnn_train 
from model.qnn_train import qnn_results, short_qnn_results 
from model.nn import fair_cnn_results
import seaborn as sns
import matplotlib.pyplot as plt


'''
Full QNN vs Short QNN Analysis
'''

print("Full QNN vs Short QNN")

qnn_accuracy = qnn_results[1]
short_qnn_accuracy = short_qnn_results[1]

print(sns.barplot(x=["Full QNN", "Short QNN"],
            y=[qnn_accuracy, short_qnn_accuracy]))



qnn_acuracy = [0.7890, 0.8258, 0.8771]
short_qnn_accuracy = [0.7828, 0.7781, 0.7836]


# Plot training loss for all models:
plt.plot(qnn_acuracy, label='Full QNN Accuracy')
plt.plot(short_qnn_accuracy, label='Short QNN Accuracy')

# Graph settings:
plt.title('Testing Accuracy Comparison of Full_QNN & Short_QNN Models:')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


'''
Full QNN vs Fair NN
'''



print("Full QNN vs Fair NN")


qnn_accuracy = qnn_results[1]
fair_cnn_accuracy = fair_cnn_results[1]

sns.barplot(x=["Full_QNN", "Fair_NN"],
            y=[qnn_accuracy, fair_cnn_accuracy])



qnn_loss = [0.5449, 0.4083, 0.3631]
fair_cnn_loss = [0.6478, 0.4103, 0.2863]

# Plot training loss for all models
plt.plot(qnn_loss, label='Full QNN Loss')
plt.plot(fair_cnn_loss, label='Fair NN Loss')

# Graph settings
plt.title('Testing Loss Comparison of Full_QNN and Fair_NN Models')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()




'''
Full QNN vs Fair NN vs Short QNN
'''

print("Full QNN vs Fair NN vs Short QNN")


qnn_acuracy = [0.7890, 0.8258, 0.8771]
short_qnn_accuracy = [0.7828, 0.7781, 0.7836]
fair_cnn_acc = [0.5388, 0.8178, 0.8555]

# Plot training loss for all models
plt.plot(qnn_acuracy, label='Full_QNN Accuracy')
plt.plot(short_qnn_accuracy, label='Short_QNN Accuracy')
plt.plot(fair_cnn_acc, label='Fair_NN Accuracy')

# Graph settings
plt.title('Testing Accuracy Comparison of all of the models Models')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()