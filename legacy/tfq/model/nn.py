import tensorflow as tf
from model.data_preprocess import y_train_nocon, y_test, x_test_bin, x_train_bin


EPOCHS = 3
BATCH_SIZE = 32


def fair_nn_model():
    """37-parameter MLP on the same binarized 4x4 inputs the QNN sees.

    This is a dense network, not a CNN: there are no convolutional layers.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(4,4,1)))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model


model = fair_nn_model()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)])

print("Classic NN model built.")
print(model.summary())


fair_nn_history = model.fit(x_train_bin,
          y_train_nocon,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_test_bin, y_test))

fair_nn_results = model.evaluate(x_test_bin, y_test)
print("Fair Classic NN model results.")
print(fair_nn_results)
