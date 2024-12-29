import tensorflow as tf
from model.qnn import y_train_hinge, model, y_test 
from model.data_preprocess import y_train_nocon, x_test_bin, x_train_bin



def Fair_CNN_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(4,4,1)))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model


model = Fair_CNN_model()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

print("Classic NN model built.")
print(model.summary())


model.fit(x_train_bin,
          y_train_nocon,
          batch_size=32,
          epochs=3,
          verbose=1,
          validation_data=(x_test_bin, y_test))

fair_cnn_results = model.evaluate(x_test_bin, y_test)
print("Fair Classic NN model results.")
print(fair_cnn_results)