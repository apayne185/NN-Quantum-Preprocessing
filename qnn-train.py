from qnn import y_train_hinge, model, y_test 
from data_preprocess import x_train_tfcirc, x_test_tfcirc, y_test_hinge




EPOCHS = 3
BATCH_SIZE = 32
NUM_EXAMPLES = 500

x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]



short_qnn_history = model.fit(
      x_train_tfcirc_sub, y_train_hinge_sub,
      batch_size=32,
      epochs=EPOCHS,
      verbose=1,
      validation_data=(x_test_tfcirc, y_test_hinge))

short_qnn_results = model.evaluate(x_test_tfcirc, y_test)
print("Short (Partial Dataset -500 examples) - Training")
print(short_qnn_results)


EPOCHS = 3
BATCH_SIZE = 32
NUM_EXAMPLES = len(x_train_tfcirc)

x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

qnn_history = model.fit(
      x_train_tfcirc_sub, y_train_hinge_sub,
      batch_size=32,
      epochs=EPOCHS,
      verbose=1,
      validation_data=(x_test_tfcirc, y_test_hinge))

qnn_results = model.evaluate(x_test_tfcirc, y_test)
print("Full Dataset - Training")
print(qnn_results)


