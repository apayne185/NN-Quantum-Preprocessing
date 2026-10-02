from model.qnn import y_train_hinge, y_test_hinge, build_qnn
from model.data_preprocess import x_train_tfcirc, x_test_tfcirc



EPOCHS = 3
BATCH_SIZE = 32
NUM_EXAMPLES = 500

x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]



short_model = build_qnn()
short_qnn_history = short_model.fit(
      x_train_tfcirc_sub, y_train_hinge_sub,
      batch_size=BATCH_SIZE,
      epochs=EPOCHS,
      verbose=1,
      validation_data=(x_test_tfcirc, y_test_hinge))

short_qnn_results = short_model.evaluate(x_test_tfcirc, y_test_hinge)
print("Short (Partial Dataset -500 examples) - Training")
print(short_qnn_results)


EPOCHS = 3
BATCH_SIZE = 32
NUM_EXAMPLES = len(x_train_tfcirc)

x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

# Fresh weights: the full run must not inherit the short run's training.
full_model = build_qnn()
qnn_history = full_model.fit(
      x_train_tfcirc_sub, y_train_hinge_sub,
      batch_size=BATCH_SIZE,
      epochs=EPOCHS,
      verbose=1,
      validation_data=(x_test_tfcirc, y_test_hinge))

qnn_results = full_model.evaluate(x_test_tfcirc, y_test_hinge)
print("Full Dataset - Training")
print(qnn_results)
