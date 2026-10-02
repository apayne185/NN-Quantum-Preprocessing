# Original TensorFlow Quantum implementation

This is the code that accompanied the paper in [`docs/ResearchPaper.pdf`](../../docs/ResearchPaper.pdf).
It follows the [TensorFlow Quantum MNIST tutorial](https://www.tensorflow.org/quantum/tutorials/mnist),
which in turn implements the QNN of [Farhi & Neven (2018)](https://arxiv.org/abs/1802.06002).
It is kept for reference and reproducibility. The maintained implementation is the
PyTorch package in [`src/qnnbench`](../../src/qnnbench).

## Fixes applied after the paper

| Problem | Effect | Fix |
|---|---|---|
| `qnn_train.py` imported `y_test_hinge` from the wrong module | `ImportError`, so the pipeline could not run | Import from `model.qnn` |
| The short and full QNN runs shared one Keras model | The "full" QNN was warm-started from the short run, so it trained for 6 epochs, not 3 | `build_qnn()` gives each run fresh weights |
| The MLP used `metrics=['accuracy']` on logits | Accuracy was thresholded at 0.5 instead of 0, so logits in (0, 0.5) were counted as the wrong class | `BinaryAccuracy(threshold=0.0)` |
| Comparison plots used hand-copied numbers | Plots titled "Testing" showed training accuracy | Plot `val_*` curves from the `History` objects |
| Hinge and cross-entropy losses on one axis | Different loss functions can't be compared by value | Separate axes |
| Plots and IPython `display()` ran at import | Every import needed IPython and drew figures | Moved behind `__main__` guards |
| `requirements.txt` listed standard-library modules | `pip install -r` failed | Pinned the working stack |

## Results after the fixes

One run of `python -m analysis.nn_qnn_compare` (3 epochs each, unseeded, CPU):

| Model | Test accuracy |
|---|---|
| Short QNN (500 examples) | 49.3% |
| Full QNN | 75.1% |
| Fair NN (37 parameters) | 91.7% |

The paper reports 90.6% for the full QNN and 82.7% for the NN. The difference comes
from the warm start and the accuracy-threshold bug above. These are single unseeded
runs; see the main README for multi-seed results from the PyTorch reimplementation.

## Running it

TFQ 0.7.3 only supports Python 3.9-3.11. From this directory:

```bash
uv venv --python 3.11 .venv-tfq
uv pip install --python .venv-tfq/bin/python -r requirements.txt
.venv-tfq/bin/python -m model.nn                  # fair NN only (seconds)
.venv-tfq/bin/python -m analysis.nn_qnn_compare   # everything (~20 min on an 8-thread CPU)
```
