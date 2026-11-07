# Smart City UAV for Pothole Detection

COS40007 Artificial Intelligence Engineering, Group 1 (Le Mai Chi, Hoang Huu Hoan, Tran Minh Hai).

## Repository layout

```
model.py                      TensorFlow/Keras linear regression: data, training, evaluation, plots, saved model
requirements.txt              Python dependencies: tensorflow, numpy, matplotlib (no versions pinned)
.github/workflows/train.yml   CI workflow that installs the dependencies and runs model.py
```

## What `model.py` does

1. Builds a synthetic dataset of 50 points on the line `y = x + 10`, with `x` from -100 to 96 in steps of 4.
2. Splits the points in order: the first 40 for training, the last 10 (`x` from 60 to 96) for testing.
   The split is not shuffled, so the test set checks how well the model extrapolates past the training range.
3. Builds a model with one `Dense(1)` layer, MAE loss and the SGD optimizer. The random seed is 42.
4. Trains for 100 epochs. Keras keeps the last 20% of the training points (8 points) for validation.
5. Predicts on the test set and computes MAE and MSE with NumPy.
6. Prints the learned slope and intercept next to the expected `y = x + 10`.

## Outputs

`model.py` writes these files to the current working directory:

| File | Contents |
|---|---|
| `model_results.png` | training data, test data and test predictions on one scatter plot |
| `training_history.png` | training and validation loss per epoch, plus a second plot from epoch 10 onwards |
| `metrics.txt` | test MAE and MSE |
| `linear_regression_model.h5` | the trained model in Keras HDF5 format |

The repository has no `.gitignore`, so after a local run git lists these files as untracked. Do not commit them.

## Run locally

You need Python 3.11, the same version CI uses.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python model.py
```

`model.py` calls `plt.show()` twice. On a machine with a display, each plot opens in a window and the
script waits until you close it.

## CI

`.github/workflows/train.yml` ("Train ML Model") runs on every push to `main`, on every pull request to
`main`, and on demand from the Actions tab (`workflow_dispatch`). On an `ubuntu-latest` runner it:

1. checks out the code and sets up Python 3.11
2. restores the pip cache, keyed on the hash of `requirements.txt`
3. runs `pip install -r requirements.txt`
4. runs `python model.py`
5. uploads `model_results.png` and `metrics.txt` as the `model-results` artifact

The runner has no display, so `plt.show()` does nothing there and the plots are only saved to file.
CI also creates `training_history.png` and `linear_regression_model.h5`, but it does not upload them.
To keep them, add them to the `path` list of the "Upload results" step.

## Workflow

Make one branch per Jira issue, named after its key (for example `SCRUM-38-train-pipeline`), and merge
it into `main` through a pull request.
