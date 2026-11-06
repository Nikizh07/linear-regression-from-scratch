# Linear Regression from Scratch

A minimal implementation of simple linear regression trained with batch gradient descent, written using only NumPy. No scikit-learn, no autograd. Every step of the training loop (prediction, loss, gradients, parameter updates) is implemented by hand.

The model learns to predict a student's exam **score** from the number of **hours studied**, using a synthetic dataset with a known ground-truth relationship, so you can check whether the learned parameters match.

## Results

| Parameter | Ground truth | Learned |
|-----------|-------------:|--------:|
| Weight `w` | 7.00 | 6.83 |
| Bias `b`   | 15.00 | 16.63 |

Final training MSE: **~59.9**, close to the injected noise variance (σ² = 64), meaning the model has captured most of the signal it can.

Example: a student who studies **10 hours** is predicted to score **~85**.

## Project Structure

```
.
├── dataset.py           # Generates the synthetic dataset
├── student_scores.csv   # Generated dataset (1,000 samples)
├── model.ipynb          # Model implementation, training, and visualization
└── README.md
```

## The Dataset

`dataset.py` generates 1,000 samples from the linear relationship:

```
Score = 15 + 7 × Hours + ε,    ε ~ N(0, 8²)
```

- `Hours` is sampled uniformly from [0, 12]
- `Score` is clipped to [0, 100] and rounded to two decimals
- A fixed random seed (`42`) keeps the data reproducible

## How It Works

The model is a straight line:

$$\hat{y} = wX + b$$

**Loss: Mean Squared Error**

$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Gradients**

$$\frac{\partial L}{\partial w} = -\frac{2}{n}\sum_{i=1}^{n}X_i(y_i - \hat{y}_i) \qquad \frac{\partial L}{\partial b} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)$$

**Update rule**

$$w \leftarrow w - \alpha \frac{\partial L}{\partial w} \qquad b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$

These map directly to the functions in `model.ipynb`:

| Function | Purpose |
|----------|---------|
| `predict(X, w, b)` | Computes predictions `ŷ = wX + b` |
| `compute_loss(y_true, y_pred)` | Mean squared error |
| `compute_gradients(X, y_true, y_pred)` | Partial derivatives of the loss with respect to `w` and `b` |
| `update_parameters(w, b, dw, db, learning_rate)` | One gradient descent step |
| `train(X, y_true, w, b, epochs, learning_rate)` | Runs the full training loop and logs the loss per epoch |

Training uses initial values `w = 4`, `b = 5`, a learning rate of `0.01`, and `1000` epochs.

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy, pandas, Matplotlib, Jupyter

### Installation

```bash
git clone https://github.com/Nikizh07/linear-regression-from-scratch.git
```

```bash
cd linear-regression-from-scratch
```

```bash
python -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install numpy pandas matplotlib jupyter
```

### Usage

Regenerate the dataset (optional; `student_scores.csv` is already included):

```bash
python dataset.py
```

Open the notebook and run all cells:

```bash
jupyter notebook model.ipynb
```

The notebook will:

1. Load and plot the dataset
2. Train the model with gradient descent, printing the loss each epoch
3. Plot the fitted regression line over the data
4. Make a sample prediction

## Possible Extensions

- Train/test split and evaluation on held-out data
- Feature scaling for faster convergence
- Plotting the loss curve over epochs
- Comparing against the closed-form solution (normal equation)
- Generalizing to multiple features (multivariate linear regression)

## License

This project is open source and available for educational use.
