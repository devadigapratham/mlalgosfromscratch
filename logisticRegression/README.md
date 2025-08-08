# Logistic Regression from Scratch

## Key Concepts 

### 1. Logistic Model
The fundamental equation: `h(x) = σ(z) = 1 / (1 + e^(-z))` where `z = wx + b`
- `w` is the weight vector
- `b` is the bias term
- `x` is the input feature
- `σ(z)` is the sigmoid function
- `h(x)` is the predicted probability

### 2. Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))
```
Properties:
- Output range: [0, 1]
- Symmetric around (0, 0.5)
- Smooth and differentiable
- Maps any real number to probability

### 3. Cost Function (Log Loss)
```
J(θ) = -(1/m) * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
```
Where:
- `h(x) = sigmoid(wx + b)` (hypothesis function)
- `m` is the number of training examples
- `y` is the actual target value (0 or 1)
- This is also called Binary Cross-Entropy Loss

### 4. Gradient Descent
```
θ = θ - α * ∇J(θ)
```
Where:
- `θ` represents parameters (weights and bias)
- `α` is the learning rate
- `∇J(θ)` is the gradient of the cost function

### 5. Gradients
```
∂J/∂weights = (1/m) * X^T * (h(x) - y)
∂J/∂bias = (1/m) * Σ(h(x) - y)
```
Note: The gradient form is similar to linear regression!

### 6. Decision Boundary
```
h(x) = 0.5 when z = 0
This means: wx + b = 0
For 2D: w1*x1 + w2*x2 + b = 0
This defines a linear decision boundary
```

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Implementation
```bash
python main.py
```

## Usage Example

```python
from main import LogisticRegressionFromScratch, generate_sample_data

# Generate sample data
X, y = generate_sample_data(n_samples=200, n_features=2, n_classes=2, noise=0.1)

# Create and train model
model = LogisticRegressionFromScratch(learning_rate=0.1, max_iterations=1000)
model.fit(X, y)

# Make predictions
probabilities = model.predict_proba(X)
predictions = model.predict(X)

# Get model parameters
params = model.get_parameters()
print(f"Decision boundary: {params['weights'][0]:.4f}*x1 + {params['weights'][1]:.4f}*x2 + {params['bias']:.4f} = 0")
```


## Customization

You can modify various parameters:
- `learning_rate`: Controls training speed and stability
- `max_iterations`: Maximum training iterations
- `tolerance`: Convergence threshold
- `threshold`: Decision threshold for classification
- Data generation parameters (noise, sample size, etc.)

## Algorithm Steps

1. **Initialize** weights and bias randomly
2. **For each iteration**:
   - Compute z = X * weights + bias
   - Compute h(x) = sigmoid(z)
   - Compute cost: J(θ) = log loss
   - Compute gradients: ∇J(θ)
   - Update parameters: θ = θ - α * ∇J(θ)
3. **Repeat** until convergence

## Differences from Linear Regression

- **Activation Function**: Uses sigmoid instead of linear output
- **Cost Function**: Uses log loss instead of MSE
- **Output**: Probabilities instead of continuous values
- **Application**: Classification instead of regression
- **Decision Boundary**: Linear boundary in feature space

