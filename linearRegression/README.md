# Linear Regression from Scratch

## Key Concepts 

### 1. Linear Model
The fundamental equation: `y = mx + b`
- `m` is the slope (weight)
- `b` is the y-intercept (bias)
- `x` is the input feature
- `y` is the predicted output

### 2. Cost Function (Mean Squared Error)
```
J(θ) = (1/2m) * Σ(h(x) - y)²
```
Where:
- `h(x) = mx + b` (hypothesis function)
- `m` is the number of training examples
- `y` is the actual target value

### 3. Gradient Descent
```
θ = θ - α * ∇J(θ)
```
Where:
- `θ` represents parameters (weights and bias)
- `α` is the learning rate
- `∇J(θ)` is the gradient of the cost function

### 4. Gradients
```
∂J/∂weights = (1/m) * X^T * (predictions - y)
∂J/∂bias = (1/m) * Σ(predictions - y)
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
from main import LinearRegressionFromScratch, generate_sample_data

# Generate sample data
X, y = generate_sample_data(n_samples=100, n_features=1, noise=5)

# Create and train model
model = LinearRegressionFromScratch(learning_rate=0.01, max_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Get model parameters
params = model.get_parameters()
print(f"Model: y = {params['weights'][0]:.4f} * x + {params['bias']:.4f}")
```

## Customization

You can modify various parameters:
- `learning_rate`: Controls training speed and stability
- `max_iterations`: Maximum training iterations
- `tolerance`: Convergence threshold
- Data generation parameters (noise, sample size, etc.)

## Algorithm Steps

1. **Initialize** weights and bias randomly
2. **For each iteration**:
   - Compute predictions: `h(x) = X * weights + bias`
   - Compute cost: `J(θ) = (1/2m) * Σ(h(x) - y)²`
   - Compute gradients: `∇J(θ)`
   - Update parameters: `θ = θ - α * ∇J(θ)`
3. **Repeat** until convergence

