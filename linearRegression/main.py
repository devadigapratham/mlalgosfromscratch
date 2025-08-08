"""
Linear Regression from Scratch :D

Key Concepts:
1. Linear regression finds the best-fit line through data points
2. Uses gradient descent to minimize the cost function (Mean Squared Error)
3. The model: y = mx + b (where m is slope, b is y-intercept)
4. Cost function: J(θ) = (1/2m) * Σ(h(x) - y)²
5. Gradient descent updates: θ = θ - α * ∇J(θ)

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LinearRegressionFromScratch:
    """    
    This class has:
    - Cost function (Mean Squared Error)
    - Gradient descent optimization
    - Prediction functionality
    - Training history tracking
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Initialize the linear regression model.
        
        Parameters that has to be passed:
        1. learning_rate : float
            Step size for gradient descent (alpha)
        2. max_iterations : int
            Maximum number of training iterations
        3. tolerance : float
            Convergence tolerance for early stopping
        """

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.weights_history = []
        
    def _initialize_parameters(self, n_features):
        #Initialize weights and bias randomly! 
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
    def _compute_cost(self, X, y, weights, bias):
        # Compute the Mean Squared Error cost function.
        # Cost function: J(θ) = (1/2m) * Σ(h(x) - y)²
        # where h(x) = X * weights + bias
        # m here, is the number of training examples!
        m = X.shape[0]
        predictions = np.dot(X, weights) + bias
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost
    
    def _compute_gradients(self, X, y, weights, bias):
        # Compute gradients for gradient descent.
        # Gradients:
        # ∂J/∂weights = (1/m) * X^T * (predictions - y)
        # ∂J/∂bias = (1/m) * Σ(predictions - y)
        m = X.shape[0]
        predictions = np.dot(X, weights) + bias        
        # Compute gradients
        dw = (1 / m) * np.dot(X.T, (predictions - y))
        db = (1 / m) * np.sum(predictions - y)
        
        return dw, db
    
    def fit(self, X, y, verbose=True):
        """
        We will train the linear regression model using gradient descent.
        Parameters:
        1. X : array-like
            Training features (n_samples, n_features)
        2. y : array-like
            Target values (n_samples,)
        3. verbose : bool
            Whether to print training progress
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y).flatten()

        # Initialize parameters
        self._initialize_parameters(X.shape[1])
        self.cost_history = []
        self.weights_history = []

        # not compulsory lol, but it's just for understanding :) 
        if verbose:
            print("Starting Linear Regression Training...")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Max iterations: {self.max_iterations}")
            print("-" * 50)
        
        for iteration in range(self.max_iterations):
            # Compute cost
            cost = self._compute_cost(X, y, self.weights, self.bias)
            self.cost_history.append(cost)
            self.weights_history.append(self.weights.copy())
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y, self.weights, self.bias)
            
            # Update parameters (gradient descent step)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress every 100 iterations
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration:4d} | Cost: {cost:.6f}")
            
            # Early stopping if cost doesn't change significantly
            if len(self.cost_history) > 1:
                if abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break
        
        if verbose:
            print("-" * 50)
            print(f"Training completed!")
            print(f"Final cost: {self.cost_history[-1]:.6f}")
            print(f"Final weights: {self.weights}")
            print(f"Final bias: {self.bias:.6f}")
        
        return self
    
    def predict(self, X):
        """
        Now, we need to make predictions using the trained model.
        
        Parameters:
        -----------
        X : array-like
            Features to predict on
            
        Returns:
        --------
        predictions : array
            Predicted values
        """
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias
    
    def get_parameters(self):
        return {
            'weights': self.weights,
            'bias': self.bias
        }

def generate_sample_data(n_samples=100, n_features=1, noise=10, random_state=42):
    """
    You will pass in your dataset, but here i'll be generating sample data for demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of data points (rows)
    n_features : int
        Number of features (columns)
    noise : float
        Amount of noise to add (randomness)
    random_state : int
        Random seed for reproducibility (for consistency)
        
    Returns:
    --------
    X : array
        Features (independent variables)
    y : array
        Target values (dependent variable)
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    return X, y

def plot_training_progress(model, X, y):
    """
    Plot the training progress showing cost function convergence.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot cost function convergence
    ax1.plot(model.cost_history)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost (MSE)')
    ax1.set_title('Cost Function Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Plot final predictions vs actual
    y_pred = model.predict(X)
    ax2.scatter(X, y, alpha=0.6, label='Actual Data')
    ax2.plot(X, y_pred, color='red', linewidth=2, label='Predicted Line')
    ax2.set_xlabel('X')
    ax2.set_ylabel('y')
    ax2.set_title('Linear Regression Fit')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training progress plot saved as 'linear_regression_training_progress.png'")

def plot_3d_training_visualization(model, X, y):
    """
    Create a 3D visualization showing the cost surface and training path.
    """
    if X.shape[1] != 1:
        print("3D visualization only works for single feature data")
        return
    
    # Create a grid of weight and bias values
    w_range = np.linspace(-2, 2, 50)
    b_range = np.linspace(-2, 2, 50)
    W, B = np.meshgrid(w_range, b_range)
    
    # Compute cost for each combination
    costs = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            costs[i, j] = model._compute_cost(X, y, np.array([W[i, j]]), B[i, j])
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot cost surface
    surface = ax.plot_surface(W, B, costs, alpha=0.6, cmap='viridis')
    
    # Plot training path
    if len(model.weights_history) > 0:
        weights_path = [w[0] for w in model.weights_history]
        bias_path = [0] * len(weights_path)  # Assuming bias starts at 0
        cost_path = model.cost_history
        
        ax.plot(weights_path, bias_path, cost_path, 'r-', linewidth=3, label='Training Path')
        ax.scatter(weights_path[0], bias_path[0], cost_path[0], c='green', s=100, label='Start')
        ax.scatter(weights_path[-1], bias_path[-1], cost_path[-1], c='red', s=100, label='End')
    
    ax.set_xlabel('Weight')
    ax.set_ylabel('Bias')
    ax.set_zlabel('Cost')
    ax.set_title('3D Cost Surface and Training Path')
    ax.legend()
    
    plt.savefig('linear_regression_3d_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("3D visualization plot saved as 'linear_regression_3d_visualization.png'")

def main():
    # Main function to demonstrate linear regression from scratch.
    
    print("LINEAR REGRESSION FROM SCRATCH")
    print("="*50)
        
    # Generate sample data
    print("\n" + "="*50)
    print("GENERATING SAMPLE DATA")
    print("="*50)
    
    X, y = generate_sample_data(n_samples=100, n_features=1, noise=5)
    print(f"Generated {X.shape[0]} samples with {X.shape[1]} feature(s)")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create and train model
    print("\n" + "="*50)
    print("TRAINING LINEAR REGRESSION MODEL")
    print("="*50)
    
    model = LinearRegressionFromScratch(learning_rate=0.01, max_iterations=1000)
    model.fit(X_train, y_train, verbose=True)
    
    # Make predictions
    print("\n" + "="*50)
    print("MAKING PREDICTIONS")
    print("="*50)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Get final parameters
    params = model.get_parameters()
    print(f"\nFinal Parameters:")
    print(f"Weights: {params['weights']}")
    print(f"Bias: {params['bias']:.4f}")
    
    # Visualizations
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Plot training progress
    plot_training_progress(model, X_train, y_train)
    
    # 3D visualization (for single feature data)
    if X_train.shape[1] == 1:
        plot_3d_training_visualization(model, X_train, y_train)
    
    # Show final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    print(f"Model successfully trained!")
    print(f"Final cost: {model.cost_history[-1]:.6f}")
    print(f"Number of iterations: {len(model.cost_history)}")
    print(f"Model equation: y = {params['weights'][0]:.4f} * x + {params['bias']:.4f}")
    
    # Demonstrate prediction
    sample_input = np.array([[2.0]])
    prediction = model.predict(sample_input)
    print(f"\nSample prediction:")
    print(f"Input: x = {sample_input[0][0]}")
    print(f"Prediction: y = {prediction[0]:.4f}")

if __name__ == "__main__":
    main()
