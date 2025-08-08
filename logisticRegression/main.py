"""
Logistic Regression from Scratch :D

Key Concepts:
1. Logistic regression is a classification algorithm that uses a sigmoid function
2. Uses gradient descent to minimize the cost function (Log Loss)
3. The model: h(x) = 1 / (1 + e^(-z)) where z = wx + b
4. Cost function: J(θ) = -(1/m) * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
5. Gradient descent updates: θ = θ - α * ∇J(θ)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LogisticRegressionFromScratch:
    """
    This class has:
    - Sigmoid activation function
    - Cost function (Log Loss)
    - Gradient descent optimization
    - Prediction functionality
    - Training history tracking
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Initialize the logistic regression model.
        
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
        
    def _sigmoid(self, z):
        # Compute the sigmoid activation function.
        # Sigmoid function: σ(z) = 1 / (1 + e^(-z))
        # This function maps any real number to a value between 0 and 1.
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, X, y, weights, bias):
        # Compute the Log Loss cost function.
        # Cost function: J(θ) = -(1/m) * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
        # where h(x) = sigmoid(X * weights + bias)
        m = X.shape[0]
        z = np.dot(X, weights) + bias
        h = self._sigmoid(z)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        
        cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost
    
    def _compute_gradients(self, X, y, weights, bias):
        # Compute gradients for gradient descent.
        # Gradients:
        # ∂J/∂weights = (1/m) * X^T * (h(x) - y)
        # ∂J/∂bias = (1/m) * Σ(h(x) - y)
        m = X.shape[0]
        z = np.dot(X, weights) + bias
        h = self._sigmoid(z)
        
        # Compute gradients
        dw = (1 / m) * np.dot(X.T, (h - y))
        db = (1 / m) * np.sum(h - y)
        
        return dw, db
    
    def fit(self, X, y, verbose=True):
        """
        We will train the logistic regression model using gradient descent.
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
        
        # Ensure y is binary (0 or 1)
        unique_values = np.unique(y)
        if len(unique_values) != 2:
            raise ValueError("Logistic regression requires binary classification (2 classes)")
        
        # Initialize parameters
        self._initialize_parameters(X.shape[1])
        
        # Training history 
        self.cost_history = []
        self.weights_history = []
        
        if verbose:
            print("Starting Logistic Regression Training...")
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
    
    def predict_proba(self, X):
        """
        Now, we need to predict probabilities using the trained model.
        
        Parameters:
        -----------
        X : array-like
            Features to predict on
            
        Returns:
        --------
        probabilities : array
            Predicted probabilities (between 0 and 1)
        """
        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Now, we need to make binary predictions using the trained model.
        
        Parameters:
        -----------
        X : array-like
            Features to predict on
        threshold : float
            Decision threshold (default: 0.5)
            
        Returns:
        --------
        predictions : array
            Binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def get_parameters(self):
        # Get the trained parameters.
        return {
            'weights': self.weights,
            'bias': self.bias
        }

def generate_sample_data(n_samples=200, n_features=2, n_classes=2, noise=0.1, random_state=42):
    """
    You will pass in your dataset, but here i'll be generating sample data for demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of data points (rows)
    n_features : int
        Number of features (columns)
    n_classes : int
        Number of classes (should be 2 for logistic regression)
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
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        n_redundant=0,
        n_informative=n_features,
        random_state=random_state,
        class_sep=1.0,
        scale=1.0
    )
    return X, y

def plot_training_progress(model, X, y):
    # Plot the training progress showing cost function convergence.
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot cost function convergence
    ax1.plot(model.cost_history)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost (Log Loss)')
    ax1.set_title('Cost Function Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Plot decision boundary
    y_pred = model.predict(X)
    ax2.scatter(X[y == 0][:, 0], X[y == 0][:, 1], alpha=0.6, label='Class 0', s=50)
    ax2.scatter(X[y == 1][:, 0], X[y == 1][:, 1], alpha=0.6, label='Class 1', s=50)
    
    # Create decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax2.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('Logistic Regression Decision Boundary')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logistic_regression_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training progress plot saved as 'logistic_regression_training_progress.png'")

def plot_sigmoid_function():
    # Plot the sigmoid function to demonstrate the activation function.
    
    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-x))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=3, color='blue')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold (0.5)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=1, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.xlabel('Input (z)')
    plt.ylabel('Sigmoid Output σ(z)')
    plt.title('Sigmoid Activation Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-10, 10)
    plt.ylim(-0.1, 1.1)
    
    plt.savefig('sigmoid_function.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Sigmoid function plot saved as 'sigmoid_function.png'")


def main():
    # Main function to demonstrate logistic regression from scratch.
    
    print("LOGISTIC REGRESSION FROM SCRATCH")
    print("="*50)
        
    # Plot sigmoid function
    print("\n" + "="*50)
    print("CREATING SIGMOID FUNCTION VISUALIZATION")
    print("="*50)
    plot_sigmoid_function()
    
    # Generate sample data
    print("\n" + "="*50)
    print("GENERATING SAMPLE DATA")
    print("="*50)
    
    X, y = generate_sample_data(n_samples=200, n_features=2, n_classes=2, noise=0.1)
    print(f"Generated {X.shape[0]} samples with {X.shape[1]} feature(s)")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create and train model
    print("\n" + "="*50)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*50)
    
    model = LogisticRegressionFromScratch(learning_rate=0.1, max_iterations=1000)
    model.fit(X_train, y_train, verbose=True)
    
    # Make predictions
    print("\n" + "="*50)
    print("MAKING PREDICTIONS")
    print("="*50)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix (Test Set):")
    print(cm)
    
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
    
    # Show final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    print(f"Model successfully trained!")
    print(f"Final cost: {model.cost_history[-1]:.6f}")
    print(f"Number of iterations: {len(model.cost_history)}")
    print(f"Decision boundary equation: {params['weights'][0]:.4f}*x1 + {params['weights'][1]:.4f}*x2 + {params['bias']:.4f} = 0")
    
    # Demonstrate prediction
    sample_input = np.array([[1.0, 2.0]])
    probability = model.predict_proba(sample_input)[0]
    prediction = model.predict(sample_input)[0]
    print(f"\nSample prediction:")
    print(f"Input: x1 = {sample_input[0][0]}, x2 = {sample_input[0][1]}")
    print(f"Probability: {probability:.4f}")
    print(f"Prediction: Class {prediction}")

if __name__ == "__main__":
    main()
