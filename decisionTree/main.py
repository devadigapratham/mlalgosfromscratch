"""
Decision Tree from Scratch :D
==========================

Key Concepts:
1. Decision trees use recursive partitioning to split data based on features
2. Uses information gain or Gini impurity to determine best splits
3. The model: Tree structure with decision nodes and leaf nodes
4. Splitting criteria: Information Gain = Entropy(parent) - Weighted_Entropy(children)
5. Recursive algorithm: Build tree by finding best splits and repeating

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from collections import Counter

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DecisionTreeFromScratch:
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='entropy'):
        """
        Initialize the decision tree model.
        
        Parameters:
        -----------
        max_depth : int, optional
            Maximum depth of the tree
        min_samples_split : int
            Minimum samples required to split a node
        min_samples_leaf : int
            Minimum samples required in a leaf node
        criterion : str
            Splitting criterion ('entropy' or 'gini')
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None
        self.feature_names = None
        
    def _entropy(self, y):
        """
        Calculate entropy for a given set of labels.
        
        Entropy: H(S) = -Σ(p_i * log2(p_i))
        where p_i is the proportion of class i in set S
        """
        if len(y) == 0:
            return 0
        
        # Count occurrences of each class
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy
    
    def _gini_impurity(self, y):
        """
        Calculate Gini impurity for a given set of labels.
        
        Gini: G(S) = 1 - Σ(p_i²)
        where p_i is the proportion of class i in set S
        """
        if len(y) == 0:
            return 0
        
        # Count occurrences of each class
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        
        # Calculate Gini impurity
        gini = 1 - sum(p**2 for p in probabilities)
        return gini
    
    def _information_gain(self, parent, left_child, right_child, criterion='entropy'):
        """
        Calculate information gain for a split.
        
        Information Gain = Impurity(parent) - Weighted_Impurity(children)
        """
        if criterion == 'entropy':
            parent_impurity = self._entropy(parent)
            left_impurity = self._entropy(left_child)
            right_impurity = self._entropy(right_child)
        else:  # gini
            parent_impurity = self._gini_impurity(parent)
            left_impurity = self._gini_impurity(left_child)
            right_impurity = self._gini_impurity(right_child)
        
        # Calculate weighted impurity of children
        n_left = len(left_child)
        n_right = len(right_child)
        n_total = len(parent)
        
        weighted_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
        
        # Information gain
        information_gain = parent_impurity - weighted_impurity
        return information_gain
    
    def _find_best_split(self, X, y):
        """
        Find the best split for a given dataset.
        
        Returns:
        --------
        best_feature : int
            Index of the best feature to split on
        best_threshold : float
            Best threshold value for the split
        best_gain : float
            Information gain of the best split
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            # Get unique values for this feature
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # Create split
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # Skip if split doesn't meet minimum requirements
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                
                # Calculate information gain
                gain = self._information_gain(y, left_y, right_y, self.criterion)
                
                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _create_leaf(self, y):
        """
        Create a leaf node with the most common class.
        """
        most_common = Counter(y).most_common(1)[0][0]
        return {'type': 'leaf', 'prediction': most_common, 'samples': len(y)}
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            return self._create_leaf(y)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no good split found, create leaf
        if best_gain <= 0:
            return self._create_leaf(y)
        
        # Create split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]
        
        # Create decision node
        node = {
            'type': 'decision',
            'feature': best_feature,
            'threshold': best_threshold,
            'gain': best_gain,
            'samples': n_samples,
            'left': self._build_tree(left_X, left_y, depth + 1),
            'right': self._build_tree(right_X, right_y, depth + 1)
        }
        
        return node
    
    def fit(self, X, y, feature_names=None):
        """
        Train the decision tree model.
        
        Parameters:
        -----------
        X : array-like
            Training features (n_samples, n_features)
        y : array-like
            Target values (n_samples,)
        feature_names : list, optional
            Names of the features
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        self.feature_names = feature_names if feature_names else [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Build the tree
        self.tree = self._build_tree(X, y)
        
        print("Decision Tree Training Completed!")
        print(f"Tree depth: {self._get_tree_depth(self.tree)}")
        print(f"Number of nodes: {self._count_nodes(self.tree)}")
        
        return self
    
    def _predict_single(self, x, node):
        """
        Make a prediction for a single sample.
        """
        if node['type'] == 'leaf':
            return node['prediction']
        
        # Decision node
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
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
        predictions = []
        
        for x in X:
            pred = self._predict_single(x, self.tree)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _get_tree_depth(self, node):
        """Get the depth of the tree."""
        if node['type'] == 'leaf':
            return 0
        
        left_depth = self._get_tree_depth(node['left'])
        right_depth = self._get_tree_depth(node['right'])
        
        return max(left_depth, right_depth) + 1
    
    def _count_nodes(self, node):
        """Count the total number of nodes in the tree."""
        if node['type'] == 'leaf':
            return 1
        
        return 1 + self._count_nodes(node['left']) + self._count_nodes(node['right'])

def generate_sample_data(n_samples=300, n_features=2, n_classes=3, random_state=42):
    """
    Generate sample data for demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of data points
    n_features : int
        Number of features
    n_classes : int
        Number of classes
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X : array
        Features
    y : array
        Target values
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        n_redundant=0,
        n_informative=n_features,
        random_state=random_state,
        class_sep=1.5,
        scale=1.0
    )
    return X, y

def plot_decision_boundary(model, X, y, feature_names=None):
    """
    Plot the decision boundary of the decision tree.
    """
    if X.shape[1] != 2:
        print("Decision boundary plot only works for 2D data")
        return
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Make predictions on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    # Plot data points
    unique_classes = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
    
    for i, class_label in enumerate(unique_classes):
        mask = y == class_label
        plt.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], 
                   label=f'Class {class_label}', alpha=0.7, s=50)
    
    # Labels and title
    feature_names = feature_names if feature_names else ['Feature 1', 'Feature 2']
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Decision Tree Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('decision_tree_boundary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Decision boundary plot saved as 'decision_tree_boundary.png'")

def main():
    """
    Main function to demonstrate decision tree from scratch.
    """
    print("DECISION TREE FROM SCRATCH")
    print("="*50)
        
    # Generate sample data
    print("\n" + "="*50)
    print("GENERATING SAMPLE DATA")
    print("="*50)
    
    X, y = generate_sample_data(n_samples=300, n_features=2, n_classes=3)
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
    print("TRAINING DECISION TREE MODEL")
    print("="*50)
    
    feature_names = ['Feature 1', 'Feature 2']
    model = DecisionTreeFromScratch(max_depth=5, min_samples_split=10, criterion='entropy')
    model.fit(X_train, y_train, feature_names=feature_names)
    
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
    
    # Visualizations
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Plot decision boundary
    plot_decision_boundary(model, X_train, y_train, feature_names)
    
    # Show final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    print(f"Model successfully trained!")
    print(f"Tree depth: {model._get_tree_depth(model.tree)}")
    print(f"Number of nodes: {model._count_nodes(model.tree)}")
    
    # Demonstrate prediction
    sample_input = np.array([[1.0, 2.0]])
    prediction = model.predict(sample_input)[0]
    print(f"\nSample prediction:")
    print(f"Input: Feature 1 = {sample_input[0][0]}, Feature 2 = {sample_input[0][1]}")
    print(f"Prediction: Class {prediction}")

if __name__ == "__main__":
    main()