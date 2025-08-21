# Decision Tree from Scratch

This repository contains a comprehensive implementation of decision trees from scratch, designed to help people understand the fundamental concepts behind this foundational classification and regression algorithm.

## Key Concepts 

### 1. Decision Tree Model
The fundamental structure: Tree with decision nodes and leaf nodes
- **Decision Nodes**: Split data based on feature thresholds
- **Leaf Nodes**: Make final predictions
- **Tree Structure**: Hierarchical organization of decisions

### 2. Entropy
```
H(S) = -Σ(p_i * log2(p_i))
```
Where:
- `p_i` is the proportion of class i in set S
- Measures uncertainty/impurity in a dataset
- Higher entropy = more uncertainty
- Lower entropy = less uncertainty

### 3. Gini Impurity
```
G(S) = 1 - Σ(p_i²)
```
Where:
- `p_i` is the proportion of class i in set S
- Alternative measure of impurity
- Ranges from 0 (pure) to 1-1/k (k classes)
- Often computationally faster than entropy

### 4. Information Gain
```
IG(S, A) = H(S) - Σ(|S_v|/|S| * H(S_v))
```
Where:
- `H(S)` is the entropy of the parent set
- `S_v` is the subset of S for value v of attribute A
- Measures how much a split reduces uncertainty
- Higher information gain = better split

### 5. Splitting Criteria
- Find feature and threshold that maximize information gain
- Information gain = Parent impurity - Weighted child impurity
- Weighted by proportion of samples in each child

### 6. Stopping Criteria
- Maximum depth reached
- Minimum samples for split not met
- All samples belong to same class
- No valid split found

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
from main import DecisionTreeFromScratch, generate_sample_data

# Generate sample data
X, y = generate_sample_data(n_samples=300, n_features=2, n_classes=3)

# Create and train model
model = DecisionTreeFromScratch(max_depth=5, min_samples_split=10, criterion='entropy')
model.fit(X, y, feature_names=['Feature 1', 'Feature 2'])

# Make predictions
predictions = model.predict(X)

# Print tree structure
model.print_tree()
```

## Key Insights

- **Entropy vs Gini**: Both measure impurity, Gini is computationally faster
- **Information Gain**: Always positive, higher is better
- **Tree Depth**: Controls model complexity and overfitting
- **Splitting**: Always binary splits (left/right) for simplicity
- **Recursion**: Natural algorithm for tree building

## Customization

You can modify various parameters:
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples required to split
- `min_samples_leaf`: Minimum samples in leaf nodes
- `criterion`: 'entropy' or 'gini' for splitting

## Algorithm Steps

1. **Start** with root node containing all data
2. **For each feature and threshold**:
   - Split data into left and right subsets
   - Calculate information gain
3. **Choose split** with maximum information gain
4. **Recursively apply** to left and right subsets
5. **Stop** when stopping criteria met

## Differences from Other Algorithms

- **Non-linear**: Can capture complex decision boundaries
- **Interpretable**: Easy to understand decision rules
- **No optimization**: Greedy algorithm, no gradient descent
- **Discrete splits**: Works well with categorical and numerical data
- **Tree structure**: Hierarchical organization of decisions
