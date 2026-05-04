"""
PRACTICAL EXAMPLE: Complete Sentiment Classification Pipeline

This file shows how to use all the implemented components together.
You can adapt this code for your actual sentiment classification task.
"""

# ============================================================================
# EXAMPLE 1: Simple Naive Bayes Classification
# ============================================================================

EXAMPLE_1 = """
# Example 1: Naive Bayes on Raw Bag-of-Words
# ============================================

from src import (
    CountVectorizer,
    NaiveBayes,
    train_test_split,
    accuracy,
    confusion_matrix,
    TextProcessor
)
import numpy as np

# Step 1: Prepare your documents and labels
documents = [
    "this movie is really good and entertaining",
    "terrible film waste of time",
    "amazing story great acting",
    "horrible bad movie disappointing",
    "excellent cinematography loved it",
    "worst movie ever",
]

labels = np.array([1, 0, 1, 0, 1, 0])  # 1=positive, 0=negative

# Step 2: Split data (stratified to maintain class distribution)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    np.array(documents),
    labels,
    test_size=0.33,
    random_state=42,
    stratify=True
)

print(f"Train: {len(X_train_text)} docs, Test: {len(X_test_text)} docs")
print(f"Train classes: {np.bincount(y_train)}")
print(f"Test classes: {np.bincount(y_test)}")

# Step 3: Convert text to count vectors (fit ONLY on train!)
vectorizer = CountVectorizer(
    lowercase=True,
    ngram_range=(1, 1),  # Unigrams only
    max_features=1000
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print(f"\\nFeature matrix shape: {X_train.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Train matrix sparsity: {1 - X_train.nnz/(X_train.shape[0]*X_train.shape[1]):.2%}")

# Step 4: Train Naive Bayes
model = NaiveBayes(alpha=1.0)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Step 6: Evaluate
acc = accuracy(y_test, y_pred)
cm, classes = confusion_matrix(y_test, y_pred)

print(f"\\nAccuracy: {acc:.4f}")
print(f"Classes: {classes}")
print(f"Confusion Matrix:\\n{cm}")

# Step 7: Print predictions for inspection
for i in range(len(X_test_text)):
    true_label = "positive" if y_test[i] == 1 else "negative"
    pred_label = "positive" if y_pred[i] == 1 else "negative"
    confidence = max(y_proba[i])
    
    print(f"\\nText: {X_test_text[i]}")
    print(f"True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.2%}")
"""

# ============================================================================
# EXAMPLE 2: Decision Tree with TF-IDF
# ============================================================================

EXAMPLE_2 = """
# Example 2: Decision Tree on TF-IDF Features
# =============================================

from src import (
    CountVectorizer,
    TfidfTransformer,
    DecisionTree,
    train_test_split,
    accuracy,
    confusion_matrix
)
import numpy as np

# Data preparation (same as Example 1)
documents = [
    "this movie is really good and entertaining",
    "terrible film waste of time",
    "amazing story great acting",
    "horrible bad movie disappointing",
    "excellent cinematography loved it",
    "worst movie ever",
]

labels = np.array([1, 0, 1, 0, 1, 0])

# Split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    np.array(documents),
    labels,
    test_size=0.33,
    random_state=42,
    stratify=True
)

# Vectorize: Count → BoW
vectorizer = CountVectorizer(
    lowercase=True,
    ngram_range=(1, 2),  # Unigrams + bigrams
    max_features=500
)

X_train_counts = vectorizer.fit_transform(X_train_text)
X_test_counts = vectorizer.transform(X_test_text)

# Transform: BoW → TF-IDF
tfidf = TfidfTransformer()
X_train = tfidf.fit_transform(X_train_counts)
X_test = tfidf.transform(X_test_counts)

print(f"TF-IDF matrix shape: {X_train.shape}")
print(f"Training matrix sparsity: {1 - X_train.nnz/(X_train.shape[0]*X_train.shape[1]):.2%}")

# Train Decision Tree
tree = DecisionTree(
    min_samples_split=2,
    max_depth=4
)

tree.fit(X_train, y_train)

# Optional: print tree structure
print("\\nTree structure:")
tree.print_tree()

# Predict
y_pred = tree.predict(X_test)

# Evaluate
acc = accuracy(y_test, y_pred)
cm, classes = confusion_matrix(y_test, y_pred)

print(f"\\nAccuracy: {acc:.4f}")
print(f"Confusion Matrix:\\n{cm}")

# Training accuracy (to check for overfitting)
y_pred_train = tree.predict(X_train)
train_acc = accuracy(y_train, y_pred_train)
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {acc:.4f}")

if train_acc > 0.1 and acc < train_acc - 0.2:
    print("WARNING: Possible overfitting. Consider reducing max_depth.")
"""

# ============================================================================
# EXAMPLE 3: Compare Models
# ============================================================================

EXAMPLE_3 = """
# Example 3: Compare Naive Bayes vs Decision Tree
# ================================================

from src import (
    CountVectorizer,
    TfidfTransformer,
    NaiveBayes,
    DecisionTree,
    train_test_split,
    accuracy,
    confusion_matrix
)
import numpy as np

# Setup
documents = [
    "this movie is really good and entertaining",
    "terrible film waste of time",
    "amazing story great acting",
    "horrible bad movie disappointing",
    "excellent cinematography loved it",
    "worst movie ever",
    "good film definitely recommend",
    "bad acting bad plot",
]

labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])

# Split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    np.array(documents),
    labels,
    test_size=0.25,
    random_state=42,
    stratify=True
)

# Prepare data
cv = CountVectorizer(lowercase=True, ngram_range=(1, 1), max_features=500)
X_train_counts = cv.fit_transform(X_train_text)
X_test_counts = cv.transform(X_test_text)

# Model 1: Naive Bayes (uses raw counts)
print("Model 1: Naive Bayes (BoW)")
print("-" * 40)

nb = NaiveBayes(alpha=1.0)
nb.fit(X_train_counts, y_train)
y_pred_nb = nb.predict(X_test_counts)
acc_nb = accuracy(y_test, y_pred_nb)
cm_nb, _ = confusion_matrix(y_test, y_pred_nb)

print(f"Accuracy: {acc_nb:.4f}")
print(f"Confusion Matrix:\\n{cm_nb}")

# Model 2: Decision Tree (uses TF-IDF)
print("\\nModel 2: Decision Tree (TF-IDF)")
print("-" * 40)

tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_counts)
X_test_tfidf = tfidf.transform(X_test_counts)

dt = DecisionTree(max_depth=3)
dt.fit(X_train_tfidf, y_train)
y_pred_dt = dt.predict(X_test_tfidf)
acc_dt = accuracy(y_test, y_pred_dt)
cm_dt, _ = confusion_matrix(y_test, y_pred_dt)

print(f"Accuracy: {acc_dt:.4f}")
print(f"Confusion Matrix:\\n{cm_dt}")

# Comparison
print("\\nComparison")
print("=" * 40)
print(f"Naive Bayes: {acc_nb:.4f}")
print(f"Decision Tree: {acc_dt:.4f}")

if acc_nb > acc_dt:
    print(f"Winner: Naive Bayes (+{(acc_nb - acc_dt):.4f})")
elif acc_dt > acc_nb:
    print(f"Winner: Decision Tree (+{(acc_dt - acc_nb):.4f})")
else:
    print("Tie!")
"""

# ============================================================================
# EXAMPLE 4: Real Data Workflow
# ============================================================================

EXAMPLE_4 = """
# Example 4: Workflow with Real Dataset
# ======================================

from src import (
    CountVectorizer,
    TfidfTransformer,
    NaiveBayes,
    train_test_split,
    accuracy,
    confusion_matrix,
    TextProcessor
)
import numpy as np
import pandas as pd

# Load your dataset (e.g., from CSV)
# df = pd.read_csv('data/raw/imdb/IMDB_dataset.csv')
# X = df['text'].values
# y = (df['sentiment'] == 'positive').astype(int).values

# For this example, create toy data
X = np.array([
    "excellent movie highly recommend",
    "terrible waste of time",
    "amazing actors great plot",
    "boring and slow",
    "best film ever",
    "worst movie i have seen",
    "good story well acted",
    "bad production bad dialogue",
])

y = np.array([1, 0, 1, 0, 1, 0, 1, 0])

print(f"Dataset: {len(X)} documents")
print(f"Classes: {np.bincount(y)}")

# Step 1: Split with stratification
train_ratio = 0.8
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=1 - train_ratio,
    random_state=42,
    stratify=True
)

print(f"\\nTrain: {len(X_train)}, Test: {len(X_test)}")

# Step 2: Vectorize
print("\\nVectorizing...")

vectorizer = CountVectorizer(
    lowercase=True,
    ngram_range=(1, 1),
    max_features=1000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Feature matrix shape: {X_train_vec.shape}")

# Step 3: Optional TF-IDF
print("\\nApplying TF-IDF...")

tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_vec)
X_test_tfidf = tfidf.transform(X_test_vec)

# Step 4: Train model
print("\\nTraining model...")

model = NaiveBayes(alpha=1.0)
model.fit(X_train_vec, y_train)  # Use raw counts for NB

# Step 5: Evaluate
print("\\nEvaluating...")

y_pred = model.predict(X_test_vec)
acc = accuracy(y_test, y_pred)
cm, classes = confusion_matrix(y_test, y_pred)

print(f"\\nResults:")
print(f"Accuracy: {acc:.4f}")
print(f"Classes: {classes}")
print(f"Confusion Matrix:\\n{cm}")

# Step 6: Error analysis
print("\\nMisclassified samples:")
errors = y_test != y_pred
if errors.sum() > 0:
    for i in np.where(errors)[0]:
        true_label = "positive" if y_test[i] == 1 else "negative"
        pred_label = "positive" if y_pred[i] == 1 else "negative"
        print(f"  Text: {X_test[i]}")
        print(f"  True: {true_label}, Predicted: {pred_label}\\n")
"""

# ============================================================================
# KEY TIPS
# ============================================================================

TIPS = """
TIPS FOR USING THE ML PIPELINE
================================

1. Data Preparation
   ✓ Always call train_test_split FIRST (before vectorization)
   ✓ Use stratify=True for classification
   ✓ Check class distribution: np.bincount(y_train)

2. Vectorization
   ✓ Fit CountVectorizer ONLY on training documents
   ✓ Use fit_transform() on train, transform() on test
   ✓ Start with max_features=5000-10000 for real datasets
   ✓ Keep vocabulary manageable (avoid memory issues)

3. Feature Engineering
   ✓ Start with ngram_range=(1,1) (unigrams only)
   ✓ Try ngram_range=(1,2) if needed (adds bigrams)
   ✓ Bigrams help capture context ("not good" vs "good")
   ✓ But they increase feature space, so use max_features

4. Model Selection
   ✓ NaiveBayes: fast, interpretable, good for text
   ✓ DecisionTree: handles interactions, but can overfit
   ✓ For NB: use raw counts from CountVectorizer
   ✓ For DT: use TF-IDF (better features)

5. Hyperparameter Tuning
   ✓ NaiveBayes: mainly alpha (1.0 usually good)
   ✓ DecisionTree: max_depth (prevent overfitting)
   ✓ DecisionTree: min_samples_split (>= 2)
   ✓ Start simple, increase complexity if needed

6. Evaluation
   ✓ Always compute: accuracy + confusion matrix
   ✓ Check train vs test accuracy (overfitting?)
   ✓ Look at false positives and false negatives
   ✓ Adjust model if classes are imbalanced

7. Common Pitfalls
   ✗ DON'T fit vectorizer on test set (data leakage!)
   ✗ DON'T forget to transform test data with learned vocab
   ✗ DON'T use dense matrices for large datasets
   ✗ DON'T skip stratification (class imbalance issues)
   ✗ DON'T tune hyperparameters on test set

8. Performance
   ✓ Sparse matrices: efficient for high-dimensional text
   ✓ Handles 100k+ documents easily
   ✓ Fast prediction (milliseconds per batch)
   ✓ Memory efficient

9. Debugging
   ✓ Use model.print_tree() to inspect DecisionTree
   ✓ Check vectorizer.vocabulary_ to see learned tokens
   ✓ Print shape of each matrix to verify flow
   ✓ Use small subset first to test pipeline

10. Production Tips
    ✓ Save trained models (pickle or joblib)
    ✓ Save vectorizer and TF-IDF weights
    ✓ Version your preprocessing (important!)
    ✓ Monitor for feature/class distribution drift
"""

# ============================================================================
# PRINT ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EXAMPLE 1: Simple Naive Bayes Classification")
    print("="*70)
    print(EXAMPLE_1)
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Decision Tree with TF-IDF")
    print("="*70)
    print(EXAMPLE_2)
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Compare Models")
    print("="*70)
    print(EXAMPLE_3)
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Real Data Workflow")
    print("="*70)
    print(EXAMPLE_4)
    
    print("\n" + "="*70)
    print("KEY TIPS")
    print("="*70)
    print(TIPS)
