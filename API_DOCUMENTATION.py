"""
API Documentation and Usage Guide for ML Pipeline

This document provides complete API reference and usage examples for all
implemented components.
"""

# ============================================================================
# 1. FEATURE EXTRACTION
# ============================================================================

FEATURE_EXTRACTION_GUIDE = """
========================================
1. CountVectorizer - Text to Sparse Counts
========================================

USAGE:
------
from src.feature_extraction.count_vectorizer import CountVectorizer

# Create vectorizer with unigrams and bigrams, limit vocabulary to 5000
vectorizer = CountVectorizer(
    lowercase=True,              # Convert to lowercase
    ngram_range=(1, 2),          # Unigrams and bigrams
    max_features=5000            # Keep top 5000 tokens
)

# Fit on training documents ONLY (no data leakage!)
X_train = vectorizer.fit_transform(docs_train)

# Transform test documents using learned vocabulary
X_test = vectorizer.transform(docs_test)

# Access vocabulary
vocab = vectorizer.vocabulary_  # dict: {token → feature_index}

OUTPUT:
-------
X_train : scipy.sparse.csr_matrix of shape (n_documents, n_features)
  - Each row = one document
  - Each column = one token
  - Value = raw count of token in document
  - Memory efficient: only stores non-zero values

PARAMETERS:
-----------
lowercase : bool, default=True
  Convert text to lowercase before tokenization
ngram_range : tuple, default=(1, 1)
  Range of n-gram sizes: (min_n, max_n)
  Examples:
    - (1, 1) → only unigrams
    - (1, 2) → unigrams and bigrams
    - (2, 2) → only bigrams
max_features : int or None, default=None
  If int: keep only top-k most frequent tokens
  If None: keep all tokens

METHODS:
--------
fit(documents)
  Learn vocabulary from training documents
  
transform(documents)
  Convert documents to count matrix using learned vocabulary
  
fit_transform(documents)
  Fit and transform in one step (equivalent to fit().transform())

NOTES:
------
- Tokenization: simple whitespace split
- NOT using sklearn
- Outputs sparse CSR matrix (memory efficient)
- Vocabulary built ONLY on training data
- Test data transformed using training vocabulary


========================================
2. TfidfTransformer - Count to TF-IDF Weights
========================================

USAGE:
------
from src.feature_extraction.tfidf_vectorizer import TfidfTransformer

# Create transformer
tfidf = TfidfTransformer(smooth_idf=True)

# Fit on training count matrix
X_train_tfidf = tfidf.fit_transform(X_train_counts)

# Transform test count matrix
X_test_tfidf = tfidf.transform(X_test_counts)

FORMULA:
--------
idf(t) = log((N + 1) / (df(t) + 1)) + 1
where:
  N = total number of documents
  df(t) = number of documents containing term t

tfidf(d, t) = tf(d, t) * idf(t)

Then L2 normalization applied to each row (document):
  x_normalized = x / ||x||_2

PARAMETERS:
-----------
smooth_idf : bool, default=True
  If True: add 1 to document frequencies (Laplace smoothing)
  If False: standard IDF without smoothing

METHODS:
--------
fit(X)
  Compute IDF weights from sparse count matrix X
  
transform(X)
  Apply TF-IDF weighting and L2 normalization
  
fit_transform(X)
  Fit and transform in one step

NOTES:
------
- Works directly with sparse matrices
- Input: scipy.sparse.csr_matrix (from CountVectorizer)
- Output: scipy.sparse.csr_matrix (L2 normalized)
- NO dense matrix conversion
- Each row has unit L2 norm (||x||_2 = 1)

PIPELINE EXAMPLE:
-----------------
from src.feature_extraction.count_vectorizer import CountVectorizer
from src.feature_extraction.tfidf_vectorizer import TfidfTransformer

# Step 1: Count vectorization
cv = CountVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_counts = cv.fit_transform(docs_train)
X_test_counts = cv.transform(docs_test)

# Step 2: TF-IDF transformation (optional)
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_counts)
X_test_tfidf = tfidf.transform(X_test_counts)

# Now use X_train_tfidf, X_test_tfidf with models
"""

# ============================================================================
# 2. MODELS
# ============================================================================

MODELS_GUIDE = """
========================================
3. BaseModel - Abstract Interface
========================================

All models inherit from BaseModel and implement:

INTERFACE:
----------
class MyModel(BaseModel):
    
    def fit(self, X, y):
        '''Train model'''
        pass
    
    def predict(self, X):
        '''Make predictions'''
        return np.ndarray  # shape (n_samples,)

PARAMETERS:
-----------
X : array-like or sparse matrix, shape (n_samples, n_features)
  Training/prediction features
y : array-like, shape (n_samples,)
  Training labels

RETURNS:
--------
predict() returns: np.ndarray of shape (n_samples,) with class labels


========================================
4. DecisionTree - Entropy-Based Tree Classifier
========================================

USAGE:
------
from src.models.decision_tree.decision_tree import DecisionTree

# Create tree
tree = DecisionTree(
    min_samples_split=2,  # Min samples to split node
    max_depth=5           # Max tree depth
)

# Train (converts sparse to dense internally if needed)
tree.fit(X_train, y_train)

# Predict
y_pred = tree.predict(X_test)

# Optional: inspect tree structure
tree.print_tree()

PARAMETERS:
-----------
min_samples_split : int, default=2
  Minimum number of samples required at node to allow split
max_depth : int, default=2
  Maximum depth of tree (limits overfitting)

METHODS:
--------
fit(X, y) → self
  Train decision tree on (X, y)
  
predict(X) → np.ndarray
  Return predicted class labels

ALGORITHM:
----------
- Information gain based on entropy
- Threshold: uses actual feature values observed in training
- Split criterion: maximize information gain
- Leaf: majority class within node

NOTES:
------
- Works with dense and sparse matrices
- Sparse matrices converted to dense internally
- Supports binary and multi-class classification
- Deterministic (no randomness)
- Can overfit with deep trees


========================================
5. NaiveBayes - Multinomial Classifier
========================================

USAGE:
------
from src.models.naive_bayes.naive_bayes import NaiveBayes

# Create classifier
nb = NaiveBayes(
    alpha=1.0  # Laplace smoothing
)

# Train (converts sparse to dense internally if needed)
nb.fit(X_train, y_train)

# Predict class labels
y_pred = nb.predict(X_test)

# Get class probabilities
y_proba = nb.predict_proba(X_test)

PARAMETERS:
-----------
alpha : float, default=1.0
  Laplace smoothing parameter
  Prevents zero probabilities and log(0) errors
  alpha=1.0 is standard add-one smoothing

METHODS:
--------
fit(X, y) → self
  Train on (X, y)
  
predict(X) → np.ndarray
  Return predicted class labels
  
predict_proba(X) → np.ndarray
  Return class probabilities
  Shape: (n_samples, n_classes)
  Each row sums to 1.0

ALGORITHM:
----------
Multinomial Naive Bayes:
- P(class) = count(class) / n_samples
- P(feature|class) = (count + alpha) / (total + alpha * n_features)
- log_posterior(x) = log P(class) + Σ(x_i * log P(feature_i|class))
- Prediction: argmax log_posterior

All computations in log space to avoid underflow.

IDEAL FOR:
----------
- Text classification with Bag of Words
- Count-based features (from CountVectorizer)
- Sparse high-dimensional data
- When interpretability matters

NOTES:
------
- Works with dense and sparse matrices
- Sparse matrices converted to dense internally
- Uses log space for numerical stability
- No hyperparameter tuning needed
- Very fast to train and predict


========================================
COMPLETE EXAMPLE: Train and Evaluate
========================================

from src.feature_extraction.count_vectorizer import CountVectorizer
from src.feature_extraction.tfidf_vectorizer import TfidfTransformer
from src.models.naive_bayes.naive_bayes import NaiveBayes
from src.models.decision_tree.decision_tree import DecisionTree
from src.evaluation.accuracy import accuracy
from src.evaluation.confusion_matrix import confusion_matrix
from src.utils.helper import train_test_split

# Data
docs = [doc1, doc2, ..., docN]
labels = np.array([0, 1, 1, 0, ...])

# Split with stratification
X_train_text, X_test_text, y_train, y_test = train_test_split(
    np.array(docs), labels, test_size=0.2, stratify=True
)

# Vectorize (fit ONLY on train!)
cv = CountVectorizer(lowercase=True, ngram_range=(1, 2), max_features=5000)
X_train = cv.fit_transform(X_train_text)
X_test = cv.transform(X_test_text)

# Optional: TF-IDF
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model 1: Naive Bayes on raw counts
nb = NaiveBayes(alpha=1.0)
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
acc_nb = accuracy(y_test, y_pred_nb)
print(f"Naive Bayes accuracy: {acc_nb:.4f}")

# Train model 2: Decision Tree on TF-IDF
dt = DecisionTree(max_depth=5)
dt.fit(X_train_tfidf, y_train)
y_pred_dt = dt.predict(X_test_tfidf)
acc_dt = accuracy(y_test, y_pred_dt)
print(f"Decision Tree accuracy: {acc_dt:.4f}")

# Confusion matrices
cm_nb, classes_nb = confusion_matrix(y_test, y_pred_nb)
cm_dt, classes_dt = confusion_matrix(y_test, y_pred_dt)

print("Naive Bayes CM:")
print(cm_nb)
print("Decision Tree CM:")
print(cm_dt)
"""

# ============================================================================
# 3. EVALUATION
# ============================================================================

EVALUATION_GUIDE = """
========================================
6. Metrics - Evaluation Functions
========================================

ACCURACY:
---------
from src.evaluation.accuracy import accuracy

acc = accuracy(y_true, y_pred)

INPUT:
  y_true : array-like, shape (n_samples,)
    True class labels
  y_pred : array-like, shape (n_samples,)
    Predicted class labels

OUTPUT:
  accuracy : float in [0, 1]
    Fraction of correct predictions
    = (# correct) / (# total)

EXAMPLE:
  y_true = [0, 1, 1, 0, 1]
  y_pred = [0, 1, 0, 0, 1]
  acc = 4/5 = 0.8


CONFUSION MATRIX:
-----------------
from src.evaluation.confusion_matrix import confusion_matrix

cm, classes = confusion_matrix(y_true, y_pred)

INPUT:
  y_true : array-like, shape (n_samples,)
    True class labels
  y_pred : array-like, shape (n_samples,)
    Predicted class labels

OUTPUT:
  cm : np.ndarray, shape (n_classes, n_classes)
    Confusion matrix
    cm[i, j] = count of samples with true=i, predicted=j
  
  classes : np.ndarray
    Unique class labels in sorted order

INTERPRETATION:
  Rows = true labels
  Columns = predicted labels
  Diagonal = correct predictions
  Off-diagonal = misclassifications

EXAMPLE:
  y_true = [0, 1, 0, 1, 0, 1]
  y_pred = [0, 1, 1, 1, 0, 0]
  
  cm = [[2, 1],
        [0, 2]]
  
  classes = [0, 1]
  
  True 0, Pred 0: 2 (correct)
  True 0, Pred 1: 1 (false positive)
  True 1, Pred 0: 0 (false negative)
  True 1, Pred 1: 2 (correct)
"""

# ============================================================================
# 4. UTILITIES
# ============================================================================

UTILITIES_GUIDE = """
========================================
7. Utilities - Data Splitting
========================================

TRAIN_TEST_SPLIT:
-----------------
from src.utils.helper import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80/20 split
    random_state=42,    # Seed for reproducibility
    stratify=True       # Maintain class distribution
)

PARAMETERS:
-----------
X : array-like, shape (n_samples, n_features)
  Features dataset
  
y : array-like, shape (n_samples,)
  Target labels
  
test_size : float, default=0.2
  Fraction for test set (0.2 = 20% test, 80% train)
  
random_state : int, default=42
  Random seed for reproducibility
  
stratify : bool, default=True
  If True: maintain class distribution in train/test
  If False: simple random split

RETURNS:
--------
X_train : array-like, shape (n_train, n_features)
  Training features (~80% of data)
  
X_test : array-like, shape (n_test, n_features)
  Test features (~20% of data)
  
y_train : array-like, shape (n_train,)
  Training labels
  
y_test : array-like, shape (n_test,)
  Test labels

STRATIFIED SPLIT:
-----------------
If stratify=True, for each class:
  1. Find all samples of that class
  2. Split 80/20 maintaining ratio
  3. Combine splits from all classes
  4. Shuffle results

This ensures train/test have same class distribution as original data.

Example:
  Original: 800 class 0, 200 class 1 (80/20)
  Train: 640 class 0, 160 class 1 (80/20 maintained)
  Test: 160 class 0, 40 class 1 (80/20 maintained)

RANDOM SPLIT:
--------------
If stratify=False:
  1. Shuffle all samples randomly
  2. Split by index

No guarantee of maintaining class ratios.

REPRODUCIBILITY:
-----------------
random_state=42 ensures reproducible splits across runs.

X1, X2, y1, y2 = train_test_split(X, y, random_state=42)
# Later, calling again with same random_state gives identical split
X1b, X2b, y1b, y2b = train_test_split(X, y, random_state=42)
# X1 == X1b, y1 == y1b, etc.
"""

# ============================================================================
# 5. IMPORTS
# ============================================================================

IMPORTS_GUIDE = """
========================================
8. Importing from src
========================================

# All-in-one import
from src import (
    CountVectorizer,
    TfidfTransformer,
    BaseModel,
    DecisionTree,
    NaiveBayes,
    accuracy,
    confusion_matrix,
    train_test_split,
    TextProcessor
)

# Individual imports
from src.feature_extraction.count_vectorizer import CountVectorizer
from src.feature_extraction.tfidf_vectorizer import TfidfTransformer
from src.models.base import BaseModel
from src.models.decision_tree.decision_tree import DecisionTree
from src.models.naive_bayes.naive_bayes import NaiveBayes
from src.evaluation.accuracy import accuracy
from src.evaluation.confusion_matrix import confusion_matrix
from src.utils.helper import train_test_split
from src.preprocessing.text_processor import TextProcessor

# All are properly exported in src/__init__.py
"""

# ============================================================================
# PRINT ALL
# ============================================================================

if __name__ == "__main__":
    print(FEATURE_EXTRACTION_GUIDE)
    print("\n" + "="*70 + "\n")
    print(MODELS_GUIDE)
    print("\n" + "="*70 + "\n")
    print(EVALUATION_GUIDE)
    print("\n" + "="*70 + "\n")
    print(UTILITIES_GUIDE)
    print("\n" + "="*70 + "\n")
    print(IMPORTS_GUIDE)
