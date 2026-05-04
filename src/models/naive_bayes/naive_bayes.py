import numpy as np
from src.models.base import BaseModel


class NaiveBayes(BaseModel):
    """
    Naive Bayes classifier for classification tasks.
    
    This implementation supports both dense and sparse matrices.
    Uses log probabilities to avoid numerical underflow.
    
    For text classification with bag-of-words features:
    - Works with sparse count matrices from CountVectorizer
    - Implements Multinomial Naive Bayes
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Laplace smoothing parameter. Prevents zero probabilities.
        alpha > 0 adds smoothing to avoid log(0).
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_priors = None
        self.feature_log_probs = None
    
    def fit(self, X, y):
        """
        Train Naive Bayes classifier.
        
        Parameters:
        -----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training features. Can be dense (numpy array) or sparse (scipy.sparse).
        y : array-like of shape (n_samples,)
            Training labels.
        
        Returns:
        --------
        self : NaiveBayes
            Fitted classifier instance.
        """
        # Convert sparse to dense if needed (for multinomial NB, we need counts)
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).flatten()
        
        n_samples, n_features = X.shape
        
        # Get unique classes
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize parameters
        self.class_priors = np.zeros(n_classes)
        self.feature_log_probs = np.zeros((n_classes, n_features))
        
        # For each class, compute parameters
        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            
            # Class prior: P(class) = count(class) / n_samples
            self.class_priors[idx] = len(X_cls) / n_samples
            
            # Feature probabilities using Multinomial model with Laplace smoothing
            # P(feature|class) = (count(feature in class) + alpha) / (sum(counts in class) + alpha*n_features)
            feature_counts = X_cls.sum(axis=0)  # Sum counts for each feature within this class
            total_count = feature_counts.sum()
            
            # Laplace smoothing to avoid log(0)
            self.feature_log_probs[idx, :] = np.log(
                (feature_counts + self.alpha) / (total_count + self.alpha * n_features)
            )
        
        # Convert class priors to log space
        self.class_priors = np.log(self.class_priors)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
        Parameters:
        -----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Features to predict on.
        
        Returns:
        --------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Convert sparse to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = np.asarray(X, dtype=np.float32)
        
        # Compute log probabilities for each sample and class
        log_posteriors = self._compute_log_posteriors(X)
        
        # Predict the class with highest log posterior
        predictions = self.classes[np.argmax(log_posteriors, axis=1)]
        
        return np.array(predictions)
    
    def _compute_log_posteriors(self, X):
        """
        Compute log posterior probabilities for all samples and classes.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features (dense array).
        
        Returns:
        --------
        log_posteriors : np.ndarray of shape (n_samples, n_classes)
            Log posterior probability for each sample-class pair.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        log_posteriors = np.zeros((n_samples, n_classes))
        
        # For each class, compute log P(class | X) ∝ log P(class) + sum(log P(feature_i | class) * X_i)
        for idx in range(n_classes):
            # Log prior probability
            log_prior = self.class_priors[idx]
            
            # Log likelihood: sum of weighted log feature probabilities
            # For multinomial NB: sum_i(X_i * log P(feature_i | class))
            log_likelihood = np.dot(X, self.feature_log_probs[idx, :])
            
            log_posteriors[:, idx] = log_prior + log_likelihood
        
        return log_posteriors
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples.
        
        Parameters:
        -----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Features to predict on.
        
        Returns:
        --------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        # Convert sparse to dense if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = np.asarray(X, dtype=np.float32)
        
        # Compute log posteriors
        log_posteriors = self._compute_log_posteriors(X)
        
        # Convert log probabilities to probabilities using softmax
        # To avoid overflow, subtract max from each row
        log_posteriors_stable = log_posteriors - log_posteriors.max(axis=1, keepdims=True)
        posteriors = np.exp(log_posteriors_stable)
        
        # Normalize to sum to 1
        posteriors /= posteriors.sum(axis=1, keepdims=True)
        
        return posteriors