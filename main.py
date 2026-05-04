"""
Main entry point for sentiment analysis ML pipeline.

This script demonstrates a complete workflow:
1. Load and preprocess data
2. Split into train/test sets
3. Extract features using CountVectorizer
4. Train multiple models (Naive Bayes, Decision Tree)
5. Evaluate and compare models
6. Visualize results

Usage:
    python main.py
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    CountVectorizer,
    TfidfTransformer,
    NaiveBayes,
    DecisionTree,
    train_test_split,
    accuracy,
    confusion_matrix,
    TextProcessor
)
from src.evaluation.visualization import MetricsVisualizer
import matplotlib.pyplot as plt


def load_data(data_path=None):
    """
    Load sentiment dataset.
    
    Parameters:
    -----------
    data_path : str or None
        Path to CSV file with 'text' and 'sentiment' columns.
        If None, generates toy dataset.
    
    Returns:
    --------
    texts : np.ndarray
        Array of text documents
    labels : np.ndarray
        Array of labels (0=negative, 1=positive)
    """
    if data_path is not None and os.path.exists(data_path):
        # Load from file
        df = pd.read_csv(data_path)
        texts = df['text'].values
        labels = (df['sentiment'] == 'positive').astype(int).values
    else:
        # Toy dataset
        toy_data = [
            ("This movie is absolutely amazing and brilliant!", 1),
            ("Terrible waste of time, horrible acting", 0),
            ("Great story, excellent cinematography", 1),
            ("Bad plot, bad dialogues, disappointing", 0),
            ("Best film I've seen, highly recommended", 1),
            ("Awful movie, couldn't even finish it", 0),
            ("Good entertainment, worth watching", 1),
            ("Poor production quality and boring", 0),
            ("Outstanding performance by the actors", 1),
            ("Not worth your money, total garbage", 0),
            ("Fantastic movie, loved every moment", 1),
            ("Worst movie ever made, truly terrible", 0),
        ]
        texts = np.array([text for text, _ in toy_data])
        labels = np.array([label for _, label in toy_data])
    
    return texts, labels


def preprocess_texts(texts, remove_stopwords=False):
    """
    Preprocess text documents.
    
    Parameters:
    -----------
    texts : np.ndarray
        Array of text documents
    remove_stopwords : bool
        Whether to remove stopwords
    
    Returns:
    --------
    processed_texts : list
        List of preprocessed texts
    """
    processor = TextProcessor(remove_stopwords=remove_stopwords)
    processed = [' '.join(processor.process(text)) for text in texts]
    return processed


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and evaluate them.
    
    Parameters:
    -----------
    X_train : sparse matrix
        Training feature matrix
    X_test : sparse matrix
        Test feature matrix
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Test labels
    
    Returns:
    --------
    results : dict
        Dictionary with model results
    """
    results = {}
    
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    # Model 1: Naive Bayes on raw counts
    print("\n1. Training Naive Bayes (raw counts)...")
    nb_model = NaiveBayes(alpha=1.0)
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    acc_nb = accuracy(y_test, y_pred_nb)
    cm_nb, classes_nb = confusion_matrix(y_test, y_pred_nb)
    
    results['Naive Bayes'] = {
        'model': nb_model,
        'y_pred': y_pred_nb,
        'accuracy': acc_nb,
        'confusion_matrix': cm_nb,
        'classes': classes_nb
    }
    
    print(f"   Accuracy: {acc_nb:.4f}")
    print(f"   Confusion Matrix:\n{cm_nb}")
    
    # Model 2: Decision Tree on raw counts
    print("\n2. Training Decision Tree (raw counts)...")
    dt_model = DecisionTree(max_depth=4, min_samples_split=2)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    acc_dt = accuracy(y_test, y_pred_dt)
    cm_dt, classes_dt = confusion_matrix(y_test, y_pred_dt)
    
    results['Decision Tree'] = {
        'model': dt_model,
        'y_pred': y_pred_dt,
        'accuracy': acc_dt,
        'confusion_matrix': cm_dt,
        'classes': classes_dt
    }
    
    print(f"   Accuracy: {acc_dt:.4f}")
    print(f"   Confusion Matrix:\n{cm_dt}")
    
    # Model 3: Decision Tree on TF-IDF
    print("\n3. Training Decision Tree (TF-IDF)...")
    tfidf = TfidfTransformer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    dt_tfidf_model = DecisionTree(max_depth=4, min_samples_split=2)
    dt_tfidf_model.fit(X_train_tfidf, y_train)
    y_pred_dt_tfidf = dt_tfidf_model.predict(X_test_tfidf)
    acc_dt_tfidf = accuracy(y_test, y_pred_dt_tfidf)
    cm_dt_tfidf, classes_dt_tfidf = confusion_matrix(y_test, y_pred_dt_tfidf)
    
    results['Decision Tree (TF-IDF)'] = {
        'model': dt_tfidf_model,
        'y_pred': y_pred_dt_tfidf,
        'accuracy': acc_dt_tfidf,
        'confusion_matrix': cm_dt_tfidf,
        'classes': classes_dt_tfidf
    }
    
    print(f"   Accuracy: {acc_dt_tfidf:.4f}")
    print(f"   Confusion Matrix:\n{cm_dt_tfidf}")
    
    return results


def compare_models(results):
    """
    Compare and display model results.
    
    Parameters:
    -----------
    results : dict
        Dictionary with model results
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    accuracies = {name: data['accuracy'] for name, data in results.items()}
    
    print("\nAccuracy Comparison:")
    for model_name, acc in sorted(accuracies.items(), key=lambda x: -x[1]):
        print(f"  {model_name:25s}: {acc:.4f}")
    
    best_model = max(accuracies.items(), key=lambda x: x[1])
    print(f"\nBest Model: {best_model[0]} (Accuracy: {best_model[1]:.4f})")


def visualize_results(results, y_test):
    """
    Create visualization plots.
    
    Parameters:
    -----------
    results : dict
        Dictionary with model results
    y_test : np.ndarray
        Test labels
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Create output directory
    output_dir = PROJECT_ROOT / 'results' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Accuracy comparison
    print("\n1. Creating accuracy comparison chart...")
    accuracies = {name: data['accuracy'] for name, data in results.items()}
    fig, _ = MetricsVisualizer.plot_accuracy_comparison(accuracies)
    fig.savefig(output_dir / 'accuracy_comparison.png', dpi=100, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'accuracy_comparison.png'}")
    
    # 2. Confusion matrices
    print("\n2. Creating confusion matrix visualizations...")
    for model_name, data in results.items():
        fig, _ = MetricsVisualizer.plot_confusion_matrix(
            data['confusion_matrix'],
            data['classes'],
            title=f'Confusion Matrix - {model_name}',
            normalize=True
        )
        filename = f"cm_{model_name.replace(' ', '_').lower()}.png"
        fig.savefig(output_dir / filename, dpi=100, bbox_inches='tight')
        print(f"   Saved: {output_dir / filename}")
    
    # 3. Metrics for best model
    print("\n3. Creating detailed metrics visualization...")
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_data = results[best_model_name]
    fig, _ = MetricsVisualizer.plot_roc_style_metrics(y_test, best_data['y_pred'])
    fig.savefig(output_dir / 'best_model_metrics.png', dpi=100, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'best_model_metrics.png'}")
    
    plt.close('all')


def save_results(results, y_test, output_path=None):
    """
    Save results to file.
    
    Parameters:
    -----------
    results : dict
        Model results
    y_test : np.ndarray
        Test labels
    output_path : str or None
        Path to save results
    """
    if output_path is None:
        output_path = PROJECT_ROOT / 'results' / 'results.txt'
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SENTIMENT ANALYSIS MODEL EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for model_name, data in results.items():
            f.write(f"\nModel: {model_name}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy: {data['accuracy']:.4f}\n")
            f.write(f"Confusion Matrix:\n{data['confusion_matrix']}\n")
            f.write(f"Classes: {data['classes']}\n")
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS ML PIPELINE")
    print("="*60)
    
    # 1. Load data
    print("\n1. Loading data...")
    texts, labels = load_data()
    print(f"   Total samples: {len(texts)}")
    print(f"   Label distribution: {np.bincount(labels)}")
    
    # 2. Preprocess
    print("\n2. Preprocessing texts...")
    processed_texts = preprocess_texts(texts, remove_stopwords=False)
    print(f"   Preprocessing complete")
    
    # 3. Split data
    print("\n3. Splitting data (stratified)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        np.array(processed_texts),
        labels,
        test_size=0.2,
        random_state=42,
        stratify=True
    )
    print(f"   Train set: {len(X_train_text)} samples")
    print(f"   Test set: {len(X_test_text)} samples")
    print(f"   Train distribution: {np.bincount(y_train)}")
    print(f"   Test distribution: {np.bincount(y_test)}")
    
    # 4. Feature extraction
    print("\n4. Extracting features...")
    vectorizer = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        max_features=1000
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"   Training matrix shape: {X_train.shape}")
    print(f"   Test matrix shape: {X_test.shape}")
    
    # 5. Train and evaluate
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 6. Compare models
    compare_models(results)
    
    # 7. Visualize
    print("\nGenerating visualizations...")
    try:
        visualize_results(results, y_test)
    except Exception as e:
        print(f"   Warning: Could not generate visualizations: {e}")
    
    # 8. Save results
    save_results(results, y_test)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
