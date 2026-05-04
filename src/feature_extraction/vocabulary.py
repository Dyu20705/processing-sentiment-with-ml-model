"""
Vocabulary management for natural language processing.

This module provides a Vocabulary class for building and managing word-to-index
mappings commonly used in NLP tasks like text classification and language modeling.

Classes:
--------
Vocabulary : Manages word-to-index and index-to-word mappings
"""

import collections


class Vocabulary:
    """
    Build and manage vocabulary (word-to-index mapping) from corpus.
    
    This class handles:
    - Building vocabulary from a list of words
    - Word-to-index (word2idx) and index-to-word (idx2word) mappings
    - Freezing vocabulary to prevent modifications
    - Unknown word handling with special <UNK> token
    
    Attributes:
    -----------
    word2idx : dict
        Mapping from word string to integer index
    idx2word : dict
        Mapping from integer index to word string
    freeze : bool
        Whether vocabulary is frozen (cannot add new words)
    
    Example:
    --------
    >>> vocab = Vocabulary()
    >>> words = ['hello', 'world', 'hello', 'python', 'world']
    >>> vocab.build_from_corpus(words, max_features=100)
    >>> vocab.get_index('hello')  # Returns index of 'hello'
    >>> vocab.get_word(1)          # Returns word at index 1
    """
    
    def __init__(self):
        """Initialize empty vocabulary."""
        self.word2idx = {}
        self.idx2word = {}
        self.freeze = False

    def reverse_vocab(self):
        """
        Create reverse mapping from index to word.
        
        This creates idx2word dictionary from word2idx.
        Call this after building vocabulary to enable idx2word lookups.
        """
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def add(self, word):
        """
        Add a single word to vocabulary.
        
        Parameters:
        -----------
        word : str
            Word to add to vocabulary
        
        Raises:
        -------
        ValueError if vocabulary is frozen
        """
        if self.freeze:
            raise ValueError("Vocabulary is frozen. Cannot add new words.")

        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx

    def get_index(self, word):
        """
        Get index of a word.
        
        Parameters:
        -----------
        word : str
            Word to look up
        
        Returns:
        --------
        index : int
            Index of word, or index of <UNK> if not in vocabulary
        """
        if word in self.word2idx:
            return self.word2idx[word]
        elif '<UNK>' in self.word2idx:
            return self.word2idx['<UNK>']
        else:
            # If no <UNK> token, return 0
            return 0

    def get_word(self, index):
        """
        Get word from index.
        
        Parameters:
        -----------
        index : int
            Index to look up
        
        Returns:
        --------
        word : str
            Word at index, or <UNK> if not found
        """
        if index in self.idx2word:
            return self.idx2word[index]
        elif 0 in self.idx2word:
            return self.idx2word[0]
        else:
            return '<UNK>'
    
    def freeze_vocab(self):
        """
        Freeze vocabulary to prevent adding new words.
        
        After freezing, any attempt to add new words will raise an error.
        This is useful to prevent accidental modifications after vocabulary
        is finalized.
        """
        self.freeze = True
    
    def unfreeze_vocab(self):
        """Unfreeze vocabulary to allow adding new words."""
        self.freeze = False
    
    def build_from_corpus(self, words, max_features=None):
        """
        Build vocabulary from a corpus of words.
        
        This method counts word frequencies and creates vocabulary from
        the most frequent words (respecting max_features limit).
        
        Parameters:
        -----------
        words : list of str
            List of words from corpus
        max_features : int or None, default=None
            Maximum number of words to keep. If None, keep all unique words.
        
        Raises:
        -------
        ValueError if vocabulary is already frozen
        """
        if self.freeze:
            raise ValueError("Vocabulary is frozen. Cannot build from corpus.")
        
        # Add special <UNK> token first
        if '<UNK>' not in self.word2idx:
            self.add('<UNK>')
        
        # Count word frequencies
        word_count = collections.Counter(words)
        
        # Get most common words (respecting max_features)
        most_common = word_count.most_common(max_features)
        
        # Add words to vocabulary
        for word, _ in most_common:
            if word != '<UNK>':  # Don't add <UNK> again
                self.add(word)
        
        # Create reverse mapping
        self.reverse_vocab()
    
    def size(self):
        """
        Get size of vocabulary.
        
        Returns:
        --------
        size : int
            Number of words in vocabulary
        """
        return len(self.word2idx)
    
    def __len__(self):
        """Get vocabulary size."""
        return self.size()
    
    def __repr__(self):
        """String representation of vocabulary."""
        return f"Vocabulary(size={self.size()}, frozen={self.freeze})"
