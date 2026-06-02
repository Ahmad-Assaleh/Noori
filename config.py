#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py - Configuration for Quran Semantic Embedding System

All hyperparameters, paths, and settings in one place.
Makes it easy to experiment and tune without touching core logic.
"""

# ============================================================================
# PATHS
# ============================================================================
QURAN_XML_PATH = "quran.xml"
QURAN_CORPUS_PATH = "quranic-corpus-morphology-0.4.txt"  # Download from corpus.quran.com

# Output paths
OUTPUT_DIR = "quran_embeddings_output"
ROOT_EMBEDDINGS_NPZ = f"{OUTPUT_DIR}/root_embeddings.npz"
WORD_EMBEDDINGS_NPZ = f"{OUTPUT_DIR}/word_embeddings.npz"
VERSE_EMBEDDINGS_NPZ = f"{OUTPUT_DIR}/verse_embeddings.npz"
METADATA_PKL = f"{OUTPUT_DIR}/metadata.pkl"
COOCCURRENCE_DIR = f"{OUTPUT_DIR}/cooccurrence_matrices"

# ============================================================================
# SVD HYPERPARAMETERS
# ============================================================================

# Dimensionality of embeddings
ROOT_EMBEDDING_DIM = 200   # Fewer roots (~2000) → 200 dims captures most variance
WORD_EMBEDDING_DIM = 384   # More words (~15000) → 384 dims for better coverage

# Word embedding method
WORD_EMBEDDING_METHOD = "word2vec"  # Options: "svd", "word2vec"
# "svd": Traditional SVD on PPMI matrix (fast but low variance ~34%)
# "word2vec": Neural Word2Vec with hybrid word+root context (slower but high quality)

# Word2Vec hyperparameters (only used if WORD_EMBEDDING_METHOD = "word2vec")
WORD2VEC_WINDOW_SIZE = 5        # Context window size (±5 words)
WORD2VEC_NEGATIVE_SAMPLES = 5   # Number of negative samples
WORD2VEC_EPOCHS = 5             # Number of training epochs
WORD2VEC_BATCH_SIZE = 512       # Batch size for training
WORD2VEC_LEARNING_RATE = 0.025  # Learning rate
WORD2VEC_MIN_COUNT = 1          # Minimum word frequency

# PPMI weighting parameters (used for ROOT embeddings only if word2vec enabled)
PPMI_SHIFT = 0.0          # No shift (preserve all positive PMI values)
PPMI_CONTEXT_DISTRIBUTION_SMOOTHING = 0.75  # Smooth context distribution (standard: 0.75)

# Co-occurrence context
CONTEXT_WINDOW_TYPE = "fixed_window"  # Use fixed window for more co-occurrences
FIXED_WINDOW_SIZE = 10                # Wider window = more connections

# Minimum frequency thresholds (filter noise)
MIN_ROOT_FREQUENCY = 1    # Include ALL roots (even appearing once)
MIN_WORD_FREQUENCY = 1    # Include ALL words (even appearing once)

# ============================================================================
# VERSE EMBEDDING COMPOSITION
# ============================================================================

# How to combine word/root embeddings into verse embedding
VERSE_COMPOSITION_METHOD = "weighted_average"  # Options: "weighted_average", "attention", "pooling"

# Weighting scheme for words in verse
WORD_WEIGHT_SCHEME = "syntactic_importance"  # Options: "uniform", "idf", "syntactic_importance", "combined"

# Syntactic importance weights (higher = more semantically important)
SYNTACTIC_WEIGHTS = {
    # Core predicate elements (highest weight)
    'verb': 1.0,
    'subject': 0.9,
    'object': 0.9,
    
    # Nominal elements
    'noun': 0.8,
    'proper_noun': 0.85,
    'adjective': 0.7,
    
    # Modifiers
    'adverb': 0.6,
    'pronoun': 0.5,
    
    # Function words (lower weight, but not zero)
    'preposition': 0.3,
    'particle': 0.3,
    'conjunction': 0.2,
    
    # Default
    'unknown': 0.5
}

# Normalization strategy for verse embeddings
VERSE_NORMALIZATION = "l2"  # Options: "l2", "none", "max"

# ============================================================================
# MORPHOLOGICAL FEATURE WEIGHTS
# ============================================================================

# How much to weight different components in final verse embedding
COMPONENT_WEIGHTS = {
    'root_embedding': 0.7,      # Root-level semantics (thematic)
    'word_embedding': 0.2,      # Word-level semantics (distributional)
    'morphological_features': 0.1  # Grammatical features (structural)
}

# ============================================================================
# EVALUATION & VALIDATION
# ============================================================================

# Test cases for semantic similarity validation
TEST_CASES = {
    'mercy': {
        'query_verses': ['2:163', '55:1', '1:3'],  # Verses with رحم root
        'expected_related': ['3:31', '4:64', '39:53'],  # Verses about forgiveness/mercy without رحم
    },
    'justice': {
        'query_verses': ['4:40', '99:7', '99:8'],
        'expected_related': ['18:49', '21:47', '10:54'],
    },
    'gratitude': {
        'query_verses': ['14:7', '31:12', '2:152'],
        'expected_related': ['16:18', '27:40', '35:3'],
    }
}

# Similarity threshold for considering verses "related"
SIMILARITY_THRESHOLD = 0.5

# ============================================================================
# COMPUTATIONAL SETTINGS
# ============================================================================

# Memory and performance
MAX_COOCCURRENCE_MEMORY_GB = 8  # Max memory for sparse matrices
USE_SPARSE_MATRICES = True       # Use scipy.sparse for co-occurrence (memory efficient)
N_SVD_ITERATIONS = 10            # Number of iterations for randomized SVD
RANDOM_SEED = 42                 # For reproducibility

# Parallel processing
N_JOBS = -1  # Use all available cores (-1), or set specific number

# ============================================================================
# LOGGING & DEBUG
# ============================================================================

VERBOSE = True
LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
SAVE_INTERMEDIATE_MATRICES = True  # Save co-occurrence matrices for inspection

# ============================================================================
# VALIDATION FLAGS
# ============================================================================

def validate_config():
    """
    Validate configuration to catch errors early.
    Raises ValueError if configuration is invalid.
    """
    
    # Check dimensionality
    if ROOT_EMBEDDING_DIM < 50 or ROOT_EMBEDDING_DIM > 1000:
        raise ValueError(f"ROOT_EMBEDDING_DIM should be 50-1000, got {ROOT_EMBEDDING_DIM}")
    
    if WORD_EMBEDDING_DIM < 50 or WORD_EMBEDDING_DIM > 1000:
        raise ValueError(f"WORD_EMBEDDING_DIM should be 50-1000, got {WORD_EMBEDDING_DIM}")
    
    # Check weights sum to 1
    weight_sum = sum(COMPONENT_WEIGHTS.values())
    if  not (0.99 <= weight_sum <= 1.01):
        raise ValueError(f"COMPONENT_WEIGHTS must sum to 1.0, got {weight_sum}")
    
    # Check context window type
    valid_contexts = ["syntactic", "fixed_window", "verse_level"]
    if CONTEXT_WINDOW_TYPE not in valid_contexts:
        raise ValueError(f"CONTEXT_WINDOW_TYPE must be one of {valid_contexts}")
    
    # Check normalization
    valid_norms = ["l2", "none", "max"]
    if VERSE_NORMALIZATION not in valid_norms:
        raise ValueError(f"VERSE_NORMALIZATION must be one of {valid_norms}")
    
    print("✓ Configuration validated successfully")


if __name__ == "__main__":
    validate_config()
    print("\nConfiguration Summary:")
    print(f"  Root embedding dim: {ROOT_EMBEDDING_DIM}")
    print(f"  Word embedding dim: {WORD_EMBEDDING_DIM}")
    print(f"  Context type: {CONTEXT_WINDOW_TYPE}")
    print(f"  Component weights: {COMPONENT_WEIGHTS}")