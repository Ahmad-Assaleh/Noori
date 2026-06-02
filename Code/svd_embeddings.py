#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
svd_embeddings.py - Apply SVD to co-occurrence matrices

This is where the magic happens:
1. Take high-dimensional sparse PPMI matrix
2. Apply SVD (Singular Value Decomposition)
3. Get dense, low-dimensional embeddings that capture semantic relationships

Why SVD?
- Finds latent dimensions in the data
- Reduces noise (high-dim → low-dim)
- Creates dense embeddings from sparse co-occurrence
- Deterministic and interpretable

The result: every root/word gets a dense vector that captures its semantic role.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from typing import Dict, Tuple
import pickle

from config import (
    ROOT_EMBEDDING_DIM,
    WORD_EMBEDDING_DIM,
    N_SVD_ITERATIONS,
    RANDOM_SEED
)


class SVDEmbedding:
    """
    Apply SVD to a PPMI matrix to get dense embeddings.
    
    Technical note: We use TruncatedSVD which is optimized for sparse matrices.
    This is way faster than full SVD and handles our large (but sparse) matrices.
    """
    
    def __init__(self, n_components: int, random_state: int = RANDOM_SEED):
        self.n_components = n_components
        self.random_state = random_state
        
        # SVD components
        self.svd_model = None
        self.embeddings = None  # The actual dense embeddings
        
        # Metadata
        self.explained_variance_ratio = None
        self.singular_values = None
    
    
    def fit_transform(self, ppmi_matrix: sparse.csr_matrix) -> np.ndarray:
        """
        Apply SVD to PPMI matrix.
        
        Args:
            ppmi_matrix: Sparse PPMI co-occurrence matrix (vocab_size × vocab_size)
        
        Returns:
            embeddings: Dense embeddings (vocab_size × n_components)
        """
        print(f"\nApplying SVD (reducing to {self.n_components} dimensions)...")
        
        vocab_size = ppmi_matrix.shape[0]
        print(f"  Input matrix: {vocab_size} × {vocab_size}")
        print(f"  Matrix density: {ppmi_matrix.nnz / (vocab_size * vocab_size) * 100:.2f}%")
        
        # Initialize TruncatedSVD
        # This uses randomized SVD which is much faster for large sparse matrices
        self.svd_model = TruncatedSVD(
            n_components=self.n_components,
            n_iter=N_SVD_ITERATIONS,
            random_state=self.random_state
        )
        
        # Fit and transform
        print(f"  Running SVD...")
        self.embeddings = self.svd_model.fit_transform(ppmi_matrix)
        
        # Store metadata
        self.explained_variance_ratio = self.svd_model.explained_variance_ratio_
        self.singular_values = self.svd_model.singular_values_
        
        # Check for zero-norm embeddings (diagnostic)
        norms = np.linalg.norm(self.embeddings, axis=1)
        num_zeros = np.sum(norms < 1e-10)
        
        # Report results
        total_variance = self.explained_variance_ratio.sum() * 100
        print(f"  ✓ SVD complete")
        print(f"  Output embeddings: {self.embeddings.shape[0]} × {self.embeddings.shape[1]}")
        print(f"  Explained variance: {total_variance:.1f}%")
        print(f"  Top 5 singular values: {self.singular_values[:5]}")
        
        if num_zeros > 0:
            print(f"  ⚠ Warning: {num_zeros} embeddings have near-zero norm")
            print(f"    These items had no meaningful co-occurrences")
            print(f"    ({num_zeros / vocab_size * 100:.1f}% of vocabulary)")
        
        return self.embeddings
    
    
    def get_embeddings(self) -> np.ndarray:
        """Get the computed embeddings."""
        if self.embeddings is None:
            raise ValueError("Must call fit_transform first")
        return self.embeddings
    
    
    def save(self, filepath: str):
        """Save SVD model and embeddings."""
        data = {
            'n_components': self.n_components,
            'embeddings': self.embeddings,
            'explained_variance_ratio': self.explained_variance_ratio,
            'singular_values': self.singular_values
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    
    @staticmethod
    def load(filepath: str) -> 'SVDEmbedding':
        """Load saved SVD model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = SVDEmbedding(n_components=data['n_components'])
        model.embeddings = data['embeddings']
        model.explained_variance_ratio = data['explained_variance_ratio']
        model.singular_values = data['singular_values']
        
        return model


class EmbeddingDatabase:
    """
    Store and query root/word embeddings.
    
    This is a simple lookup table with helpful query methods.
    """
    
    def __init__(self,
                 root_embeddings: np.ndarray,
                 word_embeddings: np.ndarray,
                 root_vocab: list,
                 word_vocab: list):
        
        self.root_embeddings = root_embeddings
        self.word_embeddings = word_embeddings
        self.root_vocab = root_vocab
        self.word_vocab = word_vocab
        
        # Create lookup dicts
        self.root_to_idx = {root: idx for idx, root in enumerate(root_vocab)}
        self.word_to_idx = {word: idx for idx, word in enumerate(word_vocab)}
        
        print(f"\nEmbedding database created:")
        print(f"  Root embeddings: {root_embeddings.shape}")
        print(f"  Word embeddings: {word_embeddings.shape}")
    
    
    def get_root_embedding(self, root: str) -> np.ndarray:
        """Get embedding for a root."""
        idx = self.root_to_idx.get(root)
        if idx is None:
            return None
        return self.root_embeddings[idx]
    
    
    def get_word_embedding(self, word: str) -> np.ndarray:
        """Get embedding for a word."""
        idx = self.word_to_idx.get(word)
        if idx is None:
            return None
        return self.word_embeddings[idx]
    
    
    def find_similar_roots(self, root: str, top_k: int = 10) -> list:
        """
        Find most similar roots using cosine similarity.
        
        Returns:
            List of (root, similarity) tuples
        """
        query_emb = self.get_root_embedding(root)
        if query_emb is None:
            return []
        
        # Check if query has zero norm
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            print(f"Warning: Root '{root}' has zero embedding (no co-occurrences)")
            return []
        
        # Normalize query
        query_norm_vec = query_emb / query_norm
        
        # Normalize all embeddings (with safety check)
        norms = np.linalg.norm(self.root_embeddings, axis=1, keepdims=True)
        
        # Replace zero norms with 1 to avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        
        embeddings_norm = self.root_embeddings / norms
        
        # Compute similarities
        similarities = embeddings_norm @ query_norm_vec
        
        # Get top k (excluding self and zero-norm embeddings)
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            if self.root_vocab[idx] == root:
                continue  # Skip self
            
            # Skip if embedding has zero norm (no valid similarity)
            if norms[idx, 0] == 1.0 and np.allclose(self.root_embeddings[idx], 0):
                continue
            
            similarity = float(similarities[idx])
            
            # Skip nan values
            if np.isnan(similarity):
                continue
            
            results.append((self.root_vocab[idx], similarity))
            
            if len(results) >= top_k:
                break
        
        return results
    
    
    def find_similar_words(self, word: str, top_k: int = 10) -> list:
        """
        Find most similar words using cosine similarity.
        
        Returns:
            List of (word, similarity) tuples
        """
        query_emb = self.get_word_embedding(word)
        if query_emb is None:
            return []
        
        # Check if query has zero norm
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            print(f"Warning: Word '{word}' has zero embedding (no co-occurrences)")
            return []
        
        # Normalize query
        query_norm_vec = query_emb / query_norm
        
        # Normalize all embeddings (with safety check)
        norms = np.linalg.norm(self.word_embeddings, axis=1, keepdims=True)
        
        # Replace zero norms with 1 to avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        
        embeddings_norm = self.word_embeddings / norms
        
        # Compute similarities
        similarities = embeddings_norm @ query_norm_vec
        
        # Get top k (excluding self and zero-norm embeddings)
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            if self.word_vocab[idx] == word:
                continue  # Skip self
            
            # Skip if embedding has zero norm
            if norms[idx, 0] == 1.0 and np.allclose(self.word_embeddings[idx], 0):
                continue
            
            similarity = float(similarities[idx])
            
            # Skip nan values
            if np.isnan(similarity):
                continue
            
            results.append((self.word_vocab[idx], similarity))
            
            if len(results) >= top_k:
                break
        
        return results
    
    
    def save(self, filepath: str):
        """Save embedding database."""
        data = {
            'root_embeddings': self.root_embeddings,
            'word_embeddings': self.word_embeddings,
            'root_vocab': self.root_vocab,
            'word_vocab': self.word_vocab
        }
        np.savez_compressed(filepath, **data)
    
    
    @staticmethod
    def load(filepath: str) -> 'EmbeddingDatabase':
        """Load embedding database."""
        data = np.load(filepath, allow_pickle=True)
        
        return EmbeddingDatabase(
            root_embeddings=data['root_embeddings'],
            word_embeddings=data['word_embeddings'],
            root_vocab=data['root_vocab'].tolist(),
            word_vocab=data['word_vocab'].tolist()
        )


def build_embeddings_from_matrices(root_ppmi: sparse.csr_matrix,
                                   word_ppmi: sparse.csr_matrix,
                                   root_vocab: list,
                                   word_vocab: list) -> EmbeddingDatabase:
    """
    Main function: Build embeddings from PPMI matrices.
    
    Args:
        root_ppmi: Root co-occurrence PPMI matrix
        word_ppmi: Word co-occurrence PPMI matrix
        root_vocab: List of roots (in matrix order)
        word_vocab: List of words (in matrix order)
    
    Returns:
        EmbeddingDatabase with dense embeddings
    """
    print("\n" + "="*70)
    print("BUILDING EMBEDDINGS WITH SVD")
    print("="*70)
    
    # Build root embeddings
    print("\n[1/2] Root embeddings")
    root_svd = SVDEmbedding(n_components=ROOT_EMBEDDING_DIM)
    root_embeddings = root_svd.fit_transform(root_ppmi)
    
    # Build word embeddings
    print("\n[2/2] Word embeddings")
    word_svd = SVDEmbedding(n_components=WORD_EMBEDDING_DIM)
    word_embeddings = word_svd.fit_transform(word_ppmi)
    
    # Create database
    print("\n" + "="*70)
    print("EMBEDDINGS COMPLETE")
    print("="*70)
    
    db = EmbeddingDatabase(
        root_embeddings=root_embeddings,
        word_embeddings=word_embeddings,
        root_vocab=root_vocab,
        word_vocab=word_vocab
    )
    
    return db


if __name__ == "__main__":
    print("SVD embeddings module")
    print("This should be used via the main pipeline")