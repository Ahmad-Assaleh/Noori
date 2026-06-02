#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cooccurrence.py - Build co-occurrence matrices with PPMI weighting

This module builds the foundation for our embeddings:
1. Root-root co-occurrence (thematic relationships)
2. Word-word co-occurrence (distributional semantics)

Uses PPMI (Positive Pointwise Mutual Information) weighting to:
- Downweight common function words
- Upweight meaningful semantic associations
- Handle frequency biases properly

The key insight: raw counts are meaningless. PPMI captures true association strength.
"""

import numpy as np
from scipy import sparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import pickle

from config import (
    PPMI_SHIFT,
    PPMI_CONTEXT_DISTRIBUTION_SMOOTHING,
    MIN_ROOT_FREQUENCY,
    MIN_WORD_FREQUENCY,
    USE_SPARSE_MATRICES,
    CONTEXT_WINDOW_TYPE,
    FIXED_WINDOW_SIZE
)


class CooccurrenceMatrixBuilder:
    """
    Build co-occurrence matrices from Quran text with proper weighting.
    
    The goal: capture which roots/words are meaningfully related,
    not just which ones happen to appear together by chance.
    """
    
    def __init__(self, corpus_parser):
        self.parser = corpus_parser
        
        # Vocabularies
        self.root_vocab: List[str] = []
        self.word_vocab: List[str] = []
        self.root_to_idx: Dict[str, int] = {}
        self.word_to_idx: Dict[str, int] = {}
        
        # Frequency counts
        self.root_counts: Counter = Counter()
        self.word_counts: Counter = Counter()
        
        # Co-occurrence counts (raw)
        self.root_cooccurrence: Dict[Tuple[str, str], float] = defaultdict(float)
        self.word_cooccurrence: Dict[Tuple[str, str], float] = defaultdict(float)
        
        # Statistics
        self.total_root_contexts = 0
        self.total_word_contexts = 0
    
    
    def build_vocabularies(self, verses: List[Tuple[str, str]]):
        """
        Build vocabularies from all verses.
        Filter out rare items to reduce noise.
        
        Args:
            verses: List of (verse_id, verse_text) tuples
        """
        print("\nBuilding vocabularies...")
        
        # Count frequencies
        for verse_id, verse_text in verses:
            annotations = self.parser.parse_verse_words(verse_id, verse_text)
            
            for ann in annotations:
                # Count roots
                if ann.root:
                    self.root_counts[ann.root] += 1
                
                # Count words (use transliterated form from Corpus)
                # In the new parser, ann.word is the transliterated form
                self.word_counts[ann.word] += 1
        
        # Filter by minimum frequency
        self.root_vocab = [
            root for root, count in self.root_counts.items()
            if count >= MIN_ROOT_FREQUENCY
        ]
        self.word_vocab = [
            word for word, count in self.word_counts.items()
            if count >= MIN_WORD_FREQUENCY
        ]
        
        # Sort for consistency
        self.root_vocab.sort()
        self.word_vocab.sort()
        
        # Create index mappings
        self.root_to_idx = {root: idx for idx, root in enumerate(self.root_vocab)}
        self.word_to_idx = {word: idx for idx, word in enumerate(self.word_vocab)}
        
        print(f"  Root vocabulary: {len(self.root_vocab)} roots (filtered from {len(self.root_counts)})")
        print(f"  Word vocabulary: {len(self.word_vocab)} words (filtered from {len(self.word_counts)})")
    
    
    def extract_context_pairs_syntactic(self, verse_id: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Extract co-occurrence pairs based on SYNTACTIC DEPENDENCIES.
        
        Since we don't have dependency parsing yet, we use adjacent words
        as an approximation of syntactic relationships.
        
        Returns:
            (root_pairs, word_pairs)
        """
        annotations = self.parser.get_annotations(verse_id)
        
        root_pairs = []
        word_pairs = []
        
        # Use adjacent words as approximation of syntactic dependencies
        # (Real dependency parsing would be better, but we don't have that yet)
        for i in range(len(annotations) - 1):
            ann1 = annotations[i]
            ann2 = annotations[i + 1]
            
            # Root pairs (bidirectional)
            if ann1.root and ann2.root:
                root_pairs.append((ann1.root, ann2.root))
                root_pairs.append((ann2.root, ann1.root))
            
            # Word pairs (bidirectional)
            word_pairs.append((ann1.word, ann2.word))
            word_pairs.append((ann2.word, ann1.word))
        
        return root_pairs, word_pairs
    
    
    def extract_context_pairs_fixed_window(self, verse_id: str, window_size: int = FIXED_WINDOW_SIZE) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Extract co-occurrence pairs using FIXED WINDOW.
        
        Each word co-occurs with words within ±window_size positions.
        """
        annotations = self.parser.get_annotations(verse_id)
        
        root_pairs = []
        word_pairs = []
        
        for i, ann in enumerate(annotations):
            # Look at words within window
            start = max(0, i - window_size)
            end = min(len(annotations), i + window_size + 1)
            
            for j in range(start, end):
                if i == j:
                    continue
                
                context_ann = annotations[j]
                
                # Add pairs (only if roots exist)
                if ann.root and context_ann.root:
                    root_pairs.append((ann.root, context_ann.root))
                
                word_pairs.append((ann.word, context_ann.word))
        
        return root_pairs, word_pairs
    
    
    def extract_context_pairs_verse_level(self, verse_id: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Extract co-occurrence pairs at VERSE LEVEL.
        
        All words in the same verse co-occur with each other.
        Simple but loses fine-grained structure.
        """
        annotations = self.parser.get_annotations(verse_id)
        
        root_pairs = []
        word_pairs = []
        
        # All pairs in verse
        for i, ann1 in enumerate(annotations):
            for j, ann2 in enumerate(annotations):
                if i == j:
                    continue
                
                # Add root pairs (only if both have roots)
                if ann1.root and ann2.root:
                    root_pairs.append((ann1.root, ann2.root))
                
                word_pairs.append((ann1.word, ann2.word))
        
        return root_pairs, word_pairs
    
    
    def build_cooccurrence_counts(self, verses: List[Tuple[str, str]]):
        """
        Build raw co-occurrence counts from all verses.
        
        Args:
            verses: List of (verse_id, verse_text) tuples
        """
        print(f"\nBuilding co-occurrence counts (context: {CONTEXT_WINDOW_TYPE})...")
        
        for verse_id, verse_text in verses:
            # Parse verse
            self.parser.parse_verse_words(verse_id, verse_text)
            
            # Extract context pairs based on chosen method
            if CONTEXT_WINDOW_TYPE == "syntactic":
                root_pairs, word_pairs = self.extract_context_pairs_syntactic(verse_id)
            elif CONTEXT_WINDOW_TYPE == "fixed_window":
                root_pairs, word_pairs = self.extract_context_pairs_fixed_window(verse_id)
            else:  # verse_level
                root_pairs, word_pairs = self.extract_context_pairs_verse_level(verse_id)
            
            # Count occurrences
            for root1, root2 in root_pairs:
                if root1 in self.root_to_idx and root2 in self.root_to_idx:
                    self.root_cooccurrence[(root1, root2)] += 1
                    self.total_root_contexts += 1
            
            for word1, word2 in word_pairs:
                if word1 in self.word_to_idx and word2 in self.word_to_idx:
                    self.word_cooccurrence[(word1, word2)] += 1
                    self.total_word_contexts += 1
        
        print(f"  Root co-occurrences: {len(self.root_cooccurrence)} unique pairs")
        print(f"  Word co-occurrences: {len(self.word_cooccurrence)} unique pairs")
        print(f"  Total contexts: {self.total_root_contexts} (roots), {self.total_word_contexts} (words)")
    
    
    def compute_ppmi_matrix(self, cooccurrence_dict: Dict[Tuple[str, str], float],
                           vocab_to_idx: Dict[str, int],
                           vocab_counts: Counter,
                           total_contexts: int,
                           vocab_name: str = "item") -> sparse.csr_matrix:
        """
        Compute PPMI (Positive Pointwise Mutual Information) matrix.
        
        PPMI formula:
            PPMI(w, c) = max(0, log(P(w,c) / (P(w) * P(c)^α)))
        
        Where:
            P(w,c) = count(w,c) / total_contexts
            P(w) = count(w) / total_contexts
            P(c)^α = smoothed context distribution (α = 0.75 is standard)
        
        PPMI > 0 means words co-occur MORE than expected by chance.
        PPMI = 0 means no meaningful association.
        
        This is THE KEY to handling frequency biases properly.
        """
        print(f"\nComputing PPMI matrix for {vocab_name}...")
        
        vocab_size = len(vocab_to_idx)
        
        # Build sparse matrix (memory efficient)
        if USE_SPARSE_MATRICES:
            matrix = sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)
        else:
            matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        
        # Compute smoothed context distribution
        alpha = PPMI_CONTEXT_DISTRIBUTION_SMOOTHING
        total_count = sum(vocab_counts.values())
        
        # P(context)^alpha for each item
        p_context = {}
        for item, count in vocab_counts.items():
            if item in vocab_to_idx:
                p_context[item] = (count / total_count) ** alpha
        
        # Normalize smoothed distribution
        p_context_sum = sum(p_context.values())
        for item in p_context:
            p_context[item] /= p_context_sum
        
        # Compute PPMI for each pair
        num_pairs = len(cooccurrence_dict)
        print(f"  Processing {num_pairs} co-occurrence pairs...")
        
        for (item1, item2), count in cooccurrence_dict.items():
            idx1 = vocab_to_idx.get(item1)
            idx2 = vocab_to_idx.get(item2)
            
            if idx1 is None or idx2 is None:
                continue
            
            # Joint probability
            p_joint = count / total_contexts
            
            # Marginal probabilities
            p_item1 = vocab_counts[item1] / total_count
            p_item2_smoothed = p_context[item2]
            
            # PMI
            if p_joint > 0 and p_item1 > 0 and p_item2_smoothed > 0:
                pmi = np.log(p_joint / (p_item1 * p_item2_smoothed))
                
                # Shifted PPMI (helps with sparsity)
                ppmi = max(0, pmi - PPMI_SHIFT)
                
                matrix[idx1, idx2] = ppmi
        
        # Convert to efficient format
        if USE_SPARSE_MATRICES:
            matrix = matrix.tocsr()
            density = matrix.nnz / (vocab_size * vocab_size) * 100
            print(f"  Matrix density: {density:.4f}%")
            print(f"  Non-zero entries: {matrix.nnz:,}")
            
            # Diagnostic: check for rows with no non-zero entries
            row_sums = np.array(matrix.sum(axis=1)).flatten()
            zero_rows = np.sum(row_sums == 0)
            if zero_rows > 0:
                print(f"  ⚠ Warning: {zero_rows} items have NO co-occurrences")
                print(f"    These will get zero embeddings after SVD")
        
        print(f"  ✓ PPMI matrix computed: {vocab_size} × {vocab_size}")
        
        return matrix
    
    
    def build_all_matrices(self, verses: List[Tuple[str, str]]) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """
        Build all co-occurrence matrices with PPMI weighting.
        
        Returns:
            (root_ppmi_matrix, word_ppmi_matrix)
        """
        # Step 1: Build vocabularies
        self.build_vocabularies(verses)
        
        # Step 2: Build co-occurrence counts
        self.build_cooccurrence_counts(verses)
        
        # Step 3: Compute PPMI matrices
        root_matrix = self.compute_ppmi_matrix(
            self.root_cooccurrence,
            self.root_to_idx,
            self.root_counts,
            self.total_root_contexts,
            vocab_name="roots"
        )
        
        word_matrix = self.compute_ppmi_matrix(
            self.word_cooccurrence,
            self.word_to_idx,
            self.word_counts,
            self.total_word_contexts,
            vocab_name="words"
        )
        
        return root_matrix, word_matrix
    
    
    def save(self, filepath: str):
        """Save builder state."""
        data = {
            'root_vocab': self.root_vocab,
            'word_vocab': self.word_vocab,
            'root_to_idx': self.root_to_idx,
            'word_to_idx': self.word_to_idx,
            'root_counts': dict(self.root_counts),
            'word_counts': dict(self.word_counts),
            'statistics': {
                'total_root_contexts': self.total_root_contexts,
                'total_word_contexts': self.total_word_contexts,
                'num_root_pairs': len(self.root_cooccurrence),
                'num_word_pairs': len(self.word_cooccurrence),
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    
    @staticmethod
    def load(filepath: str, corpus_parser) -> 'CooccurrenceMatrixBuilder':
        """Load saved builder state."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        builder = CooccurrenceMatrixBuilder(corpus_parser)
        builder.root_vocab = data['root_vocab']
        builder.word_vocab = data['word_vocab']
        builder.root_to_idx = data['root_to_idx']
        builder.word_to_idx = data['word_to_idx']
        builder.root_counts = Counter(data['root_counts'])
        builder.word_counts = Counter(data['word_counts'])
        
        stats = data['statistics']
        builder.total_root_contexts = stats['total_root_contexts']
        builder.total_word_contexts = stats['total_word_contexts']
        
        return builder


if __name__ == "__main__":
    print("Co-occurrence matrix builder")
    print("This module should be used via the main pipeline")