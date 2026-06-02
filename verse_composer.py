#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verse_composer.py - Compose verse embeddings from word/root embeddings

This is THE CRITICAL MODULE for solving your problem:
- Makes verse embeddings independent of verse length
- Weights words by semantic importance, not frequency
- Combines root + word + morphological information

The goal: Two verses about "mercy" should be similar REGARDLESS of:
- How long they are
- How many rare words they have
- Surface form differences

We achieve this through:
1. Syntactic importance weighting (verbs > particles)
2. Normalization (length-independent)
3. Multi-component fusion (root + word + morphology)
"""

import numpy as np
from typing import List, Dict, Optional
import pickle

from config import (
    VERSE_COMPOSITION_METHOD,
    WORD_WEIGHT_SCHEME,
    SYNTACTIC_WEIGHTS,
    VERSE_NORMALIZATION,
    COMPONENT_WEIGHTS,
    ROOT_EMBEDDING_DIM,
    WORD_EMBEDDING_DIM
)


class VerseEmbeddingComposer:
    """
    Compose verse-level embeddings from word/root embeddings.
    
    The key insight: A verse is NOT just the average of its words.
    Some words are more semantically important than others.
    """
    
    def __init__(self, embedding_db, corpus_parser):
        self.embedding_db = embedding_db
        self.parser = corpus_parser
        
        # Storage
        self.verse_embeddings: Dict[str, np.ndarray] = {}
        self.verse_metadata: Dict[str, Dict] = {}
    
    
    def compute_word_weights(self, annotations: List) -> np.ndarray:
        """
        Compute semantic importance weight for each word in a verse.
        
        This is THE KEY to making embeddings length-independent and
        focused on semantic content rather than function words.
        
        Args:
            annotations: List of WordAnnotation objects for the verse
        
        Returns:
            weights: Array of weights (one per word), normalized to sum to 1
        """
        weights = np.zeros(len(annotations))
        
        for i, ann in enumerate(annotations):
            if WORD_WEIGHT_SCHEME == "uniform":
                # All words equal (baseline)
                weights[i] = 1.0
            
            elif WORD_WEIGHT_SCHEME == "syntactic_importance":
                # Weight by POS tag (verbs > nouns > particles)
                pos = ann.pos.lower()
                weights[i] = SYNTACTIC_WEIGHTS.get(pos, SYNTACTIC_WEIGHTS['unknown'])
            
            elif WORD_WEIGHT_SCHEME == "idf":
                # Weight by inverse document frequency (rare words = higher weight)
                # This is what you DON'T want for theological terms!
                root_freq = self.parser.root_frequency.get(ann.root, 1)
                weights[i] = 1.0 / np.log(1 + root_freq)
            
            elif WORD_WEIGHT_SCHEME == "combined":
                # Combine syntactic importance + IDF
                pos = ann.pos.lower()
                syntactic_weight = SYNTACTIC_WEIGHTS.get(pos, SYNTACTIC_WEIGHTS['unknown'])
                
                root_freq = self.parser.root_frequency.get(ann.root, 1)
                idf_weight = 1.0 / np.log(1 + root_freq)
                
                # Average the two
                weights[i] = (syntactic_weight + idf_weight) / 2.0
        
        # Normalize weights to sum to 1
        # This makes embedding magnitude independent of verse length
        weights = weights / (weights.sum() + 1e-10)
        
        return weights
    
    
    def get_morphological_features(self, annotations: List) -> np.ndarray:
        """
        Extract morphological features for the verse.
        
        These are grammatical/structural features that complement
        the semantic embeddings from roots/words.
        
        Returns:
            feature_vector: Fixed-size vector of morphological features
        """
        # Count features
        pos_counts = {}
        for ann in annotations:
            pos = ann.pos.lower()
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        # Define feature vector structure
        # This is a simple bag-of-POS representation
        # You can extend this with more sophisticated features
        pos_types = ['verb', 'noun', 'pronoun', 'adjective', 'adverb', 
                     'preposition', 'particle', 'conjunction']
        
        features = []
        
        # POS distribution (normalized by verse length)
        verse_length = len(annotations)
        for pos_type in pos_types:
            count = pos_counts.get(pos_type, 0)
            features.append(count / verse_length)  # Normalize
        
        # Additional features
        features.append(verse_length / 100.0)  # Verse length (scaled)
        features.append(len(set(ann.root for ann in annotations)) / verse_length)  # Root diversity
        
        return np.array(features, dtype=np.float32)
    
    
    def compose_verse_embedding(self, verse_id: str, verse_text: str) -> np.ndarray:
        """
        Compose a single verse embedding.
        
        This is the main function that brings everything together:
        1. Get annotations for the verse
        2. Compute word weights (semantic importance)
        3. Get root embeddings for each word
        4. Get word embeddings for each word
        5. Get morphological features
        6. Combine into final verse embedding
        
        Returns:
            verse_embedding: Dense vector representing the verse semantically
        """
        # Parse verse
        annotations = self.parser.parse_verse_words(verse_id, verse_text)
        
        if not annotations:
            # Empty verse - return zero vector
            total_dim = int(
                ROOT_EMBEDDING_DIM * COMPONENT_WEIGHTS['root_embedding'] +
                WORD_EMBEDDING_DIM * COMPONENT_WEIGHTS['word_embedding'] +
                10 * COMPONENT_WEIGHTS['morphological_features']  # 10 morph features
            )
            return np.zeros(total_dim, dtype=np.float32)
        
        # Compute weights
        weights = self.compute_word_weights(annotations)
        
        # Collect embeddings for each word
        root_embeddings_list = []
        word_embeddings_list = []
        
        for ann in annotations:
            # Root embedding
            root_emb = self.embedding_db.get_root_embedding(ann.root)
            if root_emb is not None:
                root_embeddings_list.append(root_emb)
            else:
                # Fallback: zero vector
                root_embeddings_list.append(np.zeros(self.embedding_db.root_embeddings.shape[1]))
            
            # Word embedding (use transliterated form)
            word_emb = self.embedding_db.get_word_embedding(ann.word)
            if word_emb is not None:
                word_embeddings_list.append(word_emb)
            else:
                # Fallback: zero vector
                word_embeddings_list.append(np.zeros(self.embedding_db.word_embeddings.shape[1]))
        
        # Convert to arrays
        root_embeddings = np.array(root_embeddings_list)  # (num_words, root_dim)
        word_embeddings = np.array(word_embeddings_list)  # (num_words, word_dim)
        
        # Weighted average (THIS IS THE KEY - weights ensure semantic focus)
        verse_root_emb = np.average(root_embeddings, axis=0, weights=weights)
        verse_word_emb = np.average(word_embeddings, axis=0, weights=weights)
        
        # Get morphological features
        verse_morph_features = self.get_morphological_features(annotations)
        
        # Combine components with configured weights
        components = [
            verse_root_emb * COMPONENT_WEIGHTS['root_embedding'],
            verse_word_emb * COMPONENT_WEIGHTS['word_embedding'],
            verse_morph_features * COMPONENT_WEIGHTS['morphological_features']
        ]
        
        verse_embedding = np.concatenate(components)
        
        # Normalize (makes embeddings length-independent)
        if VERSE_NORMALIZATION == "l2":
            norm = np.linalg.norm(verse_embedding)
            if norm > 0:
                verse_embedding = verse_embedding / norm
        elif VERSE_NORMALIZATION == "max":
            max_val = np.abs(verse_embedding).max()
            if max_val > 0:
                verse_embedding = verse_embedding / max_val
        # else: no normalization
        
        return verse_embedding
    
    
    def build_all_verse_embeddings(self, verses: List[tuple]) -> Dict[str, np.ndarray]:
        """
        Build embeddings for all verses.
        
        Args:
            verses: List of (verse_id, verse_text) tuples
        
        Returns:
            Dictionary mapping verse_id → embedding
        """
        print("\n" + "="*70)
        print("COMPOSING VERSE EMBEDDINGS")
        print("="*70)
        
        print(f"\nProcessing {len(verses)} verses...")
        print(f"  Composition method: {VERSE_COMPOSITION_METHOD}")
        print(f"  Weighting scheme: {WORD_WEIGHT_SCHEME}")
        print(f"  Normalization: {VERSE_NORMALIZATION}")
        
        for verse_id, verse_text in verses:
            verse_emb = self.compose_verse_embedding(verse_id, verse_text)
            self.verse_embeddings[verse_id] = verse_emb
            
            # Store metadata
            annotations = self.parser.get_annotations(verse_id)
            self.verse_metadata[verse_id] = {
                'length': len(annotations),
                'num_unique_roots': len(set(ann.root for ann in annotations)),
                'embedding_dim': len(verse_emb)
            }
        
        # Report statistics
        all_embeddings = np.array(list(self.verse_embeddings.values()))
        
        # Pre-normalization diversity check (compute pairwise distances)
        sample_indices = np.random.choice(len(all_embeddings), min(100, len(all_embeddings)), replace=False)
        sample_embs = all_embeddings[sample_indices]
        pairwise_sims = sample_embs @ sample_embs.T
        avg_similarity = (pairwise_sims.sum() - len(sample_embs)) / (len(sample_embs) * (len(sample_embs) - 1))
        
        print(f"\n✓ Composed {len(self.verse_embeddings)} verse embeddings")
        print(f"  Embedding dimension: {all_embeddings.shape[1]}")
        print(f"  Mean norm: {np.linalg.norm(all_embeddings, axis=1).mean():.4f}")
        print(f"  Std norm: {np.linalg.norm(all_embeddings, axis=1).std():.4f}")
        print(f"  Average cosine similarity (sample): {avg_similarity:.4f}")
        print(f"    (Lower = more diverse; 0.0-0.3 is good, >0.7 is concerning)")
        
        return self.verse_embeddings
    
    
    def calculate_similarity(self, verse_id1: str, verse_id2: str) -> float:
        """
        Calculate cosine similarity between two verses.
        
        Returns value in [0, 1] where:
        - 1.0 = identical semantic content
        - 0.0 = completely unrelated
        """
        emb1 = self.verse_embeddings.get(verse_id1)
        emb2 = self.verse_embeddings.get(verse_id2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    
    def find_similar_verses(self, verse_id: str, top_k: int = 10) -> List[tuple]:
        """
        Find most semantically similar verses.
        
        Returns:
            List of (verse_id, similarity) tuples
        """
        query_emb = self.verse_embeddings.get(verse_id)
        if query_emb is None:
            return []
        
        # Compute similarities to all other verses
        similarities = []
        for other_id, other_emb in self.verse_embeddings.items():
            if other_id == verse_id:
                continue
            
            sim = self.calculate_similarity(verse_id, other_id)
            similarities.append((other_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    
    def save(self, filepath: str):
        """Save all verse embeddings."""
        data = {
            'verse_embeddings': self.verse_embeddings,
            'verse_metadata': self.verse_metadata
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    
    @staticmethod
    def load(filepath: str, embedding_db, corpus_parser) -> 'VerseEmbeddingComposer':
        """Load saved verse embeddings."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        composer = VerseEmbeddingComposer(embedding_db, corpus_parser)
        composer.verse_embeddings = data['verse_embeddings']
        composer.verse_metadata = data['verse_metadata']
        
        return composer


if __name__ == "__main__":
    print("Verse embedding composer")
    print("This should be used via the main pipeline")