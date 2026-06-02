#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
word2vec_hybrid.py - Hybrid Word2Vec with Word + Root Context

This replaces the weak SVD-based word embeddings with a neural Word2Vec model
that learns from BOTH word context and root context.

Key innovation: Instead of just predicting words from surrounding words,
we predict words from surrounding words AND their roots. This allows:
- Rare words to borrow signal from their root
- Morphological variants to cluster naturally
- Better handling of sparse data
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class HybridWord2Vec(nn.Module):
    """
    Word2Vec that uses both word and root contexts.
    
    Architecture:
        Input: Context items (words + roots)
        Hidden: Embedding layer
        Output: Target word
    
    This is like Skip-gram, but with enriched context.
    """
    
    def __init__(self, 
                 word_vocab_size: int,
                 root_vocab_size: int,
                 embedding_dim: int = 384,
                 negative_samples: int = 5):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        
        # Separate embeddings for words and roots
        # This allows them to live in the same semantic space
        # but maintain their distinct identities
        self.word_embeddings = nn.Embedding(word_vocab_size, embedding_dim)
        self.root_embeddings = nn.Embedding(root_vocab_size, embedding_dim)
        
        # Output layer (predict target word)
        self.output_embeddings = nn.Embedding(word_vocab_size, embedding_dim)
        
        # Initialize with small random values
        self.word_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.root_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.output_embeddings.weight.data.zero_()
    
    
    def forward(self, context_words, context_roots, target_word, negative_words):
        """
        Forward pass for Skip-gram with negative sampling.
        
        Args:
            context_words: Tensor of context word indices (batch_size, context_size)
            context_roots: Tensor of context root indices (batch_size, context_size)
            target_word: Tensor of target word index (batch_size,)
            negative_words: Tensor of negative sample indices (batch_size, num_negative)
        
        Returns:
            loss: Negative sampling loss
        """
        batch_size = target_word.size(0)
        
        # Get context representations
        # Average the embeddings of context words and their roots
        word_context = self.word_embeddings(context_words)  # (batch, context_size, emb_dim)
        root_context = self.root_embeddings(context_roots)  # (batch, context_size, emb_dim)
        
        # Combine word and root context (simple average)
        context = (word_context + root_context) / 2.0  # (batch, context_size, emb_dim)
        context = context.mean(dim=1)  # (batch, emb_dim)
        
        # Get target and negative embeddings
        target_emb = self.output_embeddings(target_word)  # (batch, emb_dim)
        negative_emb = self.output_embeddings(negative_words)  # (batch, num_negative, emb_dim)
        
        # Compute scores
        positive_score = (context * target_emb).sum(dim=1)  # (batch,)
        negative_scores = torch.bmm(negative_emb, context.unsqueeze(2)).squeeze(2)  # (batch, num_negative)
        
        # Negative sampling loss
        positive_loss = -torch.log(torch.sigmoid(positive_score) + 1e-10).mean()
        negative_loss = -torch.log(torch.sigmoid(-negative_scores) + 1e-10).sum(dim=1).mean()
        
        loss = positive_loss + negative_loss
        
        return loss
    
    
    def get_word_embedding(self, word_idx):
        """Get final embedding for a word."""
        return self.word_embeddings.weight[word_idx].detach().cpu().numpy()
    
    
    def get_all_word_embeddings(self):
        """Get all word embeddings as numpy array."""
        return self.word_embeddings.weight.detach().cpu().numpy()


class HybridWord2VecTrainer:
    """
    Trainer for Hybrid Word2Vec model.
    
    Handles:
    - Building training data from Quran verses
    - Creating context windows with words + roots
    - Negative sampling
    - Training loop
    """
    
    def __init__(self, 
                 corpus_parser,
                 embedding_dim: int = 384,
                 window_size: int = 5,
                 negative_samples: int = 5,
                 min_count: int = 1):
        
        self.parser = corpus_parser
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.min_count = min_count
        
        # Build vocabularies
        self._build_vocabularies()
        
        # Initialize model
        self.model = HybridWord2Vec(
            word_vocab_size=len(self.word_to_idx),
            root_vocab_size=len(self.root_to_idx),
            embedding_dim=embedding_dim,
            negative_samples=negative_samples
        )
        
        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Hybrid Word2Vec initialized:")
        print(f"  Word vocabulary: {len(self.word_to_idx)}")
        print(f"  Root vocabulary: {len(self.root_to_idx)}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Window size: {window_size}")
        print(f"  Device: {self.device}")
    
    
    def _build_vocabularies(self):
        """Build word and root vocabularies with frequency filtering."""
        print("\nBuilding vocabularies for Word2Vec...")
        
        word_counts = defaultdict(int)
        root_counts = defaultdict(int)
        
        # Count frequencies
        for verse_id, annotations in self.parser.word_annotations.items():
            for ann in annotations:
                word_counts[ann.word] += 1
                if ann.root:
                    root_counts[ann.root] += 1
        
        # Filter by minimum count
        self.word_vocab = [w for w, c in word_counts.items() if c >= self.min_count]
        self.root_vocab = [r for r, c in root_counts.items() if c >= self.min_count]
        
        # Create index mappings
        self.word_to_idx = {w: i for i, w in enumerate(self.word_vocab)}
        self.root_to_idx = {r: i for i, r in enumerate(self.root_vocab)}
        
        # Word frequency for negative sampling (Mikolov's subsampling)
        total_words = sum(word_counts.values())
        self.word_freq = np.array([
            word_counts[w] / total_words for w in self.word_vocab
        ])
        
        # Negative sampling distribution (word_freq^0.75)
        self.neg_sample_dist = self.word_freq ** 0.75
        self.neg_sample_dist /= self.neg_sample_dist.sum()
        
        print(f"  Word vocabulary: {len(self.word_vocab)}")
        print(f"  Root vocabulary: {len(self.root_vocab)}")
    
    
    def _extract_training_pairs(self, verses: List[Tuple[str, str]]):
        """
        Extract (context, target) pairs from verses.
        
        Returns:
            List of (context_words, context_roots, target_word) tuples
        """
        training_data = []
        
        print("\nExtracting training pairs...")
        for verse_id, verse_text in tqdm(verses):
            annotations = self.parser.get_annotations(verse_id)
            
            if len(annotations) < 2:
                continue
            
            # For each word in the verse
            for target_pos, target_ann in enumerate(annotations):
                # Skip if word not in vocabulary
                if target_ann.word not in self.word_to_idx:
                    continue
                
                target_word_idx = self.word_to_idx[target_ann.word]
                
                # Extract context window
                start = max(0, target_pos - self.window_size)
                end = min(len(annotations), target_pos + self.window_size + 1)
                
                context_words = []
                context_roots = []
                
                for ctx_pos in range(start, end):
                    if ctx_pos == target_pos:
                        continue
                    
                    ctx_ann = annotations[ctx_pos]
                    
                    # Add word to context
                    if ctx_ann.word in self.word_to_idx:
                        context_words.append(self.word_to_idx[ctx_ann.word])
                    
                    # Add root to context (THIS IS THE KEY!)
                    if ctx_ann.root and ctx_ann.root in self.root_to_idx:
                        context_roots.append(self.root_to_idx[ctx_ann.root])
                
                # Skip if no context
                if not context_words and not context_roots:
                    continue
                
                # Pad to fixed size for batching
                while len(context_words) < self.window_size * 2:
                    context_words.append(0)  # Padding
                while len(context_roots) < self.window_size * 2:
                    context_roots.append(0)  # Padding
                
                training_data.append((
                    context_words[:self.window_size * 2],
                    context_roots[:self.window_size * 2],
                    target_word_idx
                ))
        
        print(f"  Extracted {len(training_data):,} training pairs")
        return training_data
    
    
    def train(self, 
              verses: List[Tuple[str, str]],
              epochs: int = 5,
              batch_size: int = 512,
              learning_rate: float = 0.025):
        """
        Train the Word2Vec model.
        
        Args:
            verses: List of (verse_id, verse_text) tuples
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
        """
        # Extract training data
        training_data = self._extract_training_pairs(verses)
        
        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"\nTraining Hybrid Word2Vec...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Shuffle data
            np.random.shuffle(training_data)
            
            # Train in batches
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                
                # Prepare batch tensors
                context_words = torch.LongTensor([x[0] for x in batch]).to(self.device)
                context_roots = torch.LongTensor([x[1] for x in batch]).to(self.device)
                target_words = torch.LongTensor([x[2] for x in batch]).to(self.device)
                
                # Sample negative words
                negative_words = np.random.choice(
                    len(self.word_vocab),
                    size=(len(batch), self.negative_samples),
                    p=self.neg_sample_dist
                )
                negative_words = torch.LongTensor(negative_words).to(self.device)
                
                # Forward pass
                loss = self.model(context_words, context_roots, target_words, negative_words)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        print("\n✓ Training complete!")
    
    
    def get_embeddings(self):
        """
        Get final word embeddings.
        
        Returns:
            Dictionary mapping word → embedding vector
        """
        all_embeddings = self.model.get_all_word_embeddings()
        
        return {
            word: all_embeddings[idx]
            for word, idx in self.word_to_idx.items()
        }
    
    
    def save_embeddings(self, filepath: str):
        """Save trained embeddings."""
        embeddings = self.get_embeddings()
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'word_vocab': self.word_vocab,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"✓ Saved Word2Vec embeddings to {filepath}")


def build_word2vec_embeddings(corpus_parser, 
                              verses: List[Tuple[str, str]]):
    """
    Main function to build Word2Vec embeddings.
    
    This replaces the SVD-based word embeddings with neural Word2Vec
    that uses hybrid word+root context.
    
    Args:
        corpus_parser: QuranCorpusParser instance
        verses: List of (verse_id, verse_text) tuples
    
    Returns:
        embeddings: Dictionary mapping word → embedding vector
        word_vocab: List of words in vocabulary
    """
    # Import config here to avoid circular imports
    from config import (
        WORD_EMBEDDING_DIM, WORD2VEC_WINDOW_SIZE, WORD2VEC_NEGATIVE_SAMPLES,
        WORD2VEC_EPOCHS, WORD2VEC_BATCH_SIZE, WORD2VEC_LEARNING_RATE,
        WORD2VEC_MIN_COUNT
    )
    
    print("\n" + "="*70)
    print("BUILDING WORD2VEC EMBEDDINGS (Hybrid Word+Root Context)")
    print("="*70)
    
    # Initialize trainer
    trainer = HybridWord2VecTrainer(
        corpus_parser=corpus_parser,
        embedding_dim=WORD_EMBEDDING_DIM,
        window_size=WORD2VEC_WINDOW_SIZE,
        negative_samples=WORD2VEC_NEGATIVE_SAMPLES,
        min_count=WORD2VEC_MIN_COUNT
    )
    
    # Train
    trainer.train(
        verses, 
        epochs=WORD2VEC_EPOCHS,
        batch_size=WORD2VEC_BATCH_SIZE,
        learning_rate=WORD2VEC_LEARNING_RATE
    )
    
    # Get embeddings
    embeddings = trainer.get_embeddings()
    
    print(f"\n✓ Built {len(embeddings)} word embeddings")
    print(f"  Dimension: {WORD_EMBEDDING_DIM}")
    print(f"  Method: Neural Word2Vec with hybrid word+root context")
    
    return embeddings, trainer.word_vocab


if __name__ == "__main__":
    print("Hybrid Word2Vec for Quranic text")
    print("This module should be used via the main pipeline")