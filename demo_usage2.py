#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_usage.py - Demonstrate how to use the built embeddings

This shows common use cases:
1. Find verses similar to a concept (e.g., "mercy")
2. Find semantically similar verses
3. Cluster verses by theme
4. Search for verses by semantic query
"""

import numpy as np
from typing import List, Tuple
import pickle

from svd_embeddings import EmbeddingDatabase
from verse_composer import VerseEmbeddingComposer
from corpus_parser import QuranCorpusParser
from config import OUTPUT_DIR


def load_system():
    """Load all components from saved files."""
    print("Loading embedding system...")
    
    # Load parser
    parser = QuranCorpusParser.load(f"{OUTPUT_DIR}/corpus_parser.pkl")
    
    # Load embeddings
    embedding_db = EmbeddingDatabase.load(f"{OUTPUT_DIR}/embedding_db.npz")
    
    # Load verse embeddings
    composer = VerseEmbeddingComposer.load(
        f"{OUTPUT_DIR}/verse_embeddings.pkl",
        embedding_db,
        parser
    )
    
    print("✓ System loaded")
    return composer, embedding_db, parser


def find_verses_about_concept(concept_roots: List[str],
                              composer: VerseEmbeddingComposer,
                              parser: QuranCorpusParser,
                              top_k: int = 20) -> List[Tuple[str, float]]:
    """
    Find verses related to a concept defined by roots.
    
    This is the key use case: Find verses about "mercy" even if they don't
    use the word mercy explicitly.
    
    Args:
        concept_roots: List of roots defining the concept (e.g., ['rHm', 'gfr'] for mercy)
        composer: Verse embedding composer
        parser: Corpus parser
        top_k: Number of results
    
    Returns:
        List of (verse_id, relevance_score) tuples
    """
    # Strategy: Find all verses containing any of these roots,
    # then find verses similar to those
    
    seed_verses = set()
    for root in concept_roots:
        # Get all verses containing this root
        for verse_id, annotations in parser.word_annotations.items():
            for ann in annotations:
                if ann.root == root:
                    seed_verses.add(verse_id)
                    break
    
    print(f"Found {len(seed_verses)} seed verses with concept roots")
    
    # Compute average embedding of seed verses
    seed_embeddings = []
    for verse_id in seed_verses:
        emb = composer.verse_embeddings.get(verse_id)
        if emb is not None:
            seed_embeddings.append(emb)
    
    if not seed_embeddings:
        return []
    
    concept_embedding = np.mean(seed_embeddings, axis=0)
    
    # Find verses most similar to this concept embedding
    similarities = []
    for verse_id, verse_emb in composer.verse_embeddings.items():
        # Cosine similarity
        sim = np.dot(concept_embedding, verse_emb) / (
            np.linalg.norm(concept_embedding) * np.linalg.norm(verse_emb) + 1e-10
        )
        similarities.append((verse_id, float(sim)))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def demo_mercy_search(composer, embedding_db, parser):
    """
    Demo: Find verses about mercy/forgiveness.
    
    This addresses your original problem: الغفور/الغفار and الرحمن
    should both come up in a mercy search.
    """
    print("\n" + "="*70)
    print("DEMO 1: Finding verses about MERCY")
    print("="*70)
    
    # Define mercy concept by roots (Corpus uses transliterated forms)
    mercy_roots = [
        'rHm',  # mercy (الرحمن، الرحيم، رحمة)
        'gfr',  # forgiveness (الغفور، الغفار، مغفرة)
        'Efw',  # pardoning (العفو، عفا)
    ]
    
    print(f"\nConcept roots: {mercy_roots}")
    print("(These are transliterated forms from Quran Corpus)")
    
    # Find related verses
    results = find_verses_about_concept(mercy_roots, composer, parser, top_k=20)
    
    print("\nTop 20 verses related to MERCY:")
    for i, (verse_id, score) in enumerate(results, 1):
        # Check which roots it has
        annotations = parser.get_annotations(verse_id)
        verse_roots = set(ann.root for ann in annotations if ann.root)
        
        has_mercy_roots = [r for r in mercy_roots if r in verse_roots]
        
        roots_display = ", ".join(has_mercy_roots) if has_mercy_roots else "related (no direct root)"
        print(f"{i:2d}. {verse_id:8s} | score: {score:.3f} | roots: {roots_display}")


def demo_verse_similarity(composer, embedding_db, parser):
    """
    Demo: Find verses similar to a specific verse.
    """
    print("\n" + "="*70)
    print("DEMO 2: Finding similar verses")
    print("="*70)
    
    # Test verse
    test_verse = "4:40"  # "Indeed, Allah does not do injustice, [even] as much as an atom's weight"
    
    print(f"\nQuery verse: {test_verse}")
    
    # Get verse text
    annotations = parser.get_annotations(test_verse)
    if annotations:
        verse_roots = [ann.root for ann in annotations]
        print(f"Roots in query: {verse_roots}")
    
    # Find similar
    similar = composer.find_similar_verses(test_verse, top_k=10)
    
    print("\nTop 10 most similar verses:")
    for i, (verse_id, sim) in enumerate(similar, 1):
        annotations = parser.get_annotations(verse_id)
        verse_roots = [ann.root for ann in annotations] if annotations else []
        print(f"{i:2d}. {verse_id:8s} | similarity: {sim:.3f}")


def demo_root_relationships(composer, embedding_db, parser):
    """
    Demo: Explore which roots are semantically related.
    """
    print("\n" + "="*70)
    print("DEMO 3: Root semantic relationships")
    print("="*70)
    
    # Test roots (transliterated forms from Corpus)
    test_roots = [
        ('rHm', 'mercy'),
        ('gfr', 'forgiveness'),
        ('Edl', 'justice'),
        ('ktb', 'writing'),
    ]
    
    for root, meaning in test_roots:
        print(f"\n{root} ({meaning}) → similar roots:")
        similar = embedding_db.find_similar_roots(root, top_k=8)
        for sim_root, sim_score in similar:
            print(f"  {sim_root}: {sim_score:.3f}")


def demo_length_independence(composer, embedding_db, parser):
    """
    Demo: Verify that verse length doesn't affect similarity unfairly.
    """
    print("\n" + "="*70)
    print("DEMO 4: Length independence verification")
    print("="*70)
    
    # Get verses of different lengths about similar themes
    test_sets = [
        {
            'theme': 'Divine attributes',
            'verses': ['112:1', '112:2', '112:3', '112:4', '2:255']  # Surah Ikhlas + Ayat al-Kursi
        },
        {
            'theme': 'Judgment/Justice',
            'verses': ['99:7', '99:8', '4:40', '21:47']
        }
    ]
    
    for test_set in test_sets:
        print(f"\nTheme: {test_set['theme']}")
        print("Verses and their lengths:")
        
        verses_data = []
        for verse_id in test_set['verses']:
            metadata = composer.verse_metadata.get(verse_id, {})
            length = metadata.get('length', 0)
            verses_data.append((verse_id, length))
            print(f"  {verse_id}: {length} words")
        
        # Compute pairwise similarities
        print("\nPairwise similarities (length should not matter):")
        for i, (vid1, len1) in enumerate(verses_data):
            for vid2, len2 in verses_data[i+1:]:
                sim = composer.calculate_similarity(vid1, vid2)
                print(f"  {vid1} ({len1}w) ↔ {vid2} ({len2}w): {sim:.3f}")


def main():
    """Run all demos."""
    # Load system
    composer, embedding_db, parser = load_system()
    
    # Run demos
    demo_mercy_search(composer, embedding_db, parser)
    demo_verse_similarity(composer, embedding_db, parser)
    demo_root_relationships(composer, embedding_db, parser)
    demo_length_independence(composer, embedding_db, parser)
    
    print("\n" + "="*70)
    print("Demos complete!")
    print("="*70)


if __name__ == "__main__":
    main()