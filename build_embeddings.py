#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_embeddings.py - Main pipeline for building Quran semantic embeddings

This orchestrates the entire process:
1. Parse Quran text and extract morphology
2. Build co-occurrence matrices with PPMI weighting
3. Apply SVD to get dense embeddings
4. Compose verse-level embeddings
5. Validate results

Usage:
    python build_embeddings.py --quran quran.xml
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Import our modules
from config import (
    validate_config, OUTPUT_DIR, WORD_EMBEDDING_METHOD,
    ROOT_EMBEDDING_DIM, WORD_EMBEDDING_DIM
)
from corpus_parser import QuranCorpusParser
from cooccurrence import CooccurrenceMatrixBuilder
from svd_embeddings import build_embeddings_from_matrices
from word2vec_hybrid import build_word2vec_embeddings
from verse_composer import VerseEmbeddingComposer


def load_quran_verses(quran_xml_path: str):
    """
    Load Quran verses from XML file.
    
    Returns:
        List of (verse_id, verse_text) tuples
    """
    print(f"\nLoading Quran from {quran_xml_path}...")
    
    # Import the verse lookup system
    try:
        from verse_lookup_system import QuranLookupSystem
        quran = QuranLookupSystem(quran_xml_path)
    except ImportError as e:
        print("Error: Cannot import verse_lookup_system")
        print("Make sure verse_lookup_system.py is in the same directory")
        print(f"Import error details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading Quran: {e}")
        print(f"Make sure {quran_xml_path} exists and is valid")
        sys.exit(1)
    
    # Extract all verses
    verses = []
    for position in range(1, 6237):  # 6236 verses in Quran
        verse = quran.get_verse_by_position(position)
        if verse and verse.text:
            # Use correct attribute names: surah_index and verse_index
            verse_id = f"{verse.surah_index}:{verse.verse_index}"
            verses.append((verse_id, verse.text))
    
    print(f"✓ Loaded {len(verses)} verses")
    return verses


def build_pipeline(quran_xml_path: str, corpus_file_path: str):
    """
    Main pipeline: build all embeddings from scratch.
    
    This is the complete process from raw text to semantic embeddings.
    """
    print("\n" + "="*70)
    print("QURAN SEMANTIC EMBEDDING PIPELINE")
    print("="*70)
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/cooccurrence_matrices", exist_ok=True)
    
    # Validate configuration
    validate_config()
    
    # ========================================================================
    # STEP 1: Load Quran text
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: LOADING QURAN TEXT")
    print("="*70)
    
    verses = load_quran_verses(quran_xml_path)
    
    # ========================================================================
    # STEP 2: Parse morphology from Corpus file
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: PARSING QURAN CORPUS MORPHOLOGY")
    print("="*70)
    
    # Parser automatically loads and parses the Corpus file on initialization
    parser = QuranCorpusParser(corpus_file_path)
    
    stats = parser.get_statistics()
    print(f"\n✓ Morphology extraction complete:")
    print(f"  Total words: {stats['total_words']:,}")
    print(f"  Unique words: {stats['unique_words']:,}")
    print(f"  Unique roots: {stats['unique_roots']:,}")
    avg_words_per_root = stats['total_words'] / stats['unique_roots'] if stats['unique_roots'] > 0 else 0
    print(f"  Avg words per root: {avg_words_per_root:.1f}")
    
    # Save parser
    parser.save(f"{OUTPUT_DIR}/corpus_parser.pkl")
    
    # ========================================================================
    # STEP 3: Build co-occurrence matrices
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: BUILDING CO-OCCURRENCE MATRICES")
    print("="*70)
    
    cooc_builder = CooccurrenceMatrixBuilder(parser)
    root_ppmi, word_ppmi = cooc_builder.build_all_matrices(verses)
    
    # Save matrices
    from scipy import sparse
    sparse.save_npz(f"{OUTPUT_DIR}/cooccurrence_matrices/root_ppmi.npz", root_ppmi)
    sparse.save_npz(f"{OUTPUT_DIR}/cooccurrence_matrices/word_ppmi.npz", word_ppmi)
    cooc_builder.save(f"{OUTPUT_DIR}/cooccurrence_builder.pkl")
    
    # ========================================================================
    # STEP 4: Build word embeddings (SVD or Word2Vec)
    # ========================================================================
    print("\n" + "="*70)
    print(f"STEP 4: BUILDING EMBEDDINGS (method: {WORD_EMBEDDING_METHOD})")
    print("="*70)
    
    if WORD_EMBEDDING_METHOD == "word2vec":
        # Use Word2Vec for word embeddings (hybrid word+root context)
        print("\n[Root Embeddings] Using SVD on root PPMI matrix")
        from svd_embeddings import SVDEmbedding, EmbeddingDatabase
        import numpy as np
        
        # Build root embeddings using SVD (same as original method)
        root_svd = SVDEmbedding(n_components=ROOT_EMBEDDING_DIM)
        root_embeddings = root_svd.fit_transform(root_ppmi)  # Only pass matrix!
        
        print("\n[Word Embeddings] Using Word2Vec with hybrid context")
        # Build word embeddings using Word2Vec
        word_embeddings_dict, word_vocab = build_word2vec_embeddings(
            corpus_parser=parser,
            verses=verses
        )
        
        # Convert dict to matrix (in vocab order)
        word_embeddings = np.array([word_embeddings_dict[w] for w in word_vocab])
        
        # Create embedding database
        embedding_db = EmbeddingDatabase(
            root_embeddings=root_embeddings,
            root_vocab=cooc_builder.root_vocab,
            word_embeddings=word_embeddings,
            word_vocab=word_vocab
        )
        
    else:  # "svd"
        # Use SVD for both root and word embeddings (original method)
        print("\n[Root & Word Embeddings] Using SVD on PPMI matrices")
        embedding_db = build_embeddings_from_matrices(
            root_ppmi=root_ppmi,
            word_ppmi=word_ppmi,
            root_vocab=cooc_builder.root_vocab,
            word_vocab=cooc_builder.word_vocab
        )
    
    # Save embeddings
    embedding_db.save(f"{OUTPUT_DIR}/embedding_db.npz")
    
    # ========================================================================
    # STEP 5: Compose verse embeddings
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: COMPOSING VERSE EMBEDDINGS")
    print("="*70)
    
    composer = VerseEmbeddingComposer(embedding_db, parser)
    verse_embeddings = composer.build_all_verse_embeddings(verses)
    
    # Save verse embeddings
    composer.save(f"{OUTPUT_DIR}/verse_embeddings.pkl")
    
    # Also save as NPZ for easy loading
    import numpy as np
    verse_ids = list(verse_embeddings.keys())
    verse_matrix = np.array([verse_embeddings[vid] for vid in verse_ids])
    np.savez_compressed(
        f"{OUTPUT_DIR}/verse_embeddings.npz",
        verse_ids=np.array(verse_ids, dtype=object),
        verse_matrix=verse_matrix
    )
    
    # ========================================================================
    # STEP 6: Validation
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: VALIDATION")
    print("="*70)
    
    run_validation(composer, embedding_db)
    
    # ========================================================================
    # COMPLETE
    # ========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    print(f"  - corpus_parser.pkl")
    print(f"  - embedding_db.npz")
    print(f"  - verse_embeddings.pkl")
    print(f"  - verse_embeddings.npz")
    
    return composer, embedding_db, parser


def run_validation(composer: VerseEmbeddingComposer, embedding_db):
    """
    Run validation tests to verify embeddings are working correctly.
    
    Tests:
    1. Root similarity (are mercy-related roots similar?)
    2. Verse similarity (are thematically related verses similar?)
    3. Length independence (are long/short verses handled fairly?)
    """
    print("\n--- Test 1: Root Similarity ---")
    
    # Test mercy-related roots (MUST use transliterated forms from Corpus)
    mercy_roots = ['rHm', 'gfr', 'Efw']  # mercy, forgiveness, pardoning
    
    print("Mercy-related roots (transliterated from Corpus):")
    for root in mercy_roots:
        similar = embedding_db.find_similar_roots(root, top_k=5)
        print(f"\n  {root} → similar roots:")
        for sim_root, sim_score in similar:
            print(f"    {sim_root}: {sim_score:.3f}")
    
    print("\n--- Test 2: Verse Similarity ---")
    
    # Test verse with رحم root
    test_verse = "2:163"  # "And your God is one God. There is no deity except Him, the Entirely Merciful, the Especially Merciful."
    
    similar_verses = composer.find_similar_verses(test_verse, top_k=10)
    
    print(f"\nVerses similar to {test_verse}:")
    for verse_id, sim_score in similar_verses:
        print(f"  {verse_id}: {sim_score:.3f}")
    
    print("\n--- Test 3: Length Independence ---")
    
    # Test verses of different lengths
    test_verses = [
        "103:1",  # Very short (3 words)
        "2:255",  # Very long (Ayat al-Kursi, ~50 words)
        "99:7",   # Medium (8 words)
    ]
    
    print("\nEmbedding norms (should be similar despite length differences):")
    for verse_id in test_verses:
        emb = composer.verse_embeddings.get(verse_id)
        metadata = composer.verse_metadata.get(verse_id)
        if emb is not None and metadata is not None:
            norm = np.linalg.norm(emb)
            print(f"  {verse_id}: length={metadata['length']}, norm={norm:.4f}")
    
    print("\n✓ Validation complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build semantic embeddings for Quran verses"
    )
    parser.add_argument(
        "--quran",
        type=str,
        default="quran.xml",
        help="Path to Quran XML file"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="quranic-corpus-morphology-0.4.txt",
        help="Path to Quran Corpus morphology TSV file (download from corpus.quran.com)"
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.quran):
        print(f"Error: Quran file not found: {args.quran}")
        sys.exit(1)
    
    if not os.path.exists(args.corpus):
        print("\n" + "="*70)
        print("ERROR: QURAN CORPUS FILE NOT FOUND")
        print("="*70)
        print(f"\nLooking for: {args.corpus}")
        print("\nThis file is REQUIRED for accurate root extraction.")
        print("\nHow to get it:")
        print("  1. Go to: http://corpus.quran.com/download/")
        print("  2. Look for: 'Morphological Segmentation (Text Format)'")
        print("  3. Download: quranic-corpus-morphology-0.4.txt")
        print("  4. Place it in the same directory as this script")
        print("\nOr specify a different path with:")
        print(f"  python build_embeddings.py --corpus /path/to/corpus.txt")
        print("="*70)
        sys.exit(1)
    
    # Run pipeline
    build_pipeline(args.quran, args.corpus)


if __name__ == "__main__":
    import numpy as np  # For validation
    main()