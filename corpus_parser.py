#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corpus_parser.py - Parse Quran Corpus TSV morphological file

Parses the REAL Quran Corpus data (not ISRI fallback).

Download from: http://corpus.quran.com/download/
File: quranic-corpus-morphology-0.4.txt

Format: Tab-separated with columns:
LOCATION    FORM    TAG    FEATURES

Example:
(1:1:1:1)   bi      P      PREFIX|bi+
(1:1:1:2)   somi    N      STEM|POS:N|LEM:{som|ROOT:smw|M|GEN

This gives us 95%+ accurate roots vs 70% with ISRI.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import unicodedata


@dataclass
class WordAnnotation:
    """Word-level annotation aggregated from morphemes."""
    verse_id: str
    word_num: int
    word: str          # Transliterated form
    root: Optional[str]
    pos: str
    gender: Optional[str]
    number: Optional[str]
    case: Optional[str]


class QuranCorpusParser:
    """
    Parse Quran Corpus TSV file for accurate morphological data.
    
    This replaces ISRI stemmer with real Corpus annotations.
    """
    
    def __init__(self, corpus_file: Optional[str] = None):
        # Storage
        self.word_annotations: Dict[str, List[WordAnnotation]] = defaultdict(list)
        self.word_to_root: Dict[str, str] = {}
        self.root_to_words: Dict[str, List[str]] = defaultdict(list)
        self.root_frequency: Dict[str, int] = defaultdict(int)
        
        # Stats
        self.total_words = 0
        
        if corpus_file:
            self.parse_corpus_file(corpus_file)
    
    
    def parse_location(self, loc: str) -> Tuple[int, int, int, int]:
        """Parse (chapter:verse:word:morpheme) → (1, 1, 1, 1)"""
        clean = loc.strip("()")
        parts = clean.split(":")
        return tuple(map(int, parts))
    
    
    def parse_features(self, features: str) -> Dict[str, str]:
        """
        Parse FEATURES column.
        
        Example: "STEM|POS:N|ROOT:smw|M|GEN"
        Returns: {'type': 'STEM', 'POS': 'N', 'ROOT': 'smw', 'gender': 'M', 'case': 'GEN'}
        """
        result = {}
        parts = features.split("|")
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if ":" in part:
                key, val = part.split(":", 1)
                result[key] = val
            else:
                # Standalone features
                if part in ['PREFIX', 'STEM', 'SUFFIX']:
                    result['type'] = part
                elif part in ['M', 'F']:
                    result['gender'] = part
                elif part in ['NOM', 'ACC', 'GEN']:
                    result['case'] = part
                elif part in ['S', 'D', 'P']:
                    result['number'] = part
                # Handle combined features like MS, MP, 2MS, etc.
                elif part == 'MS':
                    result['gender'] = 'M'
                    result['number'] = 'S'
                elif part == 'MP':
                    result['gender'] = 'M'
                    result['number'] = 'P'
                elif part == 'FP':
                    result['gender'] = 'F'
                    result['number'] = 'P'
        
        return result
    
    
    def parse_corpus_file(self, corpus_file: str):
        """Parse the Quran Corpus TSV file."""
        print(f"\nParsing Quran Corpus: {corpus_file}")
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header
        if lines and lines[0].startswith('LOCATION'):
            lines = lines[1:]
        
        print(f"Processing {len(lines)} morpheme annotations...")
        
        # Group morphemes by word
        words_dict = defaultdict(lambda: defaultdict(list))  # verse_id → word_num → [morphemes]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 4:
                continue
            
            location_str, form, tag, features_str = parts
            
            # Parse location
            try:
                chapter, verse, word_num, morpheme_num = self.parse_location(location_str)
            except:
                continue
            
            verse_id = f"{chapter}:{verse}"
            
            # Parse features
            features = self.parse_features(features_str)
            
            # Store morpheme data
            morpheme_data = {
                'form': form,
                'tag': tag,
                'features': features,
                'is_stem': (features.get('type') == 'STEM')
            }
            
            words_dict[verse_id][word_num].append(morpheme_data)
        
        # Aggregate morphemes into words
        print("Aggregating morphemes into words...")
        
        for verse_id, verse_words in words_dict.items():
            for word_num in sorted(verse_words.keys()):
                morphemes = verse_words[word_num]
                
                # Reconstruct word
                word_form = "".join(m['form'] for m in morphemes)
                
                # Find STEM morpheme (has root and POS)
                stem = None
                for m in morphemes:
                    if m['is_stem']:
                        stem = m
                        break
                
                if stem is None:
                    stem = morphemes[0]  # Fallback
                
                # Extract attributes from stem
                stem_features = stem['features']
                root = stem_features.get('ROOT')
                pos = stem_features.get('POS', stem['tag'])
                gender = stem_features.get('gender')
                number = stem_features.get('number')
                case = stem_features.get('case')
                
                # Create word annotation
                annotation = WordAnnotation(
                    verse_id=verse_id,
                    word_num=word_num,
                    word=word_form,
                    root=root,
                    pos=pos,
                    gender=gender,
                    number=number,
                    case=case
                )
                
                self.word_annotations[verse_id].append(annotation)
                self.total_words += 1
                
                # Cache mappings
                if root:
                    self.word_to_root[word_form] = root
                    self.root_to_words[root].append(word_form)
                    self.root_frequency[root] += 1
        
        print(f"✓ Parsed {self.total_words} words from {len(self.word_annotations)} verses")
        print(f"✓ Found {len(self.root_frequency)} unique roots")
    
    
    def get_verse_annotations(self, verse_id: str) -> List[WordAnnotation]:
        """Get all word annotations for a verse."""
        return self.word_annotations.get(verse_id, [])
    
    
    def get_annotations(self, verse_id: str) -> List[WordAnnotation]:
        """Alias for compatibility."""
        return self.get_verse_annotations(verse_id)
    
    
    def parse_verse_words(self, verse_id: str, verse_text: str) -> List[WordAnnotation]:
        """Get annotations for a verse."""
        return self.get_verse_annotations(verse_id)
    
    
    def get_root(self, word: str) -> Optional[str]:
        """Get root for a word (95%+ accurate from Corpus)."""
        return self.word_to_root.get(word)
    
    
    def get_statistics(self) -> Dict:
        """Get parsing statistics."""
        return {
            'total_words': self.total_words,
            'unique_words': len(self.word_to_root),
            'unique_roots': len(self.root_frequency),
        }
    
    
    def save(self, filepath: str):
        """Save parser state."""
        import pickle
        data = {
            'word_annotations': dict(self.word_annotations),
            'word_to_root': self.word_to_root,
            'root_to_words': dict(self.root_to_words),
            'root_frequency': dict(self.root_frequency),
            'statistics': self.get_statistics()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    
    @staticmethod
    def load(filepath: str) -> 'QuranCorpusParser':
        """Load saved parser state."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        parser = QuranCorpusParser()
        parser.word_annotations = defaultdict(list, data['word_annotations'])
        parser.word_to_root = data['word_to_root']
        parser.root_to_words = defaultdict(list, data['root_to_words'])
        parser.root_frequency = defaultdict(int, data['root_frequency'])
        parser.total_words = data['statistics']['total_words']
        
        return parser


if __name__ == "__main__":
    import sys
    
    corpus_file = sys.argv[1] if len(sys.argv) > 1 else "quranic-corpus-morphology-0.4.txt"
    
    print("="*70)
    print("TESTING QURAN CORPUS PARSER")
    print("="*70)
    
    try:
        parser = QuranCorpusParser(corpus_file)
        
        # Test Al-Fatiha
        print("\nTest: Surah Al-Fatiha (1:1)")
        annotations = parser.get_verse_annotations("1:1")
        
        print(f"Found {len(annotations)} words:")
        for ann in annotations:
            root_str = f"ROOT:{ann.root}" if ann.root else "no root"
            print(f"  {ann.word:15s} | {ann.pos:5s} | {root_str}")
        
        # Test mercy root
        print("\nTest: Mercy root (rHm)")
        mercy_words = parser.root_to_words.get('rHm', [])
        print(f"Found {len(mercy_words)} words:")
        for word in mercy_words[:5]:
            print(f"  - {word}")
        
        # Stats
        print("\nStatistics:")
        stats = parser.get_statistics()
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
    except FileNotFoundError:
        print(f"\nError: File not found: {corpus_file}")
        print("\nDownload from: http://corpus.quran.com/download/")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()