"""
Quran Verse Identifier - Fuzzy Text Matcher
Identifies Surah and Ayat numbers from Arabic text even with:
- Typos/spelling errors
- Partial text
- OCR errors
- Missing diacritics (tashkeel)
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict


@dataclass
class VerseMatch:
    """Represents a matched verse with its details."""
    surah: int
    ayat: int
    text: str
    normalized_text: str
    similarity_score: float
    match_type: str  # 'exact', 'fuzzy', 'partial'
    length_ratio: float = 1.0  # 1.0 = same length as query, lower = different length


class ArabicNormalizer:
    """
    Normalizes Arabic text for comparison by removing diacritics,
    normalizing letter variants, and handling common OCR errors.
    """
    
    # Arabic diacritics (tashkeel) to remove
    DIACRITICS = (
        '\u064B',  # FATHATAN
        '\u064C',  # DAMMATAN
        '\u064D',  # KASRATAN
        '\u064E',  # FATHA
        '\u064F',  # DAMMA
        '\u0650',  # KASRA
        '\u0651',  # SHADDA
        '\u0652',  # SUKUN
        '\u0653',  # MADDAH ABOVE
        '\u0654',  # HAMZA ABOVE
        '\u0655',  # HAMZA BELOW
        '\u0656',  # SUBSCRIPT ALEF
        '\u0657',  # INVERTED DAMMA
        '\u0658',  # MARK NOON GHUNNA
        '\u0670',  # SUPERSCRIPT ALEF
        '\u06D6',  # SMALL HIGH LIGATURE SAD WITH LAM WITH ALEF MAKSURA
        '\u06D7',  # SMALL HIGH LIGATURE QAF WITH LAM WITH ALEF MAKSURA
        '\u06D8',  # SMALL HIGH MEEM INITIAL FORM
        '\u06D9',  # SMALL HIGH LAM ALEF
        '\u06DA',  # SMALL HIGH JEEM
        '\u06DB',  # SMALL HIGH THREE DOTS
        '\u06DC',  # SMALL HIGH SEEN
        '\u06DF',  # SMALL HIGH ROUNDED ZERO
        '\u06E0',  # SMALL HIGH UPRIGHT RECTANGULAR ZERO
        '\u06E1',  # SMALL HIGH DOTLESS HEAD OF KHAH
        '\u06E2',  # SMALL HIGH MEEM ISOLATED FORM
        '\u06E3',  # SMALL LOW SEEN
        '\u06E4',  # SMALL HIGH MADDA
        '\u06E5',  # SMALL WAW
        '\u06E6',  # SMALL YEH
        '\u06E7',  # SMALL HIGH YEH
        '\u06E8',  # SMALL HIGH NOON
        '\u06EA',  # EMPTY CENTRE LOW STOP
        '\u06EB',  # EMPTY CENTRE HIGH STOP
        '\u06EC',  # ROUNDED HIGH STOP WITH FILLED CENTRE
        '\u06ED',  # SMALL LOW MEEM
    )
    
    # Tatweel (kashida) - elongation character
    TATWEEL = '\u0640'
    
    # Letter normalization mappings (variant -> standard)
    LETTER_NORMALIZATIONS = {
        # Alef variants -> plain Alef
        '\u0622': '\u0627',  # ALEF WITH MADDA ABOVE -> ALEF
        '\u0623': '\u0627',  # ALEF WITH HAMZA ABOVE -> ALEF
        '\u0625': '\u0627',  # ALEF WITH HAMZA BELOW -> ALEF
        '\u0671': '\u0627',  # ALEF WASLA -> ALEF
        '\u0672': '\u0627',  # ALEF WITH WAVY HAMZA ABOVE -> ALEF
        '\u0673': '\u0627',  # ALEF WITH WAVY HAMZA BELOW -> ALEF
        '\u0675': '\u0627',  # HIGH HAMZA ALEF -> ALEF
        
        # Teh Marbuta -> Heh
        '\u0629': '\u0647',  # TEH MARBUTA -> HEH
        
        # Alef Maksura -> Yeh
        '\u0649': '\u064A',  # ALEF MAKSURA -> YEH
        
        # Waw with Hamza -> Waw
        '\u0624': '\u0648',  # WAW WITH HAMZA ABOVE -> WAW
        
        # Yeh with Hamza -> Yeh
        '\u0626': '\u064A',  # YEH WITH HAMZA ABOVE -> YEH
        
        # Persian/Urdu Yeh variants -> Arabic Yeh
        '\u06CC': '\u064A',  # ARABIC LETTER FARSI YEH -> YEH
        '\u06CD': '\u064A',  # ARABIC LETTER YEH WITH TAIL -> YEH
        '\u06CE': '\u064A',  # ARABIC LETTER YEH WITH SMALL V -> YEH
        '\u06D0': '\u064A',  # ARABIC LETTER E -> YEH
        '\u06D1': '\u064A',  # ARABIC LETTER YEH WITH THREE DOTS BELOW -> YEH
        
        # Persian/Urdu Kaf variants -> Arabic Kaf
        '\u06A9': '\u0643',  # ARABIC LETTER KEHEH -> KAF
        '\u06AA': '\u0643',  # ARABIC LETTER SWASH KAF -> KAF
        
        # Extended Arabic letters (common in some prints)
        '\u06C0': '\u0647',  # HEH WITH YEH ABOVE -> HEH
        '\u06C2': '\u0647',  # HEH GOAL WITH HAMZA ABOVE -> HEH
    }
    
    # Common OCR error mappings
    OCR_CORRECTIONS = {
        # Similar looking letters
        '\u0628': ['\u062A', '\u062B', '\u0646'],  # Ba can be confused with Ta, Tha, Noon
        '\u062A': ['\u0628', '\u062B'],  # Ta can be confused with Ba, Tha
        '\u062C': ['\u062D', '\u062E'],  # Jeem can be confused with Ha, Kha
        '\u062F': ['\u0630'],  # Dal can be confused with Thal
        '\u0631': ['\u0632'],  # Ra can be confused with Zay
        '\u0633': ['\u0634'],  # Seen can be confused with Sheen
        '\u0635': ['\u0636'],  # Sad can be confused with Dad
        '\u0637': ['\u0638'],  # Tah can be confused with Zah
        '\u0639': ['\u063A'],  # Ain can be confused with Ghain
        '\u0641': ['\u0642'],  # Fa can be confused with Qaf
        '\u0643': ['\u06A9'],  # Kaf variants
        '\u0647': ['\u0629'],  # Heh can be confused with Teh Marbuta
        '\u064A': ['\u0649'],  # Yeh can be confused with Alef Maksura
    }
    
    @classmethod
    def normalize(cls, text: str, aggressive: bool = False) -> str:
        """
        Normalize Arabic text for comparison.
        
        Args:
            text: Arabic text to normalize
            aggressive: If True, applies more aggressive normalization
        
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Remove non-Arabic characters except spaces (for partial matching)
        # Keep Arabic letters, numbers, and basic punctuation
        result = text
        
        # Remove diacritics
        for diacritic in cls.DIACRITICS:
            result = result.replace(diacritic, '')
        
        # Remove tatweel
        result = result.replace(cls.TATWEEL, '')
        
        # Normalize letter variants
        for variant, standard in cls.LETTER_NORMALIZATIONS.items():
            result = result.replace(variant, standard)
        
        # Remove extra whitespace
        result = ' '.join(result.split())
        
        if aggressive:
            # Remove all non-Arabic characters except spaces
            result = re.sub(r'[^\u0600-\u06FF\s]', '', result)
            # Remove spaces for exact character matching
            result = result.replace(' ', '')
        
        return result.strip()
    
    @classmethod
    def remove_diacritics_only(cls, text: str) -> str:
        """Remove only diacritics, keeping other characters."""
        result = text
        for diacritic in cls.DIACRITICS:
            result = result.replace(diacritic, '')
        return result


class FuzzyMatcher:
    """
    Fuzzy string matching engine with multiple comparison strategies.
    """
    
    @staticmethod
    def similarity_ratio(s1: str, s2: str) -> float:
        """
        Calculate similarity ratio between two strings using SequenceMatcher.
        Returns a value between 0.0 and 1.0.
        """
        if not s1 or not s2:
            return 0.0
        return SequenceMatcher(None, s1, s2).ratio()
    
    @staticmethod
    def partial_ratio(query: str, text: str) -> float:
        """
        Calculate best partial match ratio.
        Slides the shorter string over the longer one to find best match.
        """
        if not query or not text:
            return 0.0
        
        shorter = query if len(query) <= len(text) else text
        longer = text if len(query) <= len(text) else query
        
        if len(shorter) == 0:
            return 0.0
        
        # Slide window
        best_ratio = 0.0
        window_size = len(shorter)
        
        for i in range(len(longer) - window_size + 1):
            window = longer[i:i + window_size]
            ratio = SequenceMatcher(None, shorter, window).ratio()
            best_ratio = max(best_ratio, ratio)
        
        return best_ratio
    
    @staticmethod
    def contains_subsequence(query: str, text: str, min_match: float = 0.8) -> Tuple[bool, float]:
        """
        Check if query is a subsequence of text (with some tolerance for errors).
        Returns (is_match, confidence_score).
        """
        if not query or not text:
            return False, 0.0
        
        query_len = len(query)
        text_len = len(text)
        
        if query_len > text_len:
            return False, 0.0
        
        # Try to find best matching window
        best_score = 0.0
        for start in range(text_len - query_len + 1):
            window = text[start:start + query_len]
            score = SequenceMatcher(None, query, window).ratio()
            best_score = max(best_score, score)
            
            if best_score >= min_match:
                return True, best_score
        
        return best_score >= min_match, best_score
    
    @staticmethod
    def word_match_score(query_words: List[str], text_words: List[str]) -> float:
        """
        Calculate match score based on word-level matching.
        Useful for partial verse matching.
        """
        if not query_words or not text_words:
            return 0.0
        
        matched_words = 0
        for qword in query_words:
            best_word_match = 0.0
            for tword in text_words:
                ratio = SequenceMatcher(None, qword, tword).ratio()
                best_word_match = max(best_word_match, ratio)
            if best_word_match >= 0.8:  # Word matches with 80% similarity
                matched_words += 1
        
        return matched_words / len(query_words)


class QuranVerseIdentifier:
    """
    Main class for identifying Quran verses from Arabic text input.
    Uses n-gram indexing for fast candidate retrieval.
    """
    
    def __init__(self, quran_file_path: str = None, ngram_size: int = 3):
        """
        Initialize the verse identifier.
        
        Args:
            quran_file_path: Path to the quran-uthmani.txt file
            ngram_size: Size of n-grams for indexing (default: 3)
        """
        self.verses: List[Dict] = []
        self.normalized_verses: List[Dict] = []
        self.translations: Dict[tuple, str] = {}  # (surah, ayat) -> translation
        self.normalizer = ArabicNormalizer()
        self.matcher = FuzzyMatcher()
        self.ngram_size = ngram_size
        self.ngram_index: Dict[str, Set[int]] = defaultdict(set)
        
        if quran_file_path:
            self.load_quran(quran_file_path)
            # Auto-load English translation if available
            translation_path = Path(quran_file_path).parent / 'en.yusufali.txt'
            if translation_path.exists():
                self.load_translations(str(translation_path))
    
    def _get_ngrams(self, text: str) -> Set[str]:
        """Extract n-grams from text."""
        ngrams = set()
        text = text.replace(' ', '')
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.add(text[i:i + self.ngram_size])
        return ngrams
    
    def _build_index(self) -> None:
        """Build n-gram index for fast candidate retrieval."""
        self.ngram_index.clear()
        for idx, verse in enumerate(self.normalized_verses):
            ngrams = self._get_ngrams(verse['normalized_aggressive'])
            for ngram in ngrams:
                self.ngram_index[ngram].add(idx)
        print(f"Built index with {len(self.ngram_index)} unique n-grams")
    
    def _get_candidates(self, query: str, max_candidates: int = 100) -> List[int]:
        """Get candidate verse indices based on n-gram overlap."""
        query_ngrams = self._get_ngrams(query)
        
        if not query_ngrams:
            return list(range(min(max_candidates, len(self.normalized_verses))))
        
        # Count n-gram matches per verse
        candidate_scores: Dict[int, int] = defaultdict(int)
        for ngram in query_ngrams:
            if ngram in self.ngram_index:
                for idx in self.ngram_index[ngram]:
                    candidate_scores[idx] += 1
        
        # Sort by match count and return top candidates
        sorted_candidates = sorted(
            candidate_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return [idx for idx, _ in sorted_candidates[:max_candidates]]
    
    def load_quran(self, file_path: str) -> None:
        """
        Load Quran text from file.
        
        Args:
            file_path: Path to the quran text file (format: surah|ayat|text)
        """
        self.verses = []
        self.normalized_verses = []
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Quran file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('|')
                if len(parts) >= 3:
                    try:
                        surah = int(parts[0])
                        ayat = int(parts[1])
                        text = '|'.join(parts[2:])  # Handle text with | characters
                        
                        # Store original verse
                        self.verses.append({
                            'surah': surah,
                            'ayat': ayat,
                            'text': text,
                            'line': line_num
                        })
                        
                        # Store normalized version
                        normalized = self.normalizer.normalize(text)
                        normalized_aggressive = self.normalizer.normalize(text, aggressive=True)
                        
                        self.normalized_verses.append({
                            'surah': surah,
                            'ayat': ayat,
                            'text': text,
                            'normalized': normalized,
                            'normalized_aggressive': normalized_aggressive,
                            'words': normalized.split(),
                            'line': line_num
                        })
                    except ValueError:
                        print(f"Warning: Could not parse line {line_num}: {line[:50]}...")
        
        print(f"Loaded {len(self.verses)} verses from {file_path}")
        
        # Build n-gram index for fast lookup
        self._build_index()
    
    def load_translations(self, file_path: str) -> None:
        """
        Load translations from file.
        
        Args:
            file_path: Path to translation file (format: surah|ayat|text)
        """
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: Translation file not found: {file_path}")
            return
        
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('|')
                if len(parts) >= 3:
                    try:
                        surah = int(parts[0])
                        ayat = int(parts[1])
                        text = '|'.join(parts[2:])
                        self.translations[(surah, ayat)] = text
                        count += 1
                    except ValueError:
                        continue
        
        print(f"Loaded {count} translations from {file_path}")
    
    def get_translation(self, surah: int, ayat: int) -> Optional[str]:
        """Get translation for a specific verse."""
        return self.translations.get((surah, ayat))
    
    def identify(
        self, 
        query: str, 
        top_k: int = 5,
        min_similarity: float = 0.5,
        max_candidates: int = 100,
        prefer_similar_length: bool = True
    ) -> List[VerseMatch]:
        """
        Identify the Surah and Ayat for a given Arabic text query.
        Uses n-gram indexing for fast candidate retrieval.
        
        Args:
            query: Arabic text to search for
            top_k: Number of top matches to return
            min_similarity: Minimum similarity score threshold
            max_candidates: Max candidates to evaluate (for speed)
            prefer_similar_length: If True, prioritize verses with similar length to query
        
        Returns:
            List of VerseMatch objects sorted by similarity (highest first)
        """
        if not query or not self.normalized_verses:
            return []
        
        # Normalize the query
        query_normalized = self.normalizer.normalize(query)
        query_aggressive = self.normalizer.normalize(query, aggressive=True)
        query_len = len(query_aggressive)
        
        # Get candidate verses using n-gram index (fast filtering)
        candidate_indices = self._get_candidates(query_aggressive, max_candidates)
        
        results: List[VerseMatch] = []
        
        for idx in candidate_indices:
            verse = self.normalized_verses[idx]
            verse_len = len(verse['normalized_aggressive'])
            
            # Calculate length ratio (1.0 = same length, lower = more different)
            length_ratio = min(query_len, verse_len) / max(query_len, verse_len) if max(query_len, verse_len) > 0 else 0
            
            # Calculate similarity scores
            match_type = 'fuzzy'
            
            # 1. Exact match (after normalization) - fastest check
            if query_aggressive == verse['normalized_aggressive']:
                results.append(VerseMatch(
                    surah=verse['surah'],
                    ayat=verse['ayat'],
                    text=verse['text'],
                    normalized_text=verse['normalized'],
                    similarity_score=1.0,
                    match_type='exact',
                    length_ratio=1.0
                ))
                continue
            
            # 2. Full text similarity
            full_sim = self.matcher.similarity_ratio(
                query_aggressive, verse['normalized_aggressive']
            )
            
            # 3. Partial ratio (for partial text input) - only if query is shorter
            best_score = full_sim
            if len(query_aggressive) < len(verse['normalized_aggressive']):
                partial_sim = self.matcher.partial_ratio(
                    query_aggressive, verse['normalized_aggressive']
                )
                if partial_sim > best_score:
                    best_score = partial_sim
                    match_type = 'partial'
            
            if best_score >= min_similarity:
                results.append(VerseMatch(
                    surah=verse['surah'],
                    ayat=verse['ayat'],
                    text=verse['text'],
                    normalized_text=verse['normalized'],
                    similarity_score=best_score,
                    match_type=match_type,
                    length_ratio=length_ratio
                ))
        
        # Sort by similarity score, then by length similarity (prefer closer length)
        if prefer_similar_length:
            # Combined score: similarity * 0.7 + length_ratio * 0.3
            results.sort(key=lambda x: (x.similarity_score * 0.7 + x.length_ratio * 0.3), reverse=True)
        else:
            results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results[:top_k]
    
    def identify_with_context(
        self, 
        query: str, 
        top_k: int = 1
    ) -> List[Dict]:
        """
        Identify verses and return results with additional context.
        
        Args:
            query: Arabic text to search for
            top_k: Number of top matches to return
        
        Returns:
            List of dictionaries with verse info and context
        """
        matches = self.identify(query, top_k=top_k)
        
        results = []
        for match in matches:
            translation = self.get_translation(match.surah, match.ayat)
            result = {
                'surah': match.surah,
                'ayat': match.ayat,
                'reference': f"{match.surah}:{match.ayat}",
                'text': match.text,
                'translation': translation or "Translation not available",
                'similarity': round(match.similarity_score * 100, 2),
                'length_match': round(match.length_ratio * 100, 2),
                'match_type': match.match_type,
                'confidence': self._get_confidence_level(match.similarity_score)
            }
            results.append(result)
        
        return results
    
    def identify_multiple(
        self, 
        query: str, 
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """
        Identify multiple consecutive verses from a longer text input.
        Useful when user inputs several ayats together.
        
        Args:
            query: Arabic text containing one or more verses
            min_similarity: Minimum similarity threshold for each match
        
        Returns:
            List of matched verses in order of appearance
        """
        query_normalized = self.normalizer.normalize(query, aggressive=True)
        
        if not query_normalized:
            return []
        
        # Strategy: Find ALL verses that match the start of query
        # Then use the NEXT verse in query to disambiguate (for refrains like Ar-Rahman)
        
        # Get candidates that match the beginning of the query
        matching_candidates = []
        
        for prefix_len in [20, 30, 40, 50]:
            if prefix_len > len(query_normalized):
                prefix_len = len(query_normalized)
            
            query_prefix = query_normalized[:prefix_len]
            candidate_indices = self._get_candidates(query_prefix, max_candidates=100)
            
            for idx in candidate_indices:
                if idx >= len(self.normalized_verses):
                    continue
                verse_data = self.normalized_verses[idx]
                verse_normalized = verse_data['normalized_aggressive']
                
                # Check if query starts with this verse
                if query_normalized.startswith(verse_normalized):
                    matching_candidates.append((idx, 1.0, len(verse_normalized)))
                elif query_normalized.startswith(verse_normalized[:min(15, len(verse_normalized))]):
                    score = FuzzyMatcher.similarity_ratio(
                        verse_normalized, 
                        query_normalized[:len(verse_normalized)]
                    )
                    if score > 0.8:
                        matching_candidates.append((idx, score, len(verse_normalized)))
        
        if not matching_candidates:
            # Fall back to single verse identification
            single_result = self.identify_with_context(query, top_k=1)
            return single_result if single_result else []
        
        # If multiple candidates match (e.g., refrain verses), disambiguate using next verse
        best_candidate = None
        best_total_score = 0
        
        for idx, score, verse_len in matching_candidates:
            verse_data = self.normalized_verses[idx]
            surah = verse_data['surah']
            ayat = verse_data['ayat']
            
            # Check what comes after this verse in the query
            remaining = query_normalized[verse_len:].strip()
            
            if remaining:
                # Get the next verse in the Quran
                next_verse_text = self.get_verse(surah, ayat + 1)
                if next_verse_text:
                    next_normalized = self.normalizer.normalize(next_verse_text, aggressive=True)
                    
                    # Check if remaining query starts with or contains next verse
                    if remaining.startswith(next_normalized) or next_normalized in remaining:
                        # This is likely the correct verse!
                        total_score = score + 1.0
                    else:
                        # Check fuzzy match
                        compare_len = min(len(remaining), int(len(next_normalized) * 1.5))
                        next_sim = FuzzyMatcher.similarity_ratio(next_normalized, remaining[:compare_len])
                        total_score = score + next_sim
                else:
                    total_score = score
            else:
                total_score = score
            
            if total_score > best_total_score:
                best_total_score = total_score
                best_candidate = (idx, score, verse_len)
        
        if not best_candidate:
            # Fall back to first candidate
            best_candidate = matching_candidates[0]
        
        # Build results starting from best match
        results = []
        idx, first_score, verse_len = best_candidate
        verse_data = self.normalized_verses[idx]
        surah = verse_data['surah']
        ayat = verse_data['ayat']
        first_text = verse_data['text']
        first_normalized = verse_data['normalized_aggressive']
        
        # Calculate actual similarity for first match
        first_sim = FuzzyMatcher.similarity_ratio(
            first_normalized,
            query_normalized[:len(first_normalized)]
        )
        
        # Add first match
        translation = self.get_translation(surah, ayat)
        results.append({
            'surah': surah,
            'ayat': ayat,
            'reference': f"{surah}:{ayat}",
            'text': first_text,
            'translation': translation or "Translation not available",
            'similarity': round(first_sim * 100, 2),
            'length_match': 100.0,
            'match_type': 'exact' if first_sim > 0.95 else 'fuzzy',
            'confidence': self._get_confidence_level(first_sim)
        })
        
        # Remove first verse from query and look for consecutive verses
        remaining = query_normalized
        
        # Remove the first verse text
        if first_normalized in remaining:
            idx = remaining.find(first_normalized)
            remaining = remaining[idx + len(first_normalized):].strip()
        else:
            remaining = remaining[len(first_normalized):].strip()
        
        # Look for consecutive verses
        current_ayat = ayat
        for _ in range(50):  # Up to 50 more verses
            if not remaining:
                break
                
            next_ayat = current_ayat + 1
            next_verse_text = self.get_verse(surah, next_ayat)
            
            if not next_verse_text:
                break
            
            next_normalized = self.normalizer.normalize(next_verse_text, aggressive=True)
            
            # Check if remaining text starts with or contains this verse
            starts_with = remaining.startswith(next_normalized)
            contains = next_normalized in remaining
            
            if starts_with or contains:
                translation = self.get_translation(surah, next_ayat)
                results.append({
                    'surah': surah,
                    'ayat': next_ayat,
                    'reference': f"{surah}:{next_ayat}",
                    'text': next_verse_text,
                    'translation': translation or "Translation not available",
                    'similarity': 100.0,
                    'length_match': 100.0,
                    'match_type': 'exact',
                    'confidence': 'very_high'
                })
                
                # Remove this verse from remaining
                if contains:
                    idx = remaining.find(next_normalized)
                    remaining = remaining[idx + len(next_normalized):].strip()
                else:
                    remaining = remaining[len(next_normalized):].strip()
                
                current_ayat = next_ayat
            else:
                # Try fuzzy match - compare verse with beginning of remaining text
                # Use a larger window for comparison to account for orthographic variants
                compare_len = min(len(remaining), int(len(next_normalized) * 1.5))
                sim = FuzzyMatcher.similarity_ratio(
                    next_normalized, 
                    remaining[:compare_len]
                )
                
                if sim >= 0.6:
                    translation = self.get_translation(surah, next_ayat)
                    results.append({
                        'surah': surah,
                        'ayat': next_ayat,
                        'reference': f"{surah}:{next_ayat}",
                        'text': next_verse_text,
                        'translation': translation or "Translation not available",
                        'similarity': round(sim * 100, 2),
                        'length_match': 90.0,
                        'match_type': 'fuzzy',
                        'confidence': self._get_confidence_level(sim)
                    })
                    # Remove slightly more than verse length to account for variants
                    remove_len = int(len(next_normalized) * 1.1)
                    remaining = remaining[remove_len:].strip()
                    current_ayat = next_ayat
                else:
                    # No match - stop looking for consecutive verses
                    break
        
        return results
    
    def get_verse_range(
        self, 
        surah: int, 
        start_ayat: int, 
        end_ayat: int
    ) -> List[Dict]:
        """
        Get a range of verses from a specific Surah.
        
        Args:
            surah: Surah number (1-114)
            start_ayat: Starting ayat number
            end_ayat: Ending ayat number (inclusive)
        
        Returns:
            List of verses in the range
        """
        results = []
        for ayat in range(start_ayat, end_ayat + 1):
            text = self.get_verse(surah, ayat)
            if text:
                translation = self.get_translation(surah, ayat)
                results.append({
                    'surah': surah,
                    'ayat': ayat,
                    'reference': f"{surah}:{ayat}",
                    'text': text,
                    'translation': translation or "Translation not available"
                })
        return results
    
    def _get_confidence_level(self, score: float) -> str:
        """Convert similarity score to confidence level."""
        if score >= 0.95:
            return 'very_high'
        elif score >= 0.85:
            return 'high'
        elif score >= 0.70:
            return 'medium'
        elif score >= 0.55:
            return 'low'
        else:
            return 'very_low'
    
    def get_verse(self, surah: int, ayat: int) -> Optional[str]:
        """Get a specific verse by Surah and Ayat number."""
        for verse in self.verses:
            if verse['surah'] == surah and verse['ayat'] == ayat:
                return verse['text']
        return None
    
    def get_surah_info(self) -> Dict[int, int]:
        """Get information about all Surahs (number of verses each)."""
        surah_info = {}
        for verse in self.verses:
            surah = verse['surah']
            surah_info[surah] = surah_info.get(surah, 0) + 1
        return surah_info


def demo():
    """Demonstrate the Quran Verse Identifier."""
    import os
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    quran_file = os.path.join(script_dir, 'quran-uthmani.txt')
    
    print("=" * 60)
    print("Quran Verse Identifier - Demo")
    print("=" * 60)
    
    # Initialize identifier
    identifier = QuranVerseIdentifier(quran_file)
    
    # Test cases demonstrating different error types
    test_cases = [
        # Exact match
        ("بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ", "Exact match - Bismillah"),
        
        # Without diacritics
        ("بسم الله الرحمن الرحيم", "Without diacritics"),
        
        # Partial text
        ("ٱلْحَمْدُ لِلَّهِ", "Partial verse - Alhamdulillah"),
        
        # With some typos/errors
        ("الحمد لله رب العلمين", "Missing diacritics + slight variation"),
        
        # Another partial
        ("إِيَّاكَ نَعْبُدُ", "Partial - Iyyaka na'budu"),
        
        # Longer partial
        ("قُلْ هُوَ ٱللَّهُ أَحَدٌ", "Surah Ikhlas - first verse"),
    ]
    
    print("\n" + "-" * 60)
    print("Running test cases:")
    print("-" * 60)
    
    for query, description in test_cases:
        print(f"\n📖 Test: {description}")
        print(f"   Query: {query}")
        
        results = identifier.identify_with_context(query, top_k=1)
        
        if results:
            print("   Results:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. [{result['reference']}] "
                      f"Sim: {result['similarity']}% "
                      f"Len: {result['length_match']}% "
                      f"[{result['match_type']}]")
                # Show first 60 chars of Arabic text
                preview = result['text'][:60] + "..." if len(result['text']) > 60 else result['text']
                print(f"      Arabic: {preview}")
                # Show translation
                trans = result['translation'][:80] + "..." if len(result['translation']) > 80 else result['translation']
                print(f"      English: {trans}")
        else:
            print("   No matches found.")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
