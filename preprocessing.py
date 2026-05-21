"""
Enhanced Text Preprocessing Pipeline for RescueText PH
=====================================================

This module provides a comprehensive text preprocessing pipeline designed for
bilingual English-Filipino disaster-response text classification. It implements:

1. Regex-based cleaning (URLs, mentions, hashtags)
2. Explicit tokenization using NLTK word_tokenize
3. Combined English and Tagalog stopword removal
4. English lemmatization (WordNetLemmatizer)
5. Tagalog stemming (custom suffix removal for Filipino text)
6. Sequential processing: English lemmatization -> Tagalog stemming

The preprocessing is applied uniformly to all text (no language detection),
ensuring consistent behavior during both training and inference.

Dependencies:
    - nltk (tokenization, lemmatization, stopwords)
    - re (regex patterns)
    - functools (caching)

Usage:
    from preprocessing import PreprocessedText, preprocess_text
    
    processor = PreprocessedText()
    result = processor.preprocess_text("Sample Filipino/English text with #hashtag")
    print(result.tokens)  # Cleaned and tokenized text
    print(result.lemmatized_tokens)  # After English lemmatization
    print(result.stemmed_tokens)  # After Tagalog stemming
"""

import re
import string
from typing import List, Tuple, NamedTuple
from functools import lru_cache, wraps
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (safe to call multiple times)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


# ==============================================================================
# TAGALOG STOPWORDS & STEMMERS (Bilingual Support)
# ==============================================================================

# Comprehensive Tagalog/Filipino stopwords list
TAGALOG_STOPWORDS = {
    'ang', 'ang', 'ay', 'at', 'ito', 'sa', 'ko', 'mo', 'natin', 'nila', 'na',
    'pa', 'ng', 'de', 'ka', 'nin', 'rin', 'man', 'lang', 'ho', 'po',
    'daw', 'raw', 'kasi', 'kung', 'bakit', 'ano', 'sino', 'saan', 'kailan',
    'paano', 'aba', 'abot', 'abay', 'apod', 'apoy', 'araw', 'arkila',
    'awit', 'ayaw', 'babae', 'baha', 'bahagi', 'bahay', 'baki', 'bakit',
    'bala', 'bale', 'balita', 'balo', 'banal', 'banda', 'bande', 'bandera',
    'banding', 'bandita', 'bane', 'bangga', 'bangis', 'banig', 'banilad',
    'banyo', 'banyuhay', 'baon', 'bapor', 'bara', 'barabara', 'baraha',
    'barandal', 'barang', 'barangay', 'barania', 'barantayador', 'barao',
    'baraso', 'barato', 'barbela', 'barbero', 'barbo', 'barda', 'bardo',
    'bare', 'bareblog', 'barel', 'barena', 'barenilla', 'barensye',
    'barenyador', 'bares', 'baresy', 'baret', 'bareyador', 'bareyera',
    'bari', 'barilia', 'barilo', 'barima', 'barimba', 'barimbag', 'barimbas',
    'barimbay', 'barimbon', 'barimetro', 'barinado', 'barinador',
    'namin', 'ninyo', 'ninyong', 'para', 'pero', 'dapat',
    'kailangan', 'maari', 'maaari', 'ganito', 'ganyan', 'gaya',
    'tulad', 'katulad', 'kapareho', 'kapantay', 'kasingkasi',
    'lalo', 'lalong', 'lahat', 'kahit', 'man', 'lamang', 'bilang',
    'isang', 'dalawa', 'tatlo', 'marami', 'kaunti', 'iilan',
    'isa', 'iba', 'ibang', 'igit', 'ika', 'ikaw', 'ikong', 'ikot',
    'ilad', 'ilak', 'ilalim', 'ilang', 'ilangkasa', 'ilangilang'
}

# Common English stopwords (from NLTK, extended)
ENGLISH_STOPWORDS = set(stopwords.words('english'))

# Combined stopword set for unified preprocessing
COMBINED_STOPWORDS = ENGLISH_STOPWORDS.union(TAGALOG_STOPWORDS)


def tagalog_stemmer(word: str) -> str:
    """
    Simple Tagalog/Filipino stemmer using suffix removal heuristics.
    
    This is a lightweight rule-based stemmer that removes common Filipino
    affixes (prefixes and suffixes) to extract the root word.
    
    Rules applied (in order):
    1. Remove common suffixes: -ang, -ng, -in, -an, -on
    2. Remove diminutives: -ito, -ita, -ete
    3. Remove agent nouns: -or, -er, -ist
    4. Remove adjective suffixes: -ado, -ido, -able
    
    Args:
        word (str): Word to stem
        
    Returns:
        str: Stemmed word (or original if no rules apply)
        
    Note:
        This is not linguistically perfect but serves well for
        information retrieval and text classification tasks.
    """
    if len(word) < 4:
        return word
    
    # Remove common suffixes (order matters - try longest first)
    suffixes = [
        ('anga', 3),
        ('ahan', 3),
        ('tion', 3),  # For borrowed words
        ('sion', 3),
        ('ment', 3),
        ('able', 3),
        ('ible', 3),
        ('ang', 3),
        ('ado', 3),
        ('ido', 3),
        ('ing', 3),
        ('ito', 2),
        ('ita', 2),
        ('ete', 2),
        ('ern', 2),
        ('ern', 2),
        ('ist', 2),
        ('ism', 2),
        ('ess', 2),
        ('ous', 2),
        ('ive', 2),
        ('ful', 2),
        ('less', 3),
        ('ling', 2),
        ('or', 2),
        ('er', 2),
        ('in', 2),
        ('an', 2),
        ('on', 2),
        ('ng', 2),
        ('y', 1),
    ]
    
    for suffix, min_len in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_len:
            return word[:-len(suffix)]
    
    return word


class PreprocessedText(NamedTuple):
    """
    Result container for preprocessed text at different stages.
    
    Attributes:
        original_text (str): Original input text (not modified)
        cleaned_text (str): After regex cleaning (URLs, mentions, hashtags removed)
        tokens (List[str]): After tokenization and lowercasing
        cleaned_tokens (List[str]): After stopword removal
        lemmatized_tokens (List[str]): After English lemmatization
        stemmed_tokens (List[str]): After Tagalog stemming (final output)
        token_count (int): Number of tokens after cleaning
    """
    original_text: str
    cleaned_text: str
    tokens: List[str]
    cleaned_tokens: List[str]
    lemmatized_tokens: List[str]
    stemmed_tokens: List[str]
    token_count: int


class TextPreprocessor:
    """
    Comprehensive bilingual text preprocessor for disaster-response text.
    
    This class handles the complete preprocessing pipeline:
    1. Regex cleaning (URLs, mentions, hashtags)
    2. Tokenization (word-level)
    3. Stopword removal (English + Tagalog)
    4. English lemmatization
    5. Tagalog stemming
    
    The pipeline is deterministic and language-agnostic (treats Filipino and
    English identically through the same sequence of transformations).
    
    Example:
        >>> processor = TextPreprocessor()
        >>> result = processor.preprocess_text("Check out http://example.com #hashtag")
        >>> print(result.stemmed_tokens)
        ['check', 'example', 'hashtag']
    """
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True,
                 stem: bool = True, max_vocab_size: int = 20000):
        """
        Initialize the preprocessor.
        
        Args:
            remove_stopwords (bool): Whether to remove English + Tagalog stopwords (default: True)
            lemmatize (bool): Whether to apply English lemmatization (default: True)
            stem (bool): Whether to apply Tagalog stemming (default: True)
            max_vocab_size (int): Maximum vocabulary size for tokenization (default: 20000)
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        self.max_vocab_size = max_vocab_size
        self.lemmatizer = WordNetLemmatizer()
        
        # Punctuation for removal (exclude apostrophe for contractions)
        self.punctuation = set(string.punctuation)
        self.punctuation.discard("'")
    
    def _clean_regex(self, text: str) -> str:
        """
        Apply regex-based cleaning to remove URLs, mentions, and normalize hashtags.
        
        Operations:
        - Remove URLs (http://, https://, www.)
        - Remove mentions (@username)
        - Remove hashtag symbol but keep text (#hashtag -> hashtag)
        - Normalize multiple whitespaces to single space
        - Strip leading/trailing whitespace
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtag symbol but keep content (#FloodPH -> FloodPH)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words using NLTK word_tokenize.
        
        Args:
            text (str): Cleaned text to tokenize
            
        Returns:
            List[str]: List of tokens (lowercase)
        """
        tokens = word_tokenize(text.lower())
        return tokens
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove English and Tagalog stopwords from token list.
        
        Args:
            tokens (List[str]): Tokens to filter
            
        Returns:
            List[str]: Tokens with stopwords and punctuation removed
        """
        if not self.remove_stopwords:
            return [t for t in tokens if t not in self.punctuation]
        
        filtered = [
            t for t in tokens
            if t not in COMBINED_STOPWORDS and t not in self.punctuation
        ]
        return filtered
    
    def _lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Apply English lemmatization to tokens.
        
        Uses NLTK's WordNetLemmatizer to convert words to their base form
        (e.g., "running" -> "run", "better" -> "good").
        
        For non-English or unknown words, returns original token.
        
        Args:
            tokens (List[str]): Tokens to lemmatize
            
        Returns:
            List[str]: Lemmatized tokens
        """
        if not self.lemmatize:
            return tokens
        
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def _stem(self, tokens: List[str]) -> List[str]:
        """
        Apply Tagalog stemming to tokens.
        
        Applies the Tagalog stemmer to all tokens uniformly, enabling
        root word matching for both English and Filipino text.
        
        Args:
            tokens (List[str]): Tokens to stem
            
        Returns:
            List[str]: Stemmed tokens
        """
        if not self.stem:
            return tokens
        
        return [tagalog_stemmer(t) for t in tokens]
    
    def preprocess_text(self, text: str) -> PreprocessedText:
        """
        Execute the complete preprocessing pipeline on input text.
        
        Pipeline stages:
        1. Regex cleaning (URLs, mentions, hashtags)
        2. Tokenization (NLTK word_tokenize)
        3. Lowercasing (applied in tokenization)
        4. Stopword removal (English + Tagalog)
        5. English lemmatization
        6. Tagalog stemming
        
        Args:
            text (str): Raw input text (any language mix)
            
        Returns:
            PreprocessedText: Named tuple containing all processing stages
            
        Example:
            >>> processor = TextPreprocessor()
            >>> result = processor.preprocess_text("I hate this #racism http://link.com")
            >>> print(result.stemmed_tokens)
            ['hate', 'racism']
        """
        # Store original
        original = text
        
        # Stage 1: Regex cleaning
        cleaned = self._clean_regex(text)
        
        # Stage 2: Tokenization
        tokens = self._tokenize(cleaned)
        
        # Stage 3: Stopword removal
        cleaned_tokens = self._remove_stopwords(tokens)
        
        # Stage 4: English lemmatization
        lemmatized = self._lemmatize(cleaned_tokens)
        
        # Stage 5: Tagalog stemming
        stemmed = self._stem(lemmatized)
        
        return PreprocessedText(
            original_text=original,
            cleaned_text=cleaned,
            tokens=tokens,
            cleaned_tokens=cleaned_tokens,
            lemmatized_tokens=lemmatized,
            stemmed_tokens=stemmed,
            token_count=len(stemmed)
        )
    
    def preprocess_batch(self, texts: List[str]) -> List[PreprocessedText]:
        """
        Preprocess multiple texts efficiently.
        
        Args:
            texts (List[str]): List of texts to preprocess
            
        Returns:
            List[PreprocessedText]: List of preprocessed results
        """
        return [self.preprocess_text(text) for text in texts]
    
    def get_tokens_as_string(self, text: str) -> str:
        """
        Preprocess text and return final tokens as space-separated string.
        
        Convenience method for quick token string extraction.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Final stemmed tokens joined by spaces
        """
        result = self.preprocess_text(text)
        return ' '.join(result.stemmed_tokens)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

# Global preprocessor instance (lazy initialization)
_default_processor = None


def get_default_processor() -> TextPreprocessor:
    """
    Get or create the default preprocessor instance (singleton pattern).
    
    Returns:
        TextPreprocessor: Global preprocessor instance
    """
    global _default_processor
    if _default_processor is None:
        _default_processor = TextPreprocessor()
    return _default_processor


def preprocess_text(text: str) -> PreprocessedText:
    """
    Preprocess a single text using the default preprocessor.
    
    Convenience function for one-off preprocessing tasks.
    
    Args:
        text (str): Input text
        
    Returns:
        PreprocessedText: Preprocessing result at all stages
    """
    return get_default_processor().preprocess_text(text)


def preprocess_texts(texts: List[str]) -> List[PreprocessedText]:
    """
    Preprocess multiple texts using the default preprocessor.
    
    Args:
        texts (List[str]): List of input texts
        
    Returns:
        List[PreprocessedText]: List of preprocessing results
    """
    return get_default_processor().preprocess_batch(texts)


def get_tokens_string(text: str) -> str:
    """
    Get final preprocessed tokens as space-separated string.
    
    Convenience function for direct token string retrieval.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Final tokens joined by spaces
    """
    return get_default_processor().get_tokens_as_string(text)


# ==============================================================================
# TESTING & VALIDATION
# ==============================================================================

if __name__ == '__main__':
    """
    Quick validation of the preprocessing pipeline.
    """
    processor = TextPreprocessor()
    
    # Test cases: English, Filipino, Mixed
    test_texts = [
        "Need rescue near the flooded bridge http://example.com @user #FloodPH",
        "Ang aming #TikTok ay puno ng cyberbullying laban sa LGBT community",
        "I absolutely hate this hindi ko maintindihan ang kanyang behavior #angry",
        "This is normal positive text without any issues whatsoever",
        "@mention123 #hashtag #another please visit www.link.com for more!!!"
    ]
    
    print("=" * 80)
    print("PREPROCESSING PIPELINE VALIDATION")
    print("=" * 80)
    
    for idx, text in enumerate(test_texts, 1):
        result = processor.preprocess_text(text)
        print(f"\n[Test {idx}]")
        print(f"Original:        {result.original_text}")
        print(f"Cleaned:         {result.cleaned_text}")
        print(f"Tokens:          {result.tokens}")
        print(f"Cleaned tokens:  {result.cleaned_tokens}")
        print(f"Lemmatized:      {result.lemmatized_tokens}")
        print(f"Stemmed (FINAL): {result.stemmed_tokens}")
        print(f"Token count:     {result.token_count}")
        print("-" * 80)
    
    print("\nPreprocessing pipeline validation complete!")
