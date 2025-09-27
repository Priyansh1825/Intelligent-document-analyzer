import string
import re
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    import nltk # type: ignore
    from nltk.tokenize import sent_tokenize, word_tokenize # type: ignore
    from nltk.corpus import stopwords # type: ignore
    from nltk.probability import FreqDist # type: ignore
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Some features disabled.")

try:
    from textblob import TextBlob # type: ignore
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available. Sentiment analysis disabled.")

class TextAnalyzer:
    def __init__(self):
        self.stop_words = set()
        self._setup_nltk()
    
    def _setup_nltk(self):
        """Setup NLTK data with error handling"""
        if not NLTK_AVAILABLE:
            return
            
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception:
                print("NLTK punkt download failed")
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            try:
                nltk.download('stopwords', quiet=True)
            except Exception:
                print("NLTK stopwords download failed")
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'])
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()
    
    def _simple_tokenize(self, text: str):
        """Simple tokenization if NLTK is not available"""
        sentences = text.split('.')
        words = text.split()
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        words = [w.strip() for w in words if w.strip()]
        return sentences, words
    
    def basic_statistics(self, text: str) -> dict:
        """Calculate basic text statistics"""
        if not text:
            return {
                'word_count': 0,
                'sentence_count': 0,
                'character_count': 0,
                'average_word_length': 0,
                'average_sentence_length': 0
            }
            
        cleaned_text = self.clean_text(text)
        
        if NLTK_AVAILABLE:
            try:
                words = word_tokenize(cleaned_text)
                sentences = sent_tokenize(cleaned_text)
            except Exception:
                sentences, words = self._simple_tokenize(cleaned_text)
        else:
            sentences, words = self._simple_tokenize(cleaned_text)
        
        # Filter out punctuation
        words = [word for word in words if word not in string.punctuation]
        
        word_count = len(words)
        sentence_count = len(sentences) if sentences else 1
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'character_count': len(cleaned_text),
            'average_word_length': sum(len(word) for word in words) / word_count if word_count else 0,
            'average_sentence_length': word_count / sentence_count if sentence_count else 0
        }
    
    def keyword_extraction(self, text: str, top_n: int = 10) -> list:
        """Extract top keywords using frequency analysis"""
        if not text:
            return []
            
        cleaned_text = self.clean_text(text)
        
        if NLTK_AVAILABLE:
            try:
                words = word_tokenize(cleaned_text.lower())
            except Exception:
                words = cleaned_text.lower().split()
        else:
            words = cleaned_text.lower().split()
        
        # Remove stopwords and punctuation
        words = [word for word in words 
                if word not in self.stop_words and word not in string.punctuation and len(word) > 2]
        
        if not words:
            return []
        
        # Calculate frequency
        if NLTK_AVAILABLE:
            try:
                freq_dist = FreqDist(words)
                return freq_dist.most_common(top_n)
            except Exception:
                pass
        
        # Fallback frequency calculation
        from collections import Counter
        return Counter(words).most_common(top_n)
    
    def sentiment_analysis(self, text: str) -> dict:
        """Perform sentiment analysis"""
        if not TEXTBLOB_AVAILABLE or not text:
            return {
                'polarity': 0,
                'subjectivity': 0,
                'sentiment': 'neutral'
            }
        
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            polarity = sentiment.polarity
            if polarity > 0.1:
                sentiment_label = 'positive'
            elif polarity < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return {
                'polarity': polarity,
                'subjectivity': sentiment.subjectivity,
                'sentiment': sentiment_label
            }
        except Exception:
            return {
                'polarity': 0,
                'subjectivity': 0,
                'sentiment': 'neutral'
            }
    
    def readability_score(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        if not text:
            return 0
            
        stats = self.basic_statistics(text)
        word_count = stats['word_count']
        sentence_count = stats['sentence_count']
        
        if sentence_count == 0 or word_count == 0:
            return 0
        
        avg_sentence_length = word_count / sentence_count
        syllables = self._count_syllables(text)
        avg_syllables_per_word = syllables / word_count if word_count else 0
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return round(max(0, min(100, score)), 2)  # Clamp between 0 and 100
    
    def _count_syllables(self, text: str) -> int:
        """Approximate syllable count"""
        if not text:
            return 0
            
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        
        if text:
            if text[0] in vowels:
                count += 1
            for index in range(1, len(text)):
                if text[index] in vowels and text[index-1] not in vowels:
                    count += 1
            if text.endswith('e'):
                count -= 1
            if count == 0:
                count += 1
        return count