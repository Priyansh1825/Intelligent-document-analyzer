import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from textblob import TextBlob
import string
import re

class TextAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self._setup_nltk()
    
    def _setup_nltk(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()
    
    def basic_statistics(self, text: str) -> dict:
        cleaned_text = self.clean_text(text)
        words = word_tokenize(cleaned_text)
        sentences = sent_tokenize(cleaned_text)
        
        words = [word for word in words if word not in string.punctuation]
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'character_count': len(cleaned_text),
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'average_sentence_length': len(words) / len(sentences) if sentences else 0
        }
    
    def keyword_extraction(self, text: str, top_n: int = 10) -> list:
        cleaned_text = self.clean_text(text)
        words = word_tokenize(cleaned_text.lower())
        
        words = [word for word in words 
                if word not in self.stop_words and word not in string.punctuation]
        
        freq_dist = FreqDist(words)
        return freq_dist.most_common(top_n)
    
    def sentiment_analysis(self, text: str) -> dict:
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        return {
            'polarity': sentiment.polarity,
            'subjectivity': sentiment.subjectivity,
            'sentiment': 'positive' if sentiment.polarity > 0.1 else 
                        'negative' if sentiment.polarity < -0.1 else 'neutral'
        }
    
    def readability_score(self, text: str) -> float:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        words = [word for word in words if word not in string.punctuation]
        
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        syllables = self._count_syllables(text)
        avg_syllables_per_word = syllables / len(words) if words else 0
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return round(score, 2)
    
    def _count_syllables(self, text: str) -> int:
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
    
    # Simple named entity recognition without spaCy
    def named_entity_recognition(self, text: str) -> dict:
        # This is a simple version that just looks for capitalized words
        words = word_tokenize(text)
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': []  # Geographical locations
        }
        
        # Simple pattern matching for entities
        for i, word in enumerate(words):
            if word.istitle() and len(word) > 1:
                if i > 0 and words[i-1] in ['Mr.', 'Mrs.', 'Ms.', 'Dr.']:
                    entities['PERSON'].append(word)
                elif word in ['Inc.', 'Corp.', 'Ltd.'] or (i > 0 and words[i-1].istitle()):
                    entities['ORG'].append(word)
                elif word in countries or cities:  # You could add lists of countries/cities
                    entities['GPE'].append(word)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
            
        return entities