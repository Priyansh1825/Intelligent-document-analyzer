from typing import Dict
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class AIProcessor:
    def __init__(self):
        self.summarizer = None
        self.qa_pipeline = None
        self._setup_pipelines()
    
    def _setup_pipelines(self):
        """Initialize AI models with error handling"""
        try:
            from transformers import pipeline # type: ignore
            # Use a smaller model for faster processing
            self.summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn"
            )
            
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                tokenizer="distilbert-base-cased-distilled-squad"
            )
        except ImportError as e:
            print(f"Transformers not available: {e}")
            print("Using fallback text analysis only")
        except Exception as e:
            print(f"Error loading AI models: {e}")
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        """Generate text summary"""
        if not self.summarizer:
            # Simple fallback summary - first 3 sentences
            try:
                from nltk.tokenize import sent_tokenize # type: ignore
                sentences = sent_tokenize(text)
                return " ".join(sentences[:3]) if len(sentences) > 3 else text[:200] + "..."
            except ImportError:
                return text[:200] + "..."
        
        if len(text) > 1024:
            text = text[:1024]
        
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            return f"Summary: {text[:200]}..."
    
    def answer_question(self, context: str, question: str) -> Dict:
        """Answer question based on document content"""
        if not self.qa_pipeline:
            return {
                "answer": "Question-answering requires additional AI models. Please install transformers: pip install transformers",
                "confidence": 0
            }
        
        try:
            result = self.qa_pipeline(question=question, context=context)
            return {
                "answer": result['answer'],
                "confidence": round(result['score'], 3)
            }
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "score": 0}
    
    def analyze_document(self, text: str) -> Dict:
        """Comprehensive document analysis"""
        from src.text_analyzer import TextAnalyzer
        
        analyzer = TextAnalyzer()
        
        analysis = {
            'basic_stats': analyzer.basic_statistics(text),
            'keywords': analyzer.keyword_extraction(text),
            'sentiment': analyzer.sentiment_analysis(text),
            'readability': analyzer.readability_score(text)
        }
        
        analysis['summary'] = self.summarize_text(text)
        
        return analysis