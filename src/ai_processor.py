from typing import Dict

class AIProcessor:
    def __init__(self):
        self.summarizer = None
        self.qa_pipeline = None
        self._setup_pipelines()
    
    def _setup_pipelines(self):
        try:
            from transformers import pipeline
            self.summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn"
            )
            
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad"
            )
        except Exception as e:
            print(f"AI models not available: {e}")
            print("Using fallback text analysis only")
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        if not self.summarizer:
            # Simple fallback summary - first 3 sentences
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            return " ".join(sentences[:3]) if len(sentences) > 3 else text[:200] + "..."
        
        if len(text) > 1024:
            text = text[:1024]
        
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            return f"Summary: {text[:200]}..."
    
    def answer_question(self, context: str, question: str) -> Dict:
        if not self.qa_pipeline:
            return {
                "answer": "Question-answering requires additional AI models. Please install transformers library.",
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