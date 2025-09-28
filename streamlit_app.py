import streamlit as st
import os
import sys
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.document_reader import DocumentReader
from src.text_analyzer import TextAnalyzer
from src.ai_processor import AIProcessor

# Configure the page
st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DocumentAnalyzerWeb:
    def __init__(self):
        self.document_reader = DocumentReader()
        self.text_analyzer = TextAnalyzer()
        self.ai_processor = AIProcessor()
        
    def main(self):
        # Header
        st.markdown('<div class="main-header">📊 AI Document Analyzer</div>', unsafe_allow_html=True)
        
        # Sidebar for file upload
        with st.sidebar:
            st.header("📁 Document Upload")
            uploaded_file = st.file_uploader(
                "Choose a document",
                type=['pdf', 'docx', 'txt'],
                help="Upload PDF, DOCX, or TXT files"
            )
            
            st.header("⚙️ Analysis Options")
            show_wordcloud = st.checkbox("Generate Word Cloud", value=True)
            detailed_analysis = st.checkbox("Detailed Analysis", value=True)
            
            if uploaded_file:
                if st.button("🔍 Analyze Document", type="primary"):
                    return uploaded_file, show_wordcloud, detailed_analysis
            else:
                st.info("Please upload a document to begin analysis")
                
        return None, False, False

    def analyze_document(self, uploaded_file, show_wordcloud, detailed_analysis):
        # Save uploaded file temporarily
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Read document
            with st.spinner("📖 Reading document..."):
                document = self.document_reader.read_document(file_path)
            
            # Analyze document
            with st.spinner("🔍 Analyzing content..."):
                analysis = self.ai_processor.analyze_document(document['text'])
            
            # Display results
            self.display_results(document, analysis, show_wordcloud, detailed_analysis)
            
        except Exception as e:
            st.error(f"Error analyzing document: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

    def display_results(self, document, analysis, show_wordcloud, detailed_analysis):
        # Document metadata
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Word Count", analysis['basic_stats']['word_count'])
        with col2:
            st.metric("Sentence Count", analysis['basic_stats']['sentence_count'])
        with col3:
            st.metric("Readability Score", f"{analysis['readability']}/100")
        with col4:
            sentiment_emoji = "😊" if analysis['sentiment']['sentiment'] == 'positive' else "😐" if analysis['sentiment']['sentiment'] == 'neutral' else "😞"
            st.metric("Sentiment", f"{analysis['sentiment']['sentiment']} {sentiment_emoji}")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Basic Stats", "🔑 Keywords", "📝 Summary", "❓ Q&A"])
        
        with tab1:
            self.display_basic_stats(analysis)
            
        with tab2:
            self.display_keywords(analysis, show_wordcloud)
            
        with tab3:
            self.display_summary(analysis)
            
        with tab4:
            self.display_qa(document['text'])
    
    def display_basic_stats(self, analysis):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">📊 Text Statistics</div>', unsafe_allow_html=True)
            
            stats_data = {
                'Metric': ['Word Count', 'Sentence Count', 'Character Count', 
                          'Avg Word Length', 'Avg Sentence Length', 'Readability Score'],
                'Value': [
                    analysis['basic_stats']['word_count'],
                    analysis['basic_stats']['sentence_count'],
                    analysis['basic_stats']['character_count'],
                    round(analysis['basic_stats']['average_word_length'], 2),
                    round(analysis['basic_stats']['average_sentence_length'], 2),
                    analysis['readability']
                ]
            }
            
            df_stats = pd.DataFrame(stats_data)
            st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown('<div class="section-header">😊 Sentiment Analysis</div>', unsafe_allow_html=True)
            
            sentiment = analysis['sentiment']
            
            # Sentiment gauge
            st.metric("Polarity", f"{sentiment['polarity']:.3f}")
            st.metric("Subjectivity", f"{sentiment['subjectivity']:.3f}")
            
            # Sentiment interpretation
            if sentiment['polarity'] > 0.1:
                st.success("**Positive Sentiment**: The document has a positive tone.")
            elif sentiment['polarity'] < -0.1:
                st.error("**Negative Sentiment**: The document has a negative tone.")
            else:
                st.info("**Neutral Sentiment**: The document has a neutral tone.")
            
            # Subjectivity interpretation
            if sentiment['subjectivity'] > 0.6:
                st.info("**Subjective Content**: The document contains personal opinions.")
            else:
                st.info("**Objective Content**: The document presents factual information.")
    
    def display_keywords(self, analysis, show_wordcloud):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">🔑 Top Keywords</div>', unsafe_allow_html=True)
            
            keywords = analysis['keywords']
            keyword_data = {
                'Keyword': [kw[0] for kw in keywords],
                'Frequency': [kw[1] for kw in keywords]
            }
            
            df_keywords = pd.DataFrame(keyword_data)
            st.dataframe(df_keywords, use_container_width=True, hide_index=True)
        
        with col2:
            if show_wordcloud:
                st.markdown('<div class="section-header">☁️ Word Cloud</div>', unsafe_allow_html=True)
                self.generate_wordcloud(analysis['keywords'])
    
    def generate_wordcloud(self, keywords):
        # Create word cloud from keywords
        word_freq = {word: freq for word, freq in keywords}
        
        if word_freq:
            wordcloud = WordCloud(
                width=400, 
                height=300, 
                background_color='white',
                colormap='viridis'
            ).generate_from_frequencies(word_freq)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Not enough keywords to generate word cloud.")
    
    def display_summary(self, analysis):
        st.markdown('<div class="section-header">📝 AI Summary</div>', unsafe_allow_html=True)
        
        summary = analysis['summary']
        st.write(summary)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Summary Length", f"{len(summary.split())} words")
        with col2:
            original_words = analysis['basic_stats']['word_count']
            summary_words = len(summary.split())
            compression = ((original_words - summary_words) / original_words) * 100
            st.metric("Compression Rate", f"{compression:.1f}%")
    
    def display_qa(self, text):
        st.markdown('<div class="section-header">❓ Question & Answer</div>', unsafe_allow_html=True)
        
        question = st.text_input("Ask a question about the document:")
        
        if question and text:
            if st.button("Get Answer", type="primary"):
                with st.spinner("🤔 Thinking..."):
                    answer = self.ai_processor.answer_question(text, question)
                
                st.success("**Answer:** " + answer['answer'])
                st.info(f"Confidence: {answer['confidence']:.3f}")
        
        # Sample questions
        st.markdown("**💡 Try asking:**")
        sample_questions = [
            "What is this document about?",
            "What are the main topics discussed?",
            "What is the overall sentiment?",
            "Can you summarize the key points?"
        ]
        
        for q in sample_questions:
            if st.button(f"\"{q}\"", key=q):
                with st.spinner("🤔 Thinking..."):
                    answer = self.ai_processor.answer_question(text, q)
                st.success("**Answer:** " + answer['answer'])

def main():
    app = DocumentAnalyzerWeb()
    
    uploaded_file, show_wordcloud, detailed_analysis = app.main()
    
    if uploaded_file:
        app.analyze_document(uploaded_file, show_wordcloud, detailed_analysis)

if __name__ == "__main__":
    main()