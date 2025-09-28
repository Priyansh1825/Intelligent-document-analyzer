import streamlit as st
import PyPDF2
from docx import Document
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

st.set_page_config(page_title="Simple Document Analyzer", layout="wide")

st.title("📊 Simple Document Analyzer")
st.write("Upload a document (PDF, DOCX, or TXT) to analyze its content")

uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt'])

if uploaded_file:
    # Read file content
    text = ""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    else:  # txt file
        text = uploaded_file.getvalue().decode("utf-8")
    
    if st.button("Analyze Document"):
        # Basic statistics
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Word Count", len(words))
        with col2:
            st.metric("Sentence Count", len(sentences))
        with col3:
            st.metric("Character Count", len(text))
        
        # Sentiment analysis
        blob = TextBlob(text)
        with col4:
            sentiment = "Positive" if blob.sentiment.polarity > 0 else "Negative" if blob.sentiment.polarity < 0 else "Neutral"
            st.metric("Sentiment", sentiment)
        
        # Keywords
        stop_words = set(stopwords.words('english'))
        words_clean = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
        freq_dist = FreqDist(words_clean)
        
        st.subheader("Top Keywords")
        keywords = freq_dist.most_common(10)
        df_keywords = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
        st.dataframe(df_keywords)
        
        # Word cloud
        st.subheader("Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        # Simple summary (first 3 sentences)
        st.subheader("Summary")
        summary = " ".join(sentences[:3])
        st.write(summary)