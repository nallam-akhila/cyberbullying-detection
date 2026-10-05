import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import time
from datetime import datetime
import pandas as pd
import numpy as np
from lime.lime_text import LimeTextExplainer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from collections import Counter

# Configure Streamlit page
st.set_page_config(
    page_title="Cyberbullying & Sentiment Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for a modern, card-based UI
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Prediction result box */
    .result-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid transparent;
    }
    .cyberbullying {
        background-color: #FFF0F0;
        border-color: #F44336;
        color: #c62828;
    }
    .not-cyberbullying {
        background-color: #F1FFF1;
        border-color: #4CAF50;
        color: #2e7d32;
    }
    /* Custom style for Streamlit's metric containers */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid #E0E0E0;
    }
    /* Card styling */
    .custom-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid #E0E0E0;
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the BERT model and tokenizer (cached for performance)"""
    try:
        MODEL_PATH = "folder_to_zip/bert_cyberbullying"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

@st.cache_resource
def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

def analyze_sentiment(text, analyzer):
    """Analyze text sentiment using VADER"""
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = "Positive"
        emoji = "😊"
    elif compound <= -0.05:
        sentiment = "Negative"
        emoji = "😠"
    else:
        sentiment = "Neutral"
        emoji = "😐"
        
    return {
        'sentiment': sentiment,
        'emoji': emoji,
        'compound_score': compound,
        'scores': scores
    }

def predict_cyberbullying(text, model, tokenizer, device):
    """Predict if text contains cyberbullying"""
    try:
        inputs = tokenizer(str(text), truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            confidence = probabilities[0][predicted_class].item()
        
        is_cyberbullying = predicted_class.item() == 1
        return {
            'is_cyberbullying': is_cyberbullying,
            'confidence': confidence,
            'cyberbullying_probability': probabilities[0][1].item(),
            'not_cyberbullying_probability': probabilities[0][0].item(),
            'prediction': "🚨 Cyberbullying Detected" if is_cyberbullying else "✅ No Cyberbullying Detected"
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def lime_predictor(texts, model, tokenizer, device):
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1)
    return probabilities.cpu().numpy()

def generate_lime_explanation(text, model, tokenizer, device):
    predictor_fn = lambda texts: lime_predictor(texts, model, tokenizer, device)
    explainer = LimeTextExplainer(class_names=['Not Cyberbullying', 'Cyberbullying'])
    explanation = explainer.explain_instance(text, predictor_fn, num_features=10, labels=(1,))
    return explanation

def create_probability_chart(cyberbullying_prob, not_cyberbullying_prob):
    """Create probability comparison chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Not Cyberbullying', 'Cyberbullying'],
            y=[not_cyberbullying_prob * 100, cyberbullying_prob * 100],
            marker_color=['#4caf50', '#f44336'],
            text=[f"{not_cyberbullying_prob*100:.1f}%", f"{cyberbullying_prob*100:.1f}%"],
            textposition='auto',
        )
    ])
    fig.update_layout(title_text="<b>Cyberbullying Prediction Probabilities</b>", height=400)
    return fig

def create_sentiment_chart(scores):
    """Create a bar chart for sentiment scores"""
    sent_labels = ['Positive', 'Neutral', 'Negative']
    sent_values = [scores['pos'], scores['neu'], scores['neg']]
    colors = ['#4CAF50', '#FFC107', '#F44336']
    
    fig = go.Figure(data=[
        go.Bar(
            x=sent_labels,
            y=[v * 100 for v in sent_values],
            marker_color=colors,
            text=[f"{v*100:.1f}%" for v in sent_values],
            textposition='auto'
        )
    ])
    fig.update_layout(title_text="<b>Sentiment Score Breakdown</b>", yaxis_title="Percentage (%)", height=400)
    return fig

def generate_word_cloud(explanation, text):
    """Generate a word cloud highlighting negative words based on LIME explanation"""
    # Get the list of words and their weights from LIME explanation
    word_weights = [(word, weight) for word, weight in explanation.as_list(label=1)]
    
    # Filter for negative words (those with positive weights for cyberbullying)
    negative_words = {word: abs(weight) for word, weight in word_weights if weight > 0}
    
    if not negative_words:
        # If no negative words found, create a default message
        negative_words = {"No negative words detected": 1}
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='Reds',
        max_words=50
    ).generate_from_frequencies(negative_words)
    
    # Display the word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Negative Words Highlighted by LIME', fontsize=16)
    
    return fig

def main():
    st.markdown('<div class="main-header">🛡️ Advanced Cyberbullying & Sentiment Analysis</div>', unsafe_allow_html=True)
    
    with st.spinner("Loading AI models..."):
        model, tokenizer, device = load_model()
        sentiment_analyzer = get_sentiment_analyzer()
    
    st.success(f"✅ Models loaded successfully! Using device: {device}")
    
    

    st.header("🔍 Text Analysis")
    user_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Type or paste text here...",
        height=150
    )

    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing text..."):
                start_time = time.time()
                bullying_result = predict_cyberbullying(user_input, model, tokenizer, device)
                sentiment_result = analyze_sentiment(user_input, sentiment_analyzer)
                
                # Generate LIME explanation
                lime_explanation = generate_lime_explanation(user_input, model, tokenizer, device)
                lime_html = lime_explanation.as_html()
                
                processing_time = time.time() - start_time
            
            if bullying_result and sentiment_result:
                st.header("Results")
                
                # Main prediction card
                css_class = "cyberbullying" if bullying_result['is_cyberbullying'] else "not-cyberbullying"
                st.markdown(f"""
                <div class="result-box {css_class}">
                    <h3>{bullying_result['prediction']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(label="🎯 Confidence", value=f"{bullying_result['confidence']:.1%}")
                with col2:
                    st.metric(
                        label=f"{sentiment_result['emoji']} Sentiment", 
                        value=sentiment_result['sentiment'],
                        delta=f"Score: {sentiment_result['compound_score']:.2f}"
                    )
                with col3:
                    st.metric(label="⚡ Processing Time", value=f"{processing_time:.3f}s")
                with col4:
                    st.metric(label="📝 Text Length", value=f"{len(user_input)} chars")
                
                # Charts in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="custom-card"><div class="card-title">Cyberbullying Probability</div></div>', unsafe_allow_html=True)
                    prob_fig = create_probability_chart(
                        bullying_result['cyberbullying_probability'], 
                        bullying_result['not_cyberbullying_probability']
                    )
                    st.plotly_chart(prob_fig, use_container_width=True)
                
                with col2:
                    st.markdown('<div class="custom-card"><div class="card-title">Sentiment Analysis</div></div>', unsafe_allow_html=True)
                    sentiment_fig = create_sentiment_chart(sentiment_result['scores'])
                    st.plotly_chart(sentiment_fig, use_container_width=True)
                
                # LIME Explanation and Word Cloud
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="custom-card"><div class="card-title">LIME Explanation</div></div>', unsafe_allow_html=True)
                    st.components.v1.html(lime_html, height=400, scrolling=True)
                    st.info("""
                    **How to read this chart:** This explains the prediction for the **"Cyberbullying"** category.
                    - **:green[Green bars]** support the prediction.
                    - **:red[Red bars]** contradict the prediction.
                    """)
                
                with col2:
                    st.markdown('<div class="custom-card"><div class="card-title">Negative Words Word Cloud</div></div>', unsafe_allow_html=True)
                    wordcloud_fig = generate_word_cloud(lime_explanation, user_input)
                    st.pyplot(wordcloud_fig)
                    st.info("""
                    **Word Cloud Explanation:** This visualization highlights words that contribute to cyberbullying detection.
                    Larger words have a stronger impact on the prediction.
                    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        🛡️ Powered by BERT & VADER | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

