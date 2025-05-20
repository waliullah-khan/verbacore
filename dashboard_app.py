import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import patronus
from patronus.evals import RemoteEvaluator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set page config
st.set_page_config(
    page_title="AI Response Analysis Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

download_nltk_data()

# Set API key from .env or environment variable or st.secrets
def get_api_key():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("PATRONUS_API_KEY")
        
        # If not found in environment, check st.secrets
        if not api_key and hasattr(st, 'secrets') and 'PATRONUS_API_KEY' in st.secrets:
            api_key = st.secrets["PATRONUS_API_KEY"]
            
        return api_key
    except ImportError:
        if hasattr(st, 'secrets') and 'PATRONUS_API_KEY' in st.secrets:
            return st.secrets["PATRONUS_API_KEY"]
        return None  # Return None instead of hardcoded key

# Initialize Patronus
def init_patronus():
    api_key = get_api_key()
    if api_key:
        try:
            patronus.init(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Failed to initialize Patronus: {e}")
            return False
    else:
        st.warning("Patronus API key not found. Please set the PATRONUS_API_KEY environment variable or add it to your Streamlit secrets. Hallucination detection will not be available.")
        return False

# CSS to inject for styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-weight: bold;
        font-size: 3em;
        margin-bottom: 1em;
    }
    .section-title {
        font-weight: bold;
        font-size: 1.5em;
        margin: 1em 0;
    }
    .insight-box {
        background-color: #f1f1f1;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4c78a8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Create "pages"
def intro_page():
    st.markdown('<div class="main-title">AI Response Analysis Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the AI Response Analysis Dashboard
        
        This dashboard provides comprehensive analysis of conversation data from your AI assistants. 
        Upload your JSON file to analyze:
        
        - **Hallucination Detection**: Evaluate AI responses using Patronus
        - **Topic Modeling**: Discover main topics in conversations
        - **Sentiment Analysis**: Analyze emotional tone in messages
        - **Time Series Analysis**: Explore temporal patterns
        - **Text Embeddings & Clustering**: Group similar messages
        
        To get started, upload your JSON file in the sidebar and explore the various analysis tabs.
        """)
    
    with col2:
        st.image("https://img.freepik.com/free-vector/artificial-intelligence-concept-illustration_114360-7358.jpg?size=338&ext=jpg", use_container_width=True)

# Load and preprocess data
@st.cache_data
def load_data(file_uploaded):
    try:
        content = file_uploaded.getvalue().decode('utf-8')
        data = json.loads(content)
        df = pd.DataFrame(data)
        
        # Basic validation
        required_columns = ['content']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Required column '{col}' not found in the JSON data.")
                return None
        
        # Handle createdAt if it exists
        if 'createdAt' in df.columns:
            if isinstance(df.loc[0, 'createdAt'], dict) and '$date' in df.loc[0, 'createdAt']:
                df['createdAt'] = pd.to_datetime(df['createdAt'].apply(lambda x: x.get('$date', '')))
            else:
                try:
                    df['createdAt'] = pd.to_datetime(df['createdAt'])
                except:
                    st.warning("Could not parse 'createdAt' as datetime.")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Text preprocessing
@st.cache_data
def preprocess_text(texts):
    processed_texts = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, text in enumerate(texts):
        try:
            if not isinstance(text, str):
                text = str(text)
                
            # Convert to lowercase
            text = text.lower()
            
            # Basic cleaning
            text = ' '.join(text.split())  # Remove extra whitespace
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            
            # Filter and process tokens
            processed_tokens = []
            for token in tokens:
                if token.isalnum() and token not in stop_words and len(token) > 2:
                    try:
                        lemma = lemmatizer.lemmatize(token)
                        processed_tokens.append(lemma)
                    except:
                        processed_tokens.append(token)
            
            processed_texts.append(' '.join(processed_tokens))
            
            # Update progress
            progress = (i + 1) / len(texts)
            progress_bar.progress(progress)
            status_text.text(f"Processing text: {i+1}/{len(texts)}")
            
        except Exception as e:
            st.warning(f"Error processing text {i}: {str(e)}")
            processed_texts.append("")
    
    status_text.empty()
    progress_bar.empty()
    
    return processed_texts

# Find message pairs
@st.cache_data
def find_message_pairs(df):
    # Check if we have sender column to identify candidate/bot messages
    if 'sender' not in df.columns:
        st.warning("No 'sender' column found. Cannot identify message pairs.")
        return []
    
    # Group messages by thread ID if available
    if 'botThreadId' in df.columns:
        # Convert botThreadId if it's in MongoDB format
        if isinstance(df.loc[0, 'botThreadId'], dict) and '$oid' in df.loc[0, 'botThreadId']:
            df['thread_id'] = df['botThreadId'].apply(lambda x: x.get('$oid', ''))
        else:
            df['thread_id'] = df['botThreadId']
        
        # Group by thread
        thread_messages = df.groupby('thread_id')
        
        pairs = []
        for thread_id, messages in thread_messages:
            # Sort by creation time if available
            if 'createdAt' in messages.columns:
                messages = messages.sort_values('createdAt')
            
            # Find candidate-bot pairs
            messages_list = messages.to_dict('records')
            for i, msg in enumerate(messages_list):
                if msg.get('sender') == 'candidate' and i + 1 < len(messages_list):
                    if messages_list[i + 1].get('sender') == 'bot':
                        pairs.append({
                            'candidate_message': msg,
                            'bot_response': messages_list[i + 1],
                            'thread_id': thread_id
                        })
        
        return pairs
    else:
        # Simplified approach if no thread ID
        candidate_messages = df[df['sender'] == 'candidate'].to_dict('records')
        bot_messages = df[df['sender'] == 'bot'].to_dict('records')
        
        # Try to match by index or other heuristics
        pairs = []
        for i, cand_msg in enumerate(candidate_messages):
            if i < len(bot_messages):
                pairs.append({
                    'candidate_message': cand_msg,
                    'bot_response': bot_messages[i],
                    'thread_id': f"pair_{i}"
                })
                
        return pairs

# Topic modeling analysis
@st.cache_data
def perform_topic_modeling(texts, num_topics=10):
    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=1000,
        stop_words='english',
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )
    
    try:
        dtm = vectorizer.fit_transform(texts)
    except Exception as e:
        st.warning(f"Error creating document-term matrix: {e}. Trying with more lenient parameters...")
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=1,
            max_features=1000,
            stop_words='english',
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )
        dtm = vectorizer.fit_transform(texts)
    
    # Train LDA model
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        max_iter=10,
        learning_method='online',
        random_state=42,
        batch_size=128
    )
    
    lda.fit(dtm)
    
    # Create topic-term matrix
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics[topic_idx] = top_words
    
    # Get document topic distributions
    doc_topic_dist = lda.transform(dtm)
    
    return {
        'topics': topics,
        'doc_topic_dist': doc_topic_dist,
        'lda_model': lda,
        'vectorizer': vectorizer,
        'feature_names': feature_names
    }

# Sentiment analysis
@st.cache_data
def perform_sentiment_analysis(texts):
    analyzer = SentimentIntensityAnalyzer()
    
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            text = str(text)
        
        sentiment = analyzer.polarity_scores(text)
        results.append({
            'text': text,
            'compound': sentiment['compound'],
            'pos': sentiment['pos'],
            'neu': sentiment['neu'],
            'neg': sentiment['neg'],
            'sentiment': 'positive' if sentiment['compound'] >= 0.05 else 
                        'negative' if sentiment['compound'] <= -0.05 else 'neutral'
        })
        
        # Update progress
        progress = (i + 1) / len(texts)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing sentiment: {i+1}/{len(texts)}")
    
    status_text.empty()
    progress_bar.empty()
    
    return pd.DataFrame(results)

# Text clustering
@st.cache_data
def perform_text_clustering(texts, n_clusters=5):
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=1000,
        stop_words='english'
    )
    
    # Check if we have enough texts
    if len(texts) < n_clusters:
        st.warning(f"Not enough texts to form {n_clusters} clusters. Reducing clusters to {len(texts)}.")
        n_clusters = max(2, len(texts))
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Apply dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(texts)-1))
        tsne_results = tsne.fit_transform(tfidf_matrix.toarray())
        
        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Combine results
        clustering_results = pd.DataFrame({
            'text': texts,
            'x': tsne_results[:, 0],
            'y': tsne_results[:, 1],
            'cluster': cluster_labels
        })
        
        return {
            'clustering_results': clustering_results,
            'vectorizer': vectorizer,
            'kmeans': kmeans
        }
    except Exception as e:
        st.error(f"Error in text clustering: {e}")
        return None

# Time series analysis
@st.cache_data
def perform_time_series_analysis(df):
    # Check if we have timestamp data
    if 'createdAt' not in df.columns:
        st.warning("No 'createdAt' column found. Cannot perform time series analysis.")
        return None
    
    # Convert to datetime if not already
    df['createdAt'] = pd.to_datetime(df['createdAt'])
    
    # Sort by timestamp
    df = df.sort_values('createdAt')
    
    # Create time-based features
    df['hour'] = df['createdAt'].dt.hour
    df['day'] = df['createdAt'].dt.day
    df['month'] = df['createdAt'].dt.month
    df['year'] = df['createdAt'].dt.year
    df['dayofweek'] = df['createdAt'].dt.dayofweek
    
    # Message frequency analysis
    daily_messages = df.groupby(df['createdAt'].dt.date).size()
    weekly_messages = df.groupby(pd.Grouper(key='createdAt', freq='W')).size()
    monthly_messages = df.groupby(pd.Grouper(key='createdAt', freq='M')).size()
    
    # Pattern analysis
    hourly_pattern = df.groupby('hour').size()
    daily_pattern = df.groupby('dayofweek').size()
    monthly_pattern = df.groupby('month').size()
    
    # 7-day moving average
    if len(daily_messages) >= 7:
        ma7 = daily_messages.rolling(window=7).mean()
    else:
        ma7 = daily_messages
    
    return {
        'daily_messages': daily_messages,
        'weekly_messages': weekly_messages,
        'monthly_messages': monthly_messages,
        'hourly_pattern': hourly_pattern,
        'daily_pattern': daily_pattern, 
        'monthly_pattern': monthly_pattern,
        'ma7': ma7
    }

# Patronus evaluation
@st.cache_data
def run_patronus_evaluation(pair, intent_names):
    try:
        evaluator = RemoteEvaluator("lynx", "patronus:hallucination")
        
        candidate_msg = pair['candidate_message']
        bot_msg = pair['bot_response']
        intent_name = candidate_msg.get('intent', '')
        
        # Build context
        context = [f"Intent: {intent_name}"]
        
        # Add citations if available
        if 'citations' in bot_msg and bot_msg['citations']:
            context.extend(bot_msg['citations'])
        
        result = evaluator.evaluate(
            task_input=candidate_msg.get('content', ''),
            task_context=context,
            task_output=bot_msg.get('content', ''),
            gold_answer=""
        )
        
        return {
            'candidate_message': candidate_msg.get('content', ''),
            'bot_response': bot_msg.get('content', ''),
            'intent': intent_name,
            'evaluation_result': result
        }
    except Exception as e:
        st.error(f"Error evaluating with Patronus: {e}")
        return {
            'candidate_message': pair['candidate_message'].get('content', ''),
            'bot_response': pair['bot_response'].get('content', ''),
            'intent': pair['candidate_message'].get('intent', ''),
            'evaluation_result': f"Error: {str(e)}"
        }

# Topic modeling page
def topic_modeling_page(topic_model_results):
    st.markdown('<div class="section-title">Topic Modeling Analysis</div>', unsafe_allow_html=True)
    
    topics = topic_model_results['topics']
    doc_topic_dist = topic_model_results['doc_topic_dist']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Topic keywords table
        st.subheader("Topics and Keywords")
        
        # Define topic names based on keywords
        topic_names = {}
        topic_tables = []
        
        for topic_idx, keywords in topics.items():
            # Assign topic name based on keywords
            if 'company' in keywords and 'culture' in keywords:
                name = "Company Culture"
            elif 'job' in keywords and 'requirement' in keywords:
                name = "Job Requirements"
            elif 'salary' in keywords or 'benefit' in keywords:
                name = "Compensation & Benefits"
            elif 'skill' in keywords or 'experience' in keywords:
                name = "Skills & Experience"
            elif 'interview' in keywords:
                name = "Interview Process"
            elif 'resume' in keywords or 'application' in keywords:
                name = "Application Process"
            elif 'team' in keywords or 'management' in keywords:
                name = "Team & Management"
            elif 'role' in keywords or 'position' in keywords:
                name = "Job Roles"
            elif 'data' in keywords or 'analysis' in keywords:
                name = "Data Analysis"
            else:
                name = f"Topic {topic_idx}"
            
            topic_names[topic_idx] = name
            topic_tables.append([f"Topic {topic_idx}: {name}", ", ".join(keywords)])
        
        st.table(pd.DataFrame(topic_tables, columns=["Topic", "Keywords"]))
    
    with col2:
        # Average topic distribution
        st.subheader("Average Topic Distribution")
        avg_topic_dist = doc_topic_dist.mean(axis=0)
        
        topic_labels = [f"Topic {i}: {topic_names[i]}" for i in range(len(topics))]
        
        fig = px.bar(
            x=topic_labels, 
            y=avg_topic_dist,
            labels={'x': 'Topics', 'y': 'Average Probability'},
            title='Average Topic Distribution Across All Documents'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Topic distribution heatmap
    st.subheader("Topic Distribution Heatmap")
    
    # Limit to first 50 docs for visibility
    num_docs_to_show = min(50, doc_topic_dist.shape[0])
    
    fig = px.imshow(
        doc_topic_dist[:num_docs_to_show], 
        labels=dict(x="Topic", y="Document", color="Probability"),
        x=[f"Topic {i}: {topic_names[i]}" for i in range(len(topics))],
        color_continuous_scale="YlOrRd"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Document-topic breakdown (interactive)
    st.subheader("Interactive Document-Topic Explorer")
    
    doc_idx = st.slider("Select Document", 0, len(doc_topic_dist)-1, 0)
    
    # Display document topic distribution
    st.markdown(f"**Document {doc_idx} Topic Distribution:**")
    
    # Create bar chart for document topics
    doc_topics = doc_topic_dist[doc_idx]
    
    fig = px.bar(
        x=topic_labels,
        y=doc_topics,
        labels={'x': 'Topics', 'y': 'Probability'},
        title=f'Topic Distribution for Document {doc_idx}'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# Sentiment analysis page
def sentiment_analysis_page(sentiment_df):
    st.markdown('<div class="section-title">Sentiment Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        
        sentiment_counts = sentiment_df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Distribution of Sentiments",
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': '#4CAF50',
                'neutral': '#FFC107',
                'negative': '#F44336'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average sentiment scores
        st.subheader("Average Sentiment Scores")
        
        avg_scores = {
            'Positive': sentiment_df['pos'].mean(),
            'Neutral': sentiment_df['neu'].mean(),
            'Negative': sentiment_df['neg'].mean(),
            'Compound': sentiment_df['compound'].mean()
        }
        
        fig = px.bar(
            x=list(avg_scores.keys()),
            y=list(avg_scores.values()),
            labels={'x': 'Sentiment Type', 'y': 'Average Score'},
            title='Average Sentiment Scores',
            color=list(avg_scores.keys()),
            color_discrete_map={
                'Positive': '#4CAF50',
                'Neutral': '#FFC107',
                'Negative': '#F44336',
                'Compound': '#2196F3'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment distribution over texts
    st.subheader("Sentiment Distribution by Message")
    
    fig = px.scatter(
        sentiment_df,
        y='compound',
        color='sentiment',
        hover_data=['text'],
        labels={'compound': 'Compound Sentiment Score', 'sentiment': 'Sentiment Category'},
        title='Sentiment Distribution Across Messages',
        color_discrete_map={
            'positive': '#4CAF50',
            'neutral': '#FFC107',
            'negative': '#F44336'
        }
    )
    fig.add_hline(y=0.05, line_dash="dash", line_color="#4CAF50", annotation_text="Positive Threshold")
    fig.add_hline(y=-0.05, line_dash="dash", line_color="#F44336", annotation_text="Negative Threshold")
    st.plotly_chart(fig, use_container_width=True)
    
    # Text examples with highest/lowest sentiment
    st.subheader("Examples of Texts with Extreme Sentiment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Most Positive Messages:**")
        most_positive = sentiment_df.nlargest(3, 'compound')
        for i, row in enumerate(most_positive.itertuples()):
            st.markdown(f"""
            <div class="insight-box" style="background-color: rgba(76, 175, 80, 0.2);">
                <b>Score: {row.compound:.2f}</b><br>
                {row.text[:200]}...
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Most Negative Messages:**")
        most_negative = sentiment_df.nsmallest(3, 'compound')
        for i, row in enumerate(most_negative.itertuples()):
            st.markdown(f"""
            <div class="insight-box" style="background-color: rgba(244, 67, 54, 0.2);">
                <b>Score: {row.compound:.2f}</b><br>
                {row.text[:200]}...
            </div>
            """, unsafe_allow_html=True)

# Time series analysis page
def time_series_page(time_series_results):
    st.markdown('<div class="section-title">Time Series Analysis</div>', unsafe_allow_html=True)
    
    if not time_series_results:
        st.warning("No time data available for analysis.")
        return
    
    daily_messages = time_series_results['daily_messages']
    weekly_messages = time_series_results['weekly_messages']
    monthly_messages = time_series_results['monthly_messages']
    
    hourly_pattern = time_series_results['hourly_pattern']
    daily_pattern = time_series_results['daily_pattern']
    monthly_pattern = time_series_results['monthly_pattern']
    
    # Message frequency over time
    st.subheader("Message Frequency Over Time")
    
    time_options = ["Daily", "Weekly", "Monthly"]
    selected_time = st.radio("Select Time Granularity", time_options, horizontal=True)
    
    if selected_time == "Daily":
        time_data = daily_messages
        title = "Daily Message Frequency"
    elif selected_time == "Weekly":
        time_data = weekly_messages
        title = "Weekly Message Frequency"
    else:
        time_data = monthly_messages
        title = "Monthly Message Frequency"
    
    fig = px.line(
        x=time_data.index,
        y=time_data.values,
        labels={'x': 'Date', 'y': 'Number of Messages'},
        title=title
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Time patterns
    st.subheader("Message Distribution Patterns")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Hourly distribution
        day_parts = ["Morning (6-11)", "Afternoon (12-17)", "Evening (18-23)", "Night (0-5)"]
        day_part_values = [
            hourly_pattern.loc[[6, 7, 8, 9, 10, 11]].sum(),
            hourly_pattern.loc[[12, 13, 14, 15, 16, 17]].sum(),
            hourly_pattern.loc[[18, 19, 20, 21, 22, 23]].sum(),
            hourly_pattern.loc[[0, 1, 2, 3, 4, 5]].sum()
        ]
        
        fig = px.pie(
            values=day_part_values,
            names=day_parts,
            title="Message Distribution by Time of Day"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Day of week distribution
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        fig = px.bar(
            x=days,
            y=[daily_pattern.get(i, 0) for i in range(7)],
            labels={'x': 'Day of Week', 'y': 'Number of Messages'},
            title='Message Distribution by Day of Week'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Monthly distribution
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        fig = px.bar(
            x=months,
            y=[monthly_pattern.get(i+1, 0) for i in range(12)],
            labels={'x': 'Month', 'y': 'Number of Messages'},
            title='Message Distribution by Month'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Moving average
    st.subheader("Message Trend with Moving Average")
    
    ma7 = time_series_results['ma7']
    
    fig = go.Figure()
    
    # Add daily messages
    fig.add_trace(go.Scatter(
        x=daily_messages.index,
        y=daily_messages.values,
        mode='lines',
        name='Daily Messages'
    ))
    
    # Add 7-day moving average
    fig.add_trace(go.Scatter(
        x=ma7.index,
        y=ma7.values,
        mode='lines',
        name='7-day Moving Average',
        line=dict(dash='dash', color='red')
    ))
    
    fig.update_layout(
        title='Daily Messages with 7-day Moving Average',
        xaxis_title='Date',
        yaxis_title='Number of Messages',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Text clustering page
def text_clustering_page(clustering_results):
    st.markdown('<div class="section-title">Text Embeddings & Clustering</div>', unsafe_allow_html=True)
    
    if not clustering_results:
        st.warning("Clustering analysis failed or not available.")
        return
    
    results_df = clustering_results['clustering_results']
    
    # Cluster visualization
    st.subheader("Message Clustering Visualization")
    
    # Add hover text (truncated message)
    results_df['hover_text'] = results_df['text'].apply(lambda x: x[:100] + "..." if len(x) > 100 else x)
    
    fig = px.scatter(
        results_df,
        x='x',
        y='y',
        color='cluster',
        hover_data=['hover_text'],
        labels={'x': 't-SNE Dimension 1', 'y': 't-SNE Dimension 2', 'cluster': 'Cluster'},
        title='Text Clustering using t-SNE and K-Means'
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster statistics
    st.subheader("Cluster Statistics")
    
    cluster_stats = results_df.groupby('cluster').agg({
        'text': 'count'
    }).reset_index()
    
    cluster_stats.columns = ['Cluster', 'Number of Messages']
    
    fig = px.bar(
        cluster_stats,
        x='Cluster',
        y='Number of Messages',
        title='Messages per Cluster'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Representative messages per cluster
    st.subheader("Representative Messages for Each Cluster")
    
    # Create tabs for each cluster
    cluster_tabs = st.tabs([f"Cluster {i}" for i in range(results_df['cluster'].nunique())])
    
    for i, tab in enumerate(cluster_tabs):
        with tab:
            cluster_messages = results_df[results_df['cluster'] == i]['text'].tolist()
            
            if cluster_messages:
                st.markdown(f"**Top 5 messages from Cluster {i}:**")
                
                for j, message in enumerate(cluster_messages[:5]):
                    st.markdown(f"""
                    <div class="insight-box">
                        {j+1}. {message[:300]}...
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write(f"No messages in Cluster {i}")

# Hallucination detection page
def hallucination_detection_page(df, message_pairs, patronus_initialized):
    st.markdown('<div class="section-title">Hallucination Detection with Patronus</div>', unsafe_allow_html=True)
    
    if not patronus_initialized:
        st.warning("Patronus API key not set. Hallucination detection is not available.")
        st.info("To enable this feature, add your Patronus API key to the .env file or as an environment variable named PATRONUS_API_KEY.")
        return
    
    if not message_pairs:
        st.warning("No message pairs found for hallucination detection.")
        return
    
    # Get intent names if available
    intent_names = []
    if 'intent' in df.columns:
        intent_names = df['intent'].unique().tolist()
    
    # Select a message pair to evaluate
    st.subheader("Select Message Pair to Evaluate")
    
    pair_idx = st.selectbox(
        "Select Message Pair", 
        range(len(message_pairs)), 
        format_func=lambda i: f"Pair {i+1}: {message_pairs[i]['candidate_message'].get('content', '')[:50]}..."
    )
    
    selected_pair = message_pairs[pair_idx]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **User Message:**
        <div class="insight-box">
        {selected_pair['candidate_message'].get('content', '')}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        **Bot Response:**
        <div class="insight-box">
        {selected_pair['bot_response'].get('content', '')}
        </div>
        """, unsafe_allow_html=True)
    
    # Run Patronus evaluation on demand
    if st.button("Evaluate for Hallucinations"):
        with st.spinner("Running Patronus evaluation..."):
            result = run_patronus_evaluation(selected_pair, intent_names)
            
            # Display results
            st.subheader("Patronus Evaluation Results")
            
            if isinstance(result['evaluation_result'], str) and result['evaluation_result'].startswith('Error'):
                st.error(result['evaluation_result'])
            else:
                st.json(result['evaluation_result'])
    
    # Batch evaluation option
    st.subheader("Batch Evaluation")
    
    num_pairs = st.slider("Number of pairs to evaluate", 1, min(10, len(message_pairs)), 3)
    
    if st.button(f"Evaluate {num_pairs} Message Pairs"):
        progress_bar = st.progress(0)
        results_placeholder = st.empty()
        
        results = []
        for i in range(num_pairs):
            pair = message_pairs[i]
            results.append(run_patronus_evaluation(pair, intent_names))
            progress_bar.progress((i+1)/num_pairs)
        
        # Display results table
        results_table = []
        for res in results:
            evaluation = res['evaluation_result']
            if isinstance(evaluation, str):
                hallucination_score = "Error"
            else:
                hallucination_score = evaluation.get('hallucination_score', 'N/A')
            
            results_table.append({
                'User Message': res['candidate_message'][:50] + "...",
                'Bot Response': res['bot_response'][:50] + "...",
                'Intent': res['intent'],
                'Hallucination Score': hallucination_score
            })
        
        results_df = pd.DataFrame(results_table)
        results_placeholder.dataframe(results_df)
        
        # Plot scores if available
        if all(isinstance(res['evaluation_result'], dict) for res in results):
            scores = [res['evaluation_result'].get('hallucination_score', 0) for res in results]
            
            fig = px.bar(
                x=list(range(1, len(scores)+1)),
                y=scores,
                labels={'x': 'Message Pair', 'y': 'Hallucination Score'},
                title='Hallucination Scores for Message Pairs'
            )
            st.plotly_chart(fig, use_container_width=True)

# Main application
def main():
    # Sidebar
    with st.sidebar:
        st.title("AI Response Analysis")
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader("Upload JSON file", type=['json'])
        
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            
            # Initialize Patronus if API key available
            patronus_initialized = init_patronus()
        
        st.markdown("---")
        
        # About section
        st.markdown("### About")
        st.markdown("""
        This dashboard performs comprehensive analysis on AI assistant conversations using:
        
        - Topic Modeling
        - Sentiment Analysis 
        - Time Series Analysis
        - Text Clustering
        - Hallucination Detection
        
        Upload your JSON file to get started!
        """)
        
        st.markdown("---")
        st.markdown("Made by [Waliullah Khan](https://github.com/waliullah-khan/ai_review_demo)")
    
    # Main content
    if 'uploaded_file' not in locals() or uploaded_file is None:
        intro_page()
    else:
        # Load data
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)
        
        if df is None:
            st.error("Could not load data from the uploaded file.")
            return
        
        # Basic data info
        num_messages = len(df)
        
        # Check whether we have message pairs
        message_pairs = None
        if 'sender' in df.columns:
            with st.spinner("Finding message pairs..."):
                message_pairs = find_message_pairs(df)
        
        # Preprocess text
        with st.spinner("Preprocessing text..."):
            texts = df['content'].dropna().astype(str).tolist()
            processed_texts = preprocess_text(texts)
        
        # Run analyses
        with st.spinner("Running analyses..."):
            # Topic modeling
            topic_model_results = perform_topic_modeling(processed_texts)
            
            # Sentiment analysis
            sentiment_df = perform_sentiment_analysis(texts)
            
            # Text clustering
            clustering_results = perform_text_clustering(processed_texts)
            
            # Time series analysis
            time_series_results = None
            if 'createdAt' in df.columns:
                time_series_results = perform_time_series_analysis(df)
        
        # Create tabs for different analyses
        tabs = st.tabs([
            "Overview", 
            "Topic Modeling", 
            "Sentiment Analysis", 
            "Time Series", 
            "Text Clustering",
            "Hallucination Detection"
        ])
        
        # Overview tab
        with tabs[0]:
            st.markdown('<div class="section-title">Data Overview</div>', unsafe_allow_html=True)
            
            # Basic stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Messages", num_messages)
            
            with col2:
                if message_pairs:
                    st.metric("Conversation Pairs", len(message_pairs))
                else:
                    st.metric("Conversation Pairs", "N/A")
            
            with col3:
                if 'intent' in df.columns:
                    st.metric("Unique Intents", df['intent'].nunique())
                else:
                    st.metric("Unique Intents", "N/A")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Data structure
            st.subheader("Data Structure")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Column Names:")
                st.write(df.columns.tolist())
            
            with col2:
                st.write("Data Types:")
                st.write(df.dtypes)
            
            # Sample message pair
            if message_pairs:
                st.subheader("Sample Message Pair")
                
                sample_pair = message_pairs[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **User Message:**
                    <div class="insight-box">
                    {sample_pair['candidate_message'].get('content', '')}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    **Bot Response:**
                    <div class="insight-box">
                    {sample_pair['bot_response'].get('content', '')}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Topic modeling tab
        with tabs[1]:
            topic_modeling_page(topic_model_results)
        
        # Sentiment analysis tab
        with tabs[2]:
            sentiment_analysis_page(sentiment_df)
        
        # Time series tab
        with tabs[3]:
            time_series_page(time_series_results)
        
        # Text clustering tab
        with tabs[4]:
            text_clustering_page(clustering_results)
        
        # Hallucination detection tab
        with tabs[5]:
            hallucination_detection_page(df, message_pairs, 'patronus_initialized' in locals() and patronus_initialized)

if __name__ == "__main__":
    main()