#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import required libraries
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# --- NLTK Data Download ---
# Set NLTK data path
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
    print(f"Created NLTK data directory: {nltk_data_path}")

# Add the path to NLTK's data path
nltk.data.path.append(nltk_data_path)

def download_nltk_resource(resource, resource_path):
    """Downloads an NLTK resource if it's not already present."""
    try:
        nltk.data.find(resource_path)
        print(f"NLTK resource '{resource}' already downloaded.")
    except LookupError:
        print(f"NLTK resource '{resource}' not found. Downloading...")
        nltk.download(resource, download_dir=nltk_data_path)

# Download necessary NLTK data
download_nltk_resource('punkt', 'tokenizers/punkt')
download_nltk_resource('stopwords', 'corpora/stopwords')
download_nltk_resource('wordnet', 'corpora/wordnet')
download_nltk_resource('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')

# --- 1. Data Loading and Preprocessing ---
def load_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {file_path}")
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        print("Please ensure 'Parallel-Prod.AssistMessage.json' is in the same directory as the script.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Check file format.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        exit()

# Load the dataset
DATA_FILE = 'Parallel-Prod.AssistMessage.json'
print("--- Starting Data Loading ---")
df = load_data(DATA_FILE)

if 'content' not in df.columns:
    print(f"Error: The required column 'content' is not found in {DATA_FILE}.")
    print(f"Available columns are: {list(df.columns)}")
    exit()

# Extract content column
texts = df['content'].dropna().astype(str).tolist()
print(f"Extracted {len(texts)} non-null text entries from 'content' column.")

# Text preprocessing
def preprocess_text(text):
    """Tokenizes, removes stopwords, and lemmatizes text."""
    try:
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
        
        return ' '.join(processed_tokens)
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return ""

# Process all texts
print("--- Starting Text Preprocessing ---")
processed_texts = []
for i, text in enumerate(texts):
    processed = preprocess_text(text)
    if processed:  # Only add non-empty processed texts
        processed_texts.append(processed)
    if (i + 1) % 100 == 0:  # Print progress every 100 texts
        print(f"Processed {i + 1}/{len(texts)} texts")

print(f"Preprocessing complete. {len(processed_texts)} documents remaining after processing.")

if not processed_texts:
    print("Error: No valid documents remained after preprocessing. Cannot proceed.")
    exit()

# --- 2. Topic Model Training ---
print("--- Starting LDA Model Training ---")

# Create document-term matrix with adjusted parameters
vectorizer = CountVectorizer(
    max_df=0.95,
    min_df=2,
    max_features=1000,
    stop_words='english',  # Use built-in stop words
    token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only words with 3 or more letters
)

try:
    dtm = vectorizer.fit_transform(processed_texts)
    print(f"Created document-term matrix with {dtm.shape[1]} features")
except Exception as e:
    print(f"Error creating document-term matrix: {str(e)}")
    print("Trying with more lenient parameters...")
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=1,  # Reduced from 2
        max_features=1000,
        stop_words='english',
        token_pattern=r'\b[a-zA-Z]{2,}\b'  # Reduced from 3 to 2
    )
    dtm = vectorizer.fit_transform(processed_texts)
    print(f"Created document-term matrix with {dtm.shape[1]} features")

# Train LDA model
num_topics = 10
lda = LatentDirichletAllocation(
    n_components=num_topics,
    max_iter=10,
    learning_method='online',
    random_state=42,
    batch_size=128,
    verbose=1
)

lda.fit(dtm)

# Define topic names based on the content (updated for recruitment context)
topic_names = {
    0: "Workplace Environment",
    1: "Leadership & Assessment",
    2: "Application Process",
    3: "Candidate Profiles",
    4: "Job Requirements",
    5: "Compensation & Benefits",
    6: "Company Vision & Culture",
    7: "Salary Expectations",
    8: "Job Locations & Formats",
    9: "Job Search Strategy"
}

# Print the topics in a table format
print("\n--- Top words per topic: ---")
feature_names = vectorizer.get_feature_names_out()

# Create a list of lists for the table
table_data = []
for topic_idx in range(num_topics):  # Explicitly iterate through all topics
    topic = lda.components_[topic_idx]
    top_words_idx = topic.argsort()[:-10 - 1:-1]  # Get top 10 words
    top_words = [feature_names[i] for i in top_words_idx]
    table_data.append([f"Topic {topic_idx}: {topic_names[topic_idx]}", ", ".join(top_words)])

# Print the table with all topics
print("\nAll Topics:")
print(tabulate(table_data, headers=["Topic Name", "Top Words"], tablefmt="grid"))

# Verify the number of topics
print(f"\nTotal number of topics: {num_topics}")
print(f"Number of components in LDA model: {lda.n_components}")

# --- 3. Topic Distribution Analysis ---
print("\n--- Analyzing Topic Distributions ---")

# Get topic distributions
topic_distributions = lda.transform(dtm)

# Plot topic distribution heatmap for a subset of documents
print("--- Plotting Topic Distribution Heatmap (First 50 Docs) ---")
plt.figure(figsize=(14, 8))
num_docs_to_show = min(50, len(topic_distributions))
sns.heatmap(
    topic_distributions[:num_docs_to_show],
    cmap='YlOrRd',
    xticklabels=[f"Topic {i}: {topic_names[i]}" for i in range(num_topics)],  # Use topic names instead of IDs
    yticklabels=False
)
plt.title(f'Topic Distribution Across First {num_docs_to_show} Documents')
plt.xlabel('Topics')
plt.ylabel('Documents')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent label cutoff
heatmap_filename = "topic_distribution_heatmap.png"
plt.savefig(heatmap_filename)
print(f"Heatmap saved to {heatmap_filename}")
plt.show()

# Calculate and plot average topic distribution
print("--- Plotting Average Topic Distribution ---")
avg_topic_dist = topic_distributions.mean(axis=0)
plt.figure(figsize=(14, 6))
plt.bar(range(num_topics), avg_topic_dist)
plt.title('Average Topic Distribution Across All Documents')
plt.xlabel('Topics')
plt.ylabel('Average Probability')
plt.xticks(range(num_topics), [f"Topic {i}: {topic_names[i]}" for i in range(num_topics)], rotation=45, ha='right')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()  # Adjust layout to prevent label cutoff
avg_dist_filename = "average_topic_distribution.png"
plt.savefig(avg_dist_filename)
print(f"Average distribution plot saved to {avg_dist_filename}")
plt.show()

# Additional analysis: Print top 5 documents for each topic
print("\n--- Top Documents for Each Topic ---")
for topic_idx in range(num_topics):
    print(f"\nTopic {topic_idx}: {topic_names[topic_idx]}")
    
    # Get documents sorted by their affinity to this topic
    topic_document_scores = [(i, score) for i, score in enumerate(topic_distributions[:, topic_idx])]
    topic_document_scores = sorted(topic_document_scores, key=lambda x: x[1], reverse=True)
    
    # Print top 5 documents (or fewer if there aren't 5)
    for i, (doc_idx, score) in enumerate(topic_document_scores[:5]):
        if i < 5:  # Just to be safe
            doc_preview = texts[doc_idx][:100] + "..." if len(texts[doc_idx]) > 100 else texts[doc_idx]
            print(f"  Document {doc_idx} (score: {score:.4f}): {doc_preview}")

print("\n--- Script Finished ---")