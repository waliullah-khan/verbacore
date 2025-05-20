# -*- coding: utf-8 -*-
"""
Jupyter Notebook Conversion: Text Embeddings and Clustering Analysis

This script is a direct conversion of the provided Jupyter Notebook JSON.
It implements text embeddings and clustering analysis using:
1. Sentence Transformers for embeddings
2. UMAP for dimensionality reduction
3. HDBSCAN for clustering
4. t-SNE for visualization
"""

# # Text Embeddings and Clustering Analysis
#
# This notebook implements text embeddings and clustering analysis using:
# 1. Sentence Transformers for embeddings
# 2. UMAP for dimensionality reduction
# 3. HDBSCAN for clustering
# 4. t-SNE for visualization

# Import required libraries
import json
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ## 1. Data Loading and Preprocessing

# Load the JSON data
def load_data(file_path):
    """Loads JSON data from a file into a pandas DataFrame."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Define the path to your data file
# NOTE: You need to replace 'Parallel-Prod.AssistMessage.json'
# with the actual path to your data file.
# For demonstration, we'll create a dummy file path.
# In a real scenario, ensure this file exists and is accessible.
DATA_FILE_PATH = 'Parallel-Prod.AssistMessage.json'

# Example: Create a dummy file if it doesn't exist for the script to run
try:
    with open(DATA_FILE_PATH, 'r') as f:
        pass # File exists
except FileNotFoundError:
    print(f"Warning: Data file '{DATA_FILE_PATH}' not found. Creating a dummy file.")
    # Create a minimal dummy JSON structure
    dummy_data = [
        {"content": "This is the first message.", "intent": "greeting"},
        {"content": "Tell me about the requirements.", "intent": "query"},
        {"content": "What jobs are available?", "intent": "query"},
        {"content": "Hello there!", "intent": "greeting"},
        {"content": "How do I apply?", "intent": "instruction"},
        {"content": "Show me the job descriptions.", "intent": "query"},
    ]
    with open(DATA_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(dummy_data, f, indent=2)
    print(f"Dummy file '{DATA_FILE_PATH}' created with sample data.")


# Load the dataset
df = load_data(DATA_FILE_PATH)

# Display basic information
print("Dataset Info:")
print(df.info())
print("\nSample Messages:")
# Ensure 'content' column exists before trying to access it
if 'content' in df.columns:
    print(df['content'].head())
else:
    print("Warning: 'content' column not found in the DataFrame.")
    # Handle the case where 'content' might be missing or named differently
    # For now, we'll just print the available columns
    print("\nAvailable columns:", df.columns)
    # If you know the correct column name, replace 'content' above
    # For demonstration, let's assume the script cannot proceed without 'content'
    # and exit or raise an error. However, for robustness, we'll try to continue.
    # As a placeholder, create an empty 'content' column if missing.
    if 'content' not in df.columns and not df.empty:
        print("Creating dummy 'content' column for demonstration.")
        df['content'] = ["Sample content"] * len(df)
    elif df.empty:
         print("DataFrame is empty. Cannot proceed with analysis.")
         # Exit or handle empty dataframe appropriately
         exit() # Or raise ValueError("DataFrame is empty")


# ## 2. Generate Text Embeddings

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """Generates text embeddings using a Sentence Transformer model."""
    # Load the model
    model = SentenceTransformer(model_name)

    # Generate embeddings
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32  # Adjust batch size based on your hardware
    )

    return embeddings

# Check if 'content' column exists and has data
if 'content' in df.columns and not df['content'].empty:
    # Generate embeddings for all messages
    # Convert to list to handle potential non-string types safely if needed
    texts_list = df['content'].astype(str).tolist()
    embeddings = generate_embeddings(texts_list)
    print(f"\nEmbedding shape: {embeddings.shape}")
else:
    print("\nCannot generate embeddings: 'content' column is missing or empty.")
    # Handle this case, perhaps by exiting or skipping embedding-dependent steps
    embeddings = np.array([]) # Assign empty array to avoid errors later if possible


# ## 3. Dimensionality Reduction

def reduce_dimensions(embeddings):
    """Reduces embedding dimensions using UMAP and t-SNE."""
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        print("Warning: Embeddings array is empty. Skipping dimensionality reduction.")
        return np.array([]), np.array([]) # Return empty arrays

    # Ensure n_neighbors is less than the number of samples for UMAP
    n_samples = embeddings.shape[0]
    n_neighbors_umap = min(15, n_samples - 1)
    if n_neighbors_umap < 2: # UMAP requires at least 2 neighbors
         print(f"Warning: Not enough samples ({n_samples}) for UMAP with default settings. Skipping UMAP.")
         umap_embeddings = np.zeros((n_samples, 2)) # Placeholder
    else:
        # UMAP reduction
        umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors_umap,
            n_components=2,
            metric='cosine',
            random_state=42 # for reproducibility
        )
        umap_embeddings = umap_reducer.fit_transform(embeddings)

    # Ensure enough samples for t-SNE
    if n_samples < 2 : # t-SNE requires at least 2 samples
        print(f"Warning: Not enough samples ({n_samples}) for t-SNE. Skipping t-SNE.")
        tsne_embeddings = np.zeros((n_samples, 2)) # Placeholder
    else:
        # t-SNE reduction
        # Adjust perplexity if necessary (perplexity < n_samples)
        perplexity_tsne = min(30.0, float(n_samples - 1))
        if perplexity_tsne <= 1.0: # Perplexity must be > 1
             print(f"Warning: Cannot set valid perplexity for t-SNE with {n_samples} samples. Skipping t-SNE.")
             tsne_embeddings = np.zeros((n_samples, 2)) # Placeholder
        else:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_tsne)
            tsne_embeddings = tsne.fit_transform(embeddings)

    return umap_embeddings, tsne_embeddings


# Reduce dimensions only if embeddings were generated
if embeddings.size > 0:
    umap_embeddings, tsne_embeddings = reduce_dimensions(embeddings)

    # Add reduced dimensions to dataframe if reduction was successful
    if umap_embeddings.size > 0:
        df['umap_x'] = umap_embeddings[:, 0]
        df['umap_y'] = umap_embeddings[:, 1]
    if tsne_embeddings.size > 0:
        df['tsne_x'] = tsne_embeddings[:, 0]
        df['tsne_y'] = tsne_embeddings[:, 1]
else:
    print("Skipping dimensionality reduction as embeddings are not available.")
    umap_embeddings = np.array([]) # Ensure these are defined for later checks
    tsne_embeddings = np.array([])


# ## 4. Clustering Analysis

def perform_clustering(embeddings_for_clustering):
    """Performs clustering using HDBSCAN."""
    if embeddings_for_clustering.shape[0] == 0:
        print("Warning: No data for clustering. Skipping HDBSCAN.")
        return np.array([]), None # Return empty array and None for clusterer

    # Ensure min_cluster_size is appropriate for the number of samples
    n_samples_clust = embeddings_for_clustering.shape[0]
    min_cluster_size_hdb = max(2, min(5, n_samples_clust // 2)) # Ensure it's at least 2 and reasonable

    if n_samples_clust < min_cluster_size_hdb:
        print(f"Warning: Not enough samples ({n_samples_clust}) for HDBSCAN with min_cluster_size={min_cluster_size_hdb}. Skipping clustering.")
        # Assign all points to noise cluster -1
        clusters = np.full(n_samples_clust, -1)
        clusterer = None
    else:
        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size_hdb,
            metric='euclidean', # UMAP output is typically Euclidean
            cluster_selection_method='eom' # Excess of Mass
        )

        # Fit and predict clusters
        clusters = clusterer.fit_predict(embeddings_for_clustering)

    return clusters, clusterer

# Perform clustering using UMAP embeddings if available
if umap_embeddings.size > 0:
    clusters, clusterer = perform_clustering(umap_embeddings)
    if clusters.size > 0:
        df['cluster'] = clusters
        # Display cluster statistics
        print("\nCluster Distribution:")
        print(df['cluster'].value_counts())
    else:
        print("Clustering did not produce results.")
        df['cluster'] = -1 # Assign default cluster if clustering failed
else:
    print("Skipping clustering as UMAP embeddings are not available.")
    df['cluster'] = -1 # Assign default cluster if skipping


# ## 5. Visualization

def plot_embeddings():
    """Creates interactive scatter plots of the clustered embeddings."""
    # Check if necessary columns exist for plotting
    required_cols_umap = ['umap_x', 'umap_y', 'cluster', 'content']
    required_cols_tsne = ['tsne_x', 'tsne_y', 'cluster', 'content']
    # 'intent' is optional for hover data
    has_intent = 'intent' in df.columns
    hover_data = ['content']
    if has_intent:
        hover_data.append('intent')


    if all(col in df.columns for col in required_cols_umap):
        print("\nGenerating UMAP + HDBSCAN plot...")
        # Create interactive plot using plotly for UMAP
        fig_umap = px.scatter(
            df,
            x='umap_x',
            y='umap_y',
            color='cluster',
            hover_data=hover_data,
            title='Message Clusters (UMAP + HDBSCAN)',
            color_continuous_scale=px.colors.sequential.Viridis # Use a sequential scale if cluster numbers are meaningful or categorical otherwise
            # If clusters are categorical (including -1 for noise), treat color as discrete
            # color='cluster', color_discrete_map={-1: 'lightgrey'}, # Example mapping noise to grey
            # category_orders={"cluster": sorted(df['cluster'].unique())} # Ensure consistent legend order
        )
        # Convert cluster column to string for categorical coloring if preferred
        # df_plot = df.copy()
        # df_plot['cluster_cat'] = df_plot['cluster'].astype(str)
        # fig_umap = px.scatter(df_plot, x='umap_x', y='umap_y', color='cluster_cat', ...)

        fig_umap.show()
    else:
        print("Skipping UMAP plot: Required columns missing (umap_x, umap_y, cluster, content).")


    if all(col in df.columns for col in required_cols_tsne):
        print("\nGenerating t-SNE + HDBSCAN plot...")
        # t-SNE visualization
        fig_tsne = px.scatter(
            df,
            x='tsne_x',
            y='tsne_y',
            color='cluster',
            hover_data=hover_data,
            title='Message Clusters (t-SNE + HDBSCAN)',
            color_continuous_scale=px.colors.sequential.Viridis # Or handle categorical as above
        )
        fig_tsne.show()
    else:
        print("Skipping t-SNE plot: Required columns missing (tsne_x, tsne_y, cluster, content).")

# Plot only if clustering was performed and relevant columns exist
if 'cluster' in df.columns:
    plot_embeddings()
else:
    print("Skipping visualization as clustering was not performed or failed.")


# ## 6. Cluster Analysis

def analyze_clusters():
    """Prints sample messages and common intents for each cluster."""
    if 'cluster' not in df.columns or 'content' not in df.columns:
        print("Cannot analyze clusters: 'cluster' or 'content' column missing.")
        return

    # Analyze cluster contents
    unique_clusters = sorted(df['cluster'].unique())

    if not unique_clusters or (len(unique_clusters) == 1 and unique_clusters[0] == -1):
        print("\nNo meaningful clusters found (only noise points or no clusters).")
        return

    print("\n--- Cluster Analysis ---")
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            noise_points = df[df['cluster'] == cluster_id]
            print(f"\nNoise Points ({len(noise_points)} messages)")
            print("Sample messages classified as noise:")
            print(noise_points['content'].head())
            if 'intent' in df.columns:
                 print("\nCommon intents among noise points:")
                 print(noise_points['intent'].value_counts().head())
            print("-" * 80)
            continue # Skip detailed analysis for noise

        cluster_messages = df[df['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_messages)} messages)")
        print("Sample messages:")
        print(cluster_messages['content'].head())

        if 'intent' in df.columns:
            print("\nCommon intents:")
            print(cluster_messages['intent'].value_counts().head())
        print("-" * 80)

# Analyze clusters if they exist
if 'cluster' in df.columns:
    analyze_clusters()


# ## 7. Similarity Analysis

def find_similar_messages(query_text, n=5):
    """Finds and prints messages most similar to the query text."""
    if embeddings.size == 0 or 'content' not in df.columns:
        print("Cannot perform similarity search: Embeddings or 'content' column missing.")
        return

    # Generate embedding for query text using the same function
    query_embedding = generate_embeddings([query_text])[0]

    # Calculate cosine similarities
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    # Get top N similar messages (indices)
    # Ensure n is not larger than the number of messages
    n_capped = min(n, len(similarities))
    if n_capped == 0:
        print("No messages to compare against.")
        return

    # Get indices of top N similarities, sorted descending
    # np.argsort returns indices of ascending sort, so we take the last n and reverse
    top_indices = np.argsort(similarities)[-n_capped:][::-1]

    print(f"\n--- Similarity Search ---")
    print(f"Query: {query_text}\n")
    print(f"Top {n_capped} Similar messages:")
    for idx in top_indices:
        print(f"\nSimilarity: {similarities[idx]:.4f}")
        print(f"Message: {df['content'].iloc[idx]}")
        if 'intent' in df.columns:
            print(f"Intent: {df['intent'].iloc[idx]}")
        print("-" * 80)

# Example similarity search (only if embeddings exist)
if embeddings.size > 0:
    find_similar_messages("What are the job requirements?")
    find_similar_messages("Is there a greeting message?") # Another example


print("\nScript execution finished.")
