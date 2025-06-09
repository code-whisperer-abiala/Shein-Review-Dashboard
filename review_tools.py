import pandas as pd
import numpy as np
import argparse
import os
import joblib
import datetime
import re
from sklearn.cluster import KMeans # Only kept for potential local testing context
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer

# --- NEW IMPORTS FOR SENTIMENT TRANSFORMER ---
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch # To check for GPU availability

# --- Configuration ---
REVIEW_SOURCE = "com.zzkko"
KMEANS_MODEL_PATH = "kmeans_model.pkl"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# --- NEW SENTIMENT MODEL CONFIGURATION ---
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Set a seed for langdetect for reproducibility
DetectorFactory.seed = 0

# Determine device for PyTorch (for sentiment model)
device = 0 if torch.cuda.is_available() else -1 # 0 for GPU, -1 for CPU
print(f"Using device: {'cuda' if device == 0 else 'cpu'} for sentiment model.")

# Initialize global models (loaded once when the script starts/imported)
# This is efficient for a dashboard as models are not reloaded on every request
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"✅ Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    embedding_model = None

try:
    kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    print(f"✅ KMeans model loaded from '{KMEANS_MODEL_PATH}' successfully.")
except FileNotFoundError:
    print(f"❌ Error: KMeans model not found at '{KMEANS_MODEL_PATH}'. Please train the model first using train_kmeans.py.")
    kmeans_model = None
except Exception as e:
    print(f"❌ Error loading KMeans model: {e}")
    kmeans_model = None

# --- NEW: Load Sentiment Model Globally ---
sentiment_pipeline = None # Initialize to None
try:
    # Use the pipeline helper for zero-shot classification (handles tokenization and model inference)
    # The 'pipeline' function will automatically load tokenizer and model
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL_NAME,
        tokenizer=AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME),
        device=device # Pass the device to the pipeline
    )
    print(f"✅ Sentiment model '{SENTIMENT_MODEL_NAME}' loaded successfully.")
    # This specific model typically outputs 'LABEL_0': Negative, 'LABEL_1': Neutral, 'LABEL_2': Positive
    # We will map these labels in the apply_sentiment function.
except Exception as e:
    print(f"❌ Error loading sentiment model '{SENTIMENT_MODEL_NAME}': {e}")


# --- Helper Functions ---

def clean_text(text):
    """
    Cleans a single text string by lowercasing, removing URLs, emails,
    hashtags, mentions, punctuation, and excess whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def safe_detect_en(text):
    """
    Safely detects if a given text is in English.
    Returns True if English, False otherwise (including errors or very short text).
    """
    try:
        if len(text.strip()) < 5:
            return False
        return detect(text) == "en"
    except Exception:
        return False


def preprocess_reviews(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a series of preprocessing steps to a DataFrame of reviews.
    Includes cleaning text, filtering by length, detecting English language,
    and removing duplicates.

    Args:
        df_raw (pd.DataFrame): The raw DataFrame of reviews, expected to have a 'content' column.

    Returns:
        pd.DataFrame: The processed DataFrame with 'clean_text' and 'lang' columns.
    """
    if df_raw.empty or 'content' not in df_raw.columns:
        print("Warning: Input DataFrame is empty or missing 'content' column for preprocessing.")
        return pd.DataFrame()

    print(f"Initial reviews for preprocessing: {len(df_raw)}")

    df_raw['content'] = df_raw['content'].astype(str).fillna("")

    df_processed = df_raw.copy()
    df_processed['clean_text'] = df_processed['content'].apply(clean_text)
    print(f"Reviews after cleaning text: {len(df_processed)}")

    df_processed = df_processed[df_processed['clean_text'].str.len() > 10].copy()
    print(f"Reviews after length filter (>10 chars): {len(df_processed)}")

    df_processed['is_english'] = df_processed['clean_text'].apply(safe_detect_en)
    df_processed = df_processed[df_processed['is_english']].copy()
    print(f"Reviews after English language filter: {len(df_processed)}")

    initial_dedupe_rows = len(df_processed)
    df_processed.drop_duplicates(subset=['clean_text'], inplace=True)
    print(f"Removed {initial_dedupe_rows - len(df_processed)} duplicate reviews based on 'clean_text'.")
    print(f"Final unique, cleaned reviews after preprocessing: {len(df_processed)}")

    return df_processed.reset_index(drop=True)

def embed_reviews(texts: list) -> np.ndarray:
    """
    Generates embeddings for a list of text strings using the pre-loaded SentenceTransformer model.
    """
    if embedding_model is None:
        raise ValueError("Embedding model not loaded. Cannot embed reviews.")
    if not texts:
        return np.array([])
    print("Generating embeddings...")
    return embedding_model.encode(texts, show_progress_bar=True)

def assign_clusters(df: pd.DataFrame, embeddings: np.ndarray, kmeans_model: KMeans) -> pd.DataFrame:
    """
    Assigns cluster labels to reviews based on their embeddings using a trained KMeans model.
    """
    if kmeans_model is None:
        raise ValueError("KMeans model not loaded. Cannot assign clusters.")
    if embeddings.size == 0:
        df['embedding'] = [[]] * len(df)
        df['cluster'] = np.nan
        return df

    df["embedding"] = embeddings.tolist()
    df["cluster"] = kmeans_model.predict(embeddings)
    return df

# --- NEW: Transformer-based sentiment analysis function ---
def apply_transformer_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies sentiment analysis using a pre-trained Hugging Face transformer model.
    Maps model outputs (e.g., LABEL_0, LABEL_1, LABEL_2) to 'NEGATIVE', 'NEUTRAL', 'POSITIVE'.

    Args:
        df (pd.DataFrame): DataFrame with a 'clean_text' column.

    Returns:
        pd.DataFrame: DataFrame with 'sentiment_score' (from model) and 'sentiment' columns added.
    """
    if sentiment_pipeline is None:
        raise ValueError("Sentiment model not loaded. Cannot apply sentiment analysis.")
    if df.empty or 'clean_text' not in df.columns:
        print("Warning: Input DataFrame is empty or missing 'clean_text' column for sentiment analysis.")
        df['sentiment_score'] = np.nan
        df['sentiment'] = np.nan
        return df

    print("Applying transformer-based sentiment analysis...")
    
    texts_to_analyze = df["clean_text"].tolist()

    # --- THIS IS THE CRUCIAL LINE ---
    # Call the sentiment pipeline for the *entire* dataset here
    sentiment_results = sentiment_pipeline(texts_to_analyze) 
    # --- END CRUCIAL LINE ---

    # Add a print statement here to see what the pipeline returns for a few samples
    # For large datasets, don't print all results, just a few to inspect the format
    sample_texts = texts_to_analyze[:5] # Take first 5 texts
    # IMPORTANT: Use a *slice* of sentiment_results for sample printing, not re-run pipeline
    sample_display_results = sentiment_results[:5] 
    
    print("\n--- Sample Sentiment Pipeline Results ---")
    # Loop through the sample texts and their corresponding results
    for i, (text, res) in enumerate(zip(sample_texts, sample_display_results)):
        print(f"Text: '{text}'")
        print(f"Model Output: {res}")
    print("---------------------------------------\n")

    # The issue you identified about 'positive', 'negative' (lowercase) vs 'LABEL_0', 'LABEL_1', 'LABEL_2'
    # in the model output is also critical! The previous `sentiment_map` was based on `LABEL_X`.
    # Based on your current sample output, the model *is* outputting lowercase 'positive', 'negative', 'neutral'.
    # So, we need to adjust the sentiment_map.

    sentiment_map = {
        'negative': 'NEGATIVE',
        'neutral': 'NEUTRAL',
        'positive': 'POSITIVE'
    }

    scores = []
    labels = []
    for res in sentiment_results: # This loop now correctly uses the defined sentiment_results
        model_label = res['label'] 
        confidence_score = res['score']

        mapped_sentiment = sentiment_map.get(model_label, 'UNKNOWN') 

        final_sentiment_score = 0.0 
        if mapped_sentiment == 'POSITIVE':
            final_sentiment_score = confidence_score
        elif mapped_sentiment == 'NEGATIVE':
            final_sentiment_score = -confidence_score
        
        scores.append(final_sentiment_score)
        labels.append(mapped_sentiment)

    df["sentiment_score"] = scores
    df["sentiment"] = labels
    
    return df

def map_themes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps numerical cluster IDs to descriptive theme names.
    """
    if df.empty or 'cluster' not in df.columns:
        print("Warning: Input DataFrame is empty or missing 'cluster' column for theme mapping.")
        df['theme'] = np.nan
        return df

    # Based on your confirmed inspection:
    theme_map_k3 = {
        0: "Product Quality & Value",    # Confirmed mapping for Cluster 0
        1: "Order & Delivery Issues",    # Confirmed mapping for Cluster 1
        2: "App Experience & Ads"        # Confirmed mapping for Cluster 2
    }

    print("Mapping clusters to themes...")
    df["theme"] = df["cluster"].map(theme_map_k3)
    df["theme"] = df["theme"].fillna("Uncategorized")
    return df

# --- Main Pipeline for Dashboard Update ---

def update_reviews_pipeline():
    """
    Executes the full pipeline to pull, process, cluster, analyze, and save new reviews.
    """
    from google_play_scraper import reviews # Import here to avoid circular dependency/if not always needed

    if kmeans_model is None or embedding_model is None or sentiment_pipeline is None:
        print("🚨 Critical: One or more models not loaded. Aborting update_reviews_pipeline.")
        return

    print("📥 Pulling latest reviews from Google Play Store...")
    try:
        result, continuation_token = reviews(
            REVIEW_SOURCE,
            lang="en",
            country="us",
            count=1000
        )
        df_raw_pull = pd.json_normalize(result)
        print(f"Pulled {len(df_raw_pull)} raw reviews.")
    except Exception as e:
        print(f"❌ Error pulling reviews: {e}. Aborting.")
        return

    df_processed = preprocess_reviews(df_raw_pull)
    if df_processed.empty:
        print("No valid reviews remaining after preprocessing. Exiting.")
        return

    embeddings = embed_reviews(df_processed["clean_text"].tolist())
    if embeddings.size == 0:
        print("No embeddings generated. Exiting.")
        return

    df_clustered = assign_clusters(df_processed, embeddings, kmeans_model)

    # --- NEW: Call the transformer-based sentiment function ---
    df_sentiment = apply_transformer_sentiment(df_clustered)

    df_final = map_themes(df_sentiment)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"updated_reviews_dashboard_{timestamp}.csv"
    
    output_columns = [
        'reviewId', 'userName', 'content', 'score', 'at', 'replyContent',
        'repliedAt', 'clean_text', 'cluster', 'theme', 'sentiment_score', 'sentiment'
    ]
    for col in output_columns:
        if col not in df_final.columns:
            df_final[col] = np.nan

    df_final[output_columns].to_csv(output_filename, index=False)
    print(f"✅ Update complete. Processed {len(df_final)} reviews. File saved as {output_filename}")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review Tools Script for Dashboard Updates")
    parser.add_argument("--update", action="store_true", help="Run the full review update pipeline.")

    args = parser.parse_args()

    if args.update:
        update_reviews_pipeline()
    else:
        print("No operation specified. Use --update to run the pipeline.")
        print("Note: Model training should be done via a separate 'train_kmeans.py' script.")