# train_kmeans.py
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import joblib
import re
from langdetect import detect, DetectorFactory # Import DetectorFactory

# --- Define your cleaning and language detection functions ---
# These functions should be identical to those used for processing your dashboard's incoming reviews.
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\S*@\S*\s?', '', text) # Remove emails
    text = re.sub(r'#\w+', '', text) # Remove hashtags
    text = re.sub(r'@\w+', '', text) # Remove mentions
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = text.strip() # Remove leading/trailing whitespace
    return text

def detect_lang(text):
    try:
        # Langdetect can be sensitive to very short strings
        if len(text) < 5: # Adjust threshold as needed
            return 'unknown'
        return detect(text)
    except:
        return 'error' # Or 'unknown'

# Set a seed for langdetect for reproducibility (important for consistent results)
DetectorFactory.seed = 0

# --- Load your comprehensive, combined, raw reviews for training ---
# This CSV file should contain your combined old and new reviews
# after you've used pd.concat() and potentially saved them.
try:
    df_raw = pd.read_csv("shein_reviews_combined.csv") #
    print(f"Initial raw reviews count loaded from shein_reviews_combined.csv: {len(df_raw)}") #
except FileNotFoundError:
    print("Error: 'shein_reviews_combined.csv' not found. Please ensure your combined training data is in this file.")
    exit() # Exit if the training data isn't found

# --- Apply Cleaning and Filtering ---
df_raw['clean_text'] = df_raw['content'].apply(clean_text) #
print(f"Reviews after initial clean_text application: {len(df_raw)}") #

# Remove very short or empty entries AND explicitly create a copy
df_cleaned_length = df_raw[df_raw['clean_text'].str.len() > 10].copy() #
print(f"Reviews after length filter (>10 chars): {len(df_cleaned_length)}") #

# Detect language and filter to English
df_cleaned_length['lang'] = df_cleaned_length['clean_text'].apply(detect_lang) #
df_final_training = df_cleaned_length[df_cleaned_length['lang'] == 'en'].copy() #
print(f"Reviews after English language filter: {len(df_final_training)}") #

# Remove duplicate reviews (based on clean_text)
initial_dedupe_rows = len(df_final_training) #
df_final_training.drop_duplicates(subset=['clean_text'], inplace=True) #
print(f"Removed {initial_dedupe_rows - len(df_final_training)} duplicate reviews based on 'clean_text'.") #
print(f"Final unique, cleaned reviews count for training: {len(df_final_training)}") #

# --- Ensure there's data to train on ---
if df_final_training.empty: #
    raise ValueError("No English, non-duplicate reviews found after cleaning. Cannot train KMeans.") #

# Load embedding model
model = SentenceTransformer("all-mpnet-base-v2") #
print("Generating embeddings for training data...") #
embeddings = model.encode(df_final_training["clean_text"].tolist(), show_progress_bar=True) #

# Train KMeans with the optimal num_clusters
num_clusters = 3 # This is your agreed-upon optimal k
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10) # Added n_init=10 for robustness
print(f"Training KMeans model with k={num_clusters}...") #
kmeans.fit(embeddings) #
print("KMeans training complete.") #

# Save the trained model
joblib.dump(kmeans, "kmeans_model.pkl") # Ensure this matches your dashboard's KMEANS_MODEL_PATH
print("KMeans model saved as kmeans_model.pkl") #