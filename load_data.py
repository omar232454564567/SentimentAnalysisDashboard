import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

# Load the dataset with latin-1 encoding
try:
    # Load the dataset
    tweets_df = pd.read_csv("load_excel/sss.csv", encoding="latin-1")
    print("Dataset loaded successfully!")
    
    # Print the first few rows of the dataset
    print(tweets_df.head())
    
    # Print the original column names
    print("Original Columns:", tweets_df.columns)
    
    # Rename columns for clarity
    tweets_df.columns = ["sentiment", "id", "date", "query", "user", "text"]
    print("Updated Columns:", tweets_df.columns)

    # Map sentiment labels (0 = negative, 4 = positive)
    tweets_df["sentiment"] = tweets_df["sentiment"].map({0: "negative", 4: "positive"})
    print(tweets_df[["sentiment", "text"]].head())

    # Download NLTK stopwords
    nltk.download("stopwords")  # Downloads the stopwords corpus
    stop_words = set(stopwords.words("english"))  # Loads English stopwords

    # Function to clean text
    def clean_text(text):
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"@\w+", "", text)     # Remove mentions
        text = re.sub(r"#\w+", "", text)     # Remove hashtags
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
        text = text.lower()                  # Convert to lowercase
        text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
        return text.strip()

    # Apply the cleaning function to the 'text' column
    tweets_df["cleaned_text"] = tweets_df["text"].apply(clean_text)
    print(tweets_df[["text", "cleaned_text"]].head())

    # Save the cleaned dataset
    tweets_df.to_csv("load_excel/cleaned_tweets.csv", index=False, encoding="utf-8")
    print("Cleaned data saved to cleaned_tweets.csv")

except Exception as e:
    print("Error loading dataset:", e)
