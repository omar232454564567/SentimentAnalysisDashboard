# ========================================
# تحليل مشاعر تغريدات تويتر + رسومات بيانية
# ========================================

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ========== 1. تحميل البيانات ==========
print("Loading dataset...")
tweets_df = pd.read_csv("load_excel/cleaned_tweets.csv")
print("Dataset loaded successfully.")

# ========== 2. تقليل حجم البيانات ==========
print("Reducing dataset size to 90%...")
tweets_df = tweets_df.sample(frac=0.9, random_state=42)
print(f"New dataset size: {len(tweets_df)} rows")

# ========== 3. معالجة القيم الفارغة ==========
print("Handling NaN values...")
tweets_df["cleaned_text"] = tweets_df["cleaned_text"].fillna("")
print("NaN values handled.")

# ========== 4. تحليل المشاعر باستخدام VADER ==========
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    if not text.strip():
        return "neutral", 0.0
    scores = analyzer.polarity_scores(text)
    compound_score = scores["compound"]
    if compound_score >= 0.05:
        return "positive", compound_score
    elif compound_score <= -0.05:
        return "negative", abs(compound_score)
    else:
        return "neutral", 0.0

print("Starting sentiment analysis...")
results = []
for text in tqdm(tweets_df["cleaned_text"], desc="Processing Tweets"):
    results.append(analyze_sentiment(text))

tweets_df[["predicted_sentiment", "confidence"]] = pd.DataFrame(results)
print("Sentiment analysis completed.")

# ========== 5. حفظ البيانات بعد التحليل ==========
print("Saving analyzed dataset...")
tweets_df.to_csv("load_excel/analyzed_tweets.csv", index=False, encoding="utf-8")
print("Analyzed data saved to analyzed_tweets.csv")

# ========== 6. عرض رسومات بيانية ==========
print("Generating visualizations...")
df = pd.read_csv("load_excel/analyzed_tweets.csv")

# -- رسم عدد كل نوع من المشاعر --
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="predicted_sentiment", palette="viridis")
plt.title("Distribution of Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -- رسم دائري لنسبة كل نوع من المشاعر --
sentiment_counts = df["predicted_sentiment"].value_counts()
plt.figure(figsize=(5, 5))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["#4CAF50", "#F44336", "#FFC107"])
plt.title("Sentiment Proportions")
plt.show()

# -- WordCloud لكل نوع من المشاعر --
for sentiment in ["positive", "negative", "neutral"]:
    text = " ".join(df[df["predicted_sentiment"] == sentiment]["cleaned_text"])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud - {sentiment.capitalize()} Tweets")
    plt.show()

print("All visualizations generated successfully.")
