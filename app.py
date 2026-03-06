import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

st.title("FusionTech AI Customer Review Intelligence Dashboard")

# -----------------------
# LOAD DATA
# -----------------------

df = pd.read_csv("fusiontech_cleaned.csv")

df = df[['text','rating','brand','title_y']]
df = df.dropna(subset=['text','title_y'])

# -----------------------
# SENTIMENT MODEL
# -----------------------

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(review):

    score = analyzer.polarity_scores(str(review))["compound"]

    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["text"].apply(get_sentiment)

# -----------------------
# PRODUCTS WITH MOST NEGATIVE REVIEWS
# -----------------------

st.subheader("Products Generating the Most Customer Complaints")

negative_products = (
    df[df["sentiment"]=="Negative"]
    .groupby("title_y")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(negative_products)

# -----------------------
# PRODUCT SELECTION
# -----------------------

st.subheader("Select a Product for Detailed Analysis")

product_choice = st.selectbox(
    "Choose a product",
    sorted(df["title_y"].unique())
)

product_df = df[df["title_y"] == product_choice]

st.write("Analyzing reviews for:", product_choice)

# -----------------------
# SENTIMENT BREAKDOWN
# -----------------------

st.subheader("Sentiment Breakdown for This Product")

product_sentiment = product_df["sentiment"].value_counts()

st.bar_chart(product_sentiment)

# -----------------------
# TOP CUSTOMER COMPLAINTS
# -----------------------

st.subheader("Top 5 Customer Complaints for This Product")

negative_reviews = product_df[product_df["sentiment"]=="Negative"]["text"]

if len(negative_reviews) > 5:

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(2,2),
        max_features=20
    )

    X = vectorizer.fit_transform(negative_reviews)

    complaints = pd.DataFrame({
        "Complaint Phrase": vectorizer.get_feature_names_out(),
        "Mentions": X.toarray().sum(axis=0)
    }).sort_values("Mentions", ascending=False)

    st.table(complaints.head(5))

else:

    st.write("Not enough negative reviews for complaint analysis.")

# -----------------------
# WORD CLOUD OF ISSUES
# -----------------------

st.subheader("Common Issues Mentioned in Negative Reviews")

negative_text = " ".join(negative_reviews.astype(str))

if negative_text.strip():

    stopwords = {
        "computer","laptop","fusiontech","device",
        "product","pc","machine","system"
    }

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stopwords
    ).generate(negative_text)

    fig, ax = plt.subplots()

    ax.imshow(wordcloud)
    ax.axis("off")

    st.pyplot(fig)

else:

    st.write("No negative review text available.")

# -----------------------
# LIVE REVIEW ANALYSIS
# -----------------------

st.subheader("Live AI Review Analysis")

user_review = st.text_area("Paste a customer review to analyze sentiment")

if user_review:

    score = analyzer.polarity_scores(user_review)["compound"]

    if score >= 0.05:
        sentiment = "Positive"
    elif score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    st.write("Predicted Sentiment:", sentiment)