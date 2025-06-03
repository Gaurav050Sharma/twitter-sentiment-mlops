import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import nltk

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_data()

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/sentiment_model")

# Load and prepare tokenizer
@st.cache_resource
def prepare_tokenizer():
    # Load your original training data to build the tokenizer
    df = pd.read_csv("data/twitter_training.csv", 
                     names=["Tweet_ID", "Entity", "Sentiment", "Tweet_Content"]).dropna()
    df["Tweet_Content_Split"] = df["Tweet_Content"].apply(word_tokenize)

    lemmatiser = WordNetLemmatizer()
    stop_english = Counter(stopwords.words('english'))

    def clean_text(tokens):
        return " ".join([lemmatiser.lemmatize(word) for word in tokens 
                        if word.isalpha() and word.lower() not in stop_english])

    df["Tweet_Content_Split"] = df["Tweet_Content_Split"].apply(clean_text)
    tokenizer = Tokenizer(num_words=10000, lower=True)
    tokenizer.fit_on_texts(df["Tweet_Content_Split"])
    return tokenizer, lemmatiser, stop_english

# Class dictionary
class_to_index = {"Neutral": 0, "Irrelevant": 1, "Negative": 2, "Positive": 3}
index_to_class = {v: k for k, v in class_to_index.items()}

def main():
    st.title("Twitter Sentiment Analysis")
    st.write("Enter a tweet to predict its sentiment.")

    try:
        model = load_model()
        tokenizer, lemmatiser, stop_english = prepare_tokenizer()

        tweet_input = st.text_area("Tweet", "I love using Streamlit for NLP tasks!")

        if st.button("Analyze"):
            tokens = word_tokenize(tweet_input)
            clean_tokens = [lemmatiser.lemmatize(word) for word in tokens 
                          if word.isalpha() and word.lower() not in stop_english]
            cleaned_text = " ".join(clean_tokens)

            sequence = tokenizer.texts_to_sequences([cleaned_text])
            padded = pad_sequences(sequence, truncating='post', padding='post', maxlen=50)

            prediction = model.predict(padded)
            pred_class = np.argmax(prediction, axis=1)[0]
            sentiment = index_to_class[pred_class]

            st.subheader("Prediction")
            st.write(f"**Sentiment:** {sentiment}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please make sure all required files are present in the correct locations.")

if __name__ == "__main__":
    main() 