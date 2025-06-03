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
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load and preprocess data
train_df = pd.read_csv("twitter_training.csv", 
                      names=["Tweet_ID", "Entity", "Sentiment", "Tweet_Content"],
                      skiprows=1).dropna()

val_df = pd.read_csv("twitter_validation.csv", 
                     names=["Tweet_ID", "Entity", "Sentiment", "Tweet_Content"],
                     skiprows=1).dropna()

# Combine datasets for tokenizer fitting
all_data = pd.concat([train_df, val_df])

# Initialize lemmatizer and stopwords
lemmatiser = WordNetLemmatizer()
stop_english = Counter(stopwords.words('english'))

# Clean text
def clean_text(text):
    # Convert to lowercase and tokenize
    tokens = word_tokenize(str(text).lower())
    # Remove stopwords and non-alphabetic tokens, then lemmatize
    return " ".join([lemmatiser.lemmatize(word) for word in tokens 
                    if word.isalpha() and word.lower() not in stop_english])

# Clean text data
train_df["Tweet_Content_Clean"] = train_df["Tweet_Content"].apply(clean_text)
val_df["Tweet_Content_Clean"] = val_df["Tweet_Content"].apply(clean_text)

# Prepare tokenizer
MAX_WORDS = 10000
tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True, oov_token="<OOV>")
tokenizer.fit_on_texts(all_data["Tweet_Content"].apply(clean_text))

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_df["Tweet_Content_Clean"])
val_sequences = tokenizer.texts_to_sequences(val_df["Tweet_Content_Clean"])

# Pad sequences
MAX_LENGTH = 50
train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
val_padded = pad_sequences(val_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

# Prepare labels
class_to_index = {"Neutral": 0, "Irrelevant": 1, "Negative": 2, "Positive": 3}
train_df["Label"] = train_df["Sentiment"].map(class_to_index)
val_df["Label"] = val_df["Sentiment"].map(class_to_index)

train_labels = tf.keras.utils.to_categorical(train_df["Label"])
val_labels = tf.keras.utils.to_categorical(val_df["Label"])

# Build improved model with updated architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(50,)),  # Explicit input shape instead of input_length
    tf.keras.layers.Embedding(MAX_WORDS, 128),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile model with learning rate scheduling
initial_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.9
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps, decay_rate
)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Add early stopping and model checkpoint
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train model with validation data
model.fit(
    train_padded, 
    train_labels, 
    epochs=20,
    batch_size=32,
    validation_data=(val_padded, val_labels),
    callbacks=[early_stopping]
)

# Save model
model.save("models/sentiment_model", save_format='tf') 