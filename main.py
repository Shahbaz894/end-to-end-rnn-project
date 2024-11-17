import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import os
import re

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load IMDB Word Index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Decode reviews
def decoded_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Preprocess user input
def preprocess_text(text):
    words = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Load the model
if not os.path.exists('simple_rnn_imdb.h5'):
    st.error("Model file not found. Please upload 'simple_rnn_imdb.h5'.")
else:
    model = load_model('simple_rnn_imdb.h5')
    st.success("Model loaded successfully.")

# Streamlit App
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if not user_input.strip():
        st.warning("Please enter a valid movie review before classifying.")
    else:
        preprocess_input = preprocess_text(user_input)
        with st.spinner('Classifying sentiment...'):
            prediction = model.predict(preprocess_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        confidence = round(prediction[0][0] * 100, 2)
        st.write(f'Sentiment: {sentiment} (Confidence: {confidence}%)')
else:
    st.write('Please enter a movie review.')
