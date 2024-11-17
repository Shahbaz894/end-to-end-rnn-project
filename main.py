import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index=imdb.get_word_index()
reverse_word_index={value:key for key , value in word_index.items()}


try:
    model = load_model('simple_rnn_imdb.h5')
    print(model.summary())
except Exception as e:
    print(f"Error: {e}")

# Step 2: Helper Functions
# Function to decode reviews
def decoded_review(encoded_review):
    return ''.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


def preprocess_text(text):
    words=text.lower().split()
    # getting words
    encoded_review=[word_index.get(word,2) + 3 for word in words]
    
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review



import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)
    
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')
    