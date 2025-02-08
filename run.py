import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load dataset and train LabelEncoder (same one used during training)
data = pd.read_csv("roman urdu poetry.csv").iloc[:500]
poetry_lines = data["Poetry"].dropna().tolist()

text = " ".join(poetry_lines)
words = text.split()

word_encoder = LabelEncoder()
word_encoder.fit(words)
word_to_index = {word: i for i, word in enumerate(word_encoder.classes_)}
index_to_word = {i: word for word, i in word_to_index.items()}

# Load the trained model
model = tf.keras.models.load_model("poetry_gru_model.h5")

st.set_page_config(page_title="Nazam Generator", page_icon="\ud83d\udcc4")

# Streamlit UI
st.markdown("""
    <style>
        body {
            background-color: #f4f4f9;
        }
        .main {
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .title {
            font-size: 2.5rem;
            color: #135387;
            text-align: center;
            margin-bottom: 1rem;
        }
        .footer {
            text-align: center;
            margin-top: 2rem;
        }
        .footer a {
            color: #135387;
            margin: 0 10px;
            text-decoration: none;
            font-size: 1.2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>Nazam Generator with GRU</div>", unsafe_allow_html=True)

# User Inputs
start_text = st.text_input("Enter starting words:", "dil ke armaan")
words_per_line = st.slider("Words per Line:", 3, 15, 5)
total_lines = st.slider("Total Lines:", 2, 10, 5)

# Function to Generate Nazam
def generate_nazam(start_text, words_per_line, total_lines):
    generated_text = start_text.split()
    
    for _ in range(total_lines * words_per_line):
        encoded_input = [word_to_index.get(word, 0) for word in generated_text[-5:]]
        encoded_input = pad_sequences([encoded_input], maxlen=5, truncating="pre")
        
        predicted_index = np.argmax(model.predict(encoded_input), axis=-1)[0]
        next_word = index_to_word.get(predicted_index, "")

        generated_text.append(next_word)

        if len(generated_text) % words_per_line == 0:
            generated_text.append("\n")

    return " ".join(generated_text)

# Generate and display Nazam
if st.button("Generate Nazam"):
    nazam = generate_nazam(start_text, words_per_line, total_lines)
    st.text_area("Generated Nazam:", nazam, height=200)

st.markdown("</div>", unsafe_allow_html=True)

# Footer with Social Links
st.markdown("""
    <div class='footer'>
        <p>Connect with me:</p>
        <a href='https://github.com/yourgithubhandle' target='_blank'>GitHub</a>
        <a href='https://linkedin.com/in/yourlinkedinhandle' target='_blank'>LinkedIn</a>
    </div>
""", unsafe_allow_html=True)
