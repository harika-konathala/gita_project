import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import re

import base64
import streamlit as st

# -----------------------------
# Load background image as Base64
# -----------------------------
with open("C:/Users/91958/OneDrive/Desktop/harika/geetha-3.jpg", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Bhagavad Gita AI Chatbot", layout="wide")

# -----------------------------
# Background & CSS

# -----------------------------
st.markdown(
    f"""
    <style>
    /* Background for the app */
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}

    /* Card hover effect */
    .verse-card:hover {{
        transform: scale(1.02);
        transition: all 0.3s ease;
color:white
    }}

    /* Download button styling: remove background, make text black & bold */
    .download-button button {{
       font-weight: bold !important;
color:black;
        border-radius: 8px;
        padding: 5px 15px;
        cursor: pointer;
        transition: all 0.2s ease;
    }}

    /* Text box styling */
    .stTextInput input {{
        border-radius: 8px;
        padding: 10px;
    }}

    /* Dropdown selected value text */
    div.stSelectbox > div[role="combobox"] > div {{
        color: black !important;
        font-weight: bold !important;
    }}

    /* Make all other text white and bold */
    h1, h2, h3, h4, h5, h6, p, span, label {{
        color: white !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# -----------------------------
# Load data
# -----------------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("Bhagwad_Gita.csv")
    df["verse_id"] = "Chapter " + df["Chapter"].astype(str) + " Verse " + df["Verse"].astype(str)
    df["verse_text"] = df["EngMeaning"]
    return df

df = load_data()

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# Create FAISS index
# -----------------------------
@st.cache_resource
def create_index(_model, dataframe):
    verses = dataframe["verse_text"].tolist()
    embeddings = _model.encode(verses)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

index = create_index(model, df)

# -----------------------------
# Search function
# -----------------------------
def search_gita(question, top_k=3):
    query_vector = model.encode([question])
    distances, indices = index.search(query_vector, top_k)
    results = df.iloc[indices[0]]
    return results

# -----------------------------
# Translation function
# -----------------------------
def translate_text(text, target_lang):
    if target_lang == "English":
        return text

    lang_map = {
        "Hindi": "hi","Telugu": "te","Tamil": "ta","Kannada": "kn",
        "Malayalam": "ml","Marathi": "mr","Gujarati": "gu",
        "Bengali": "bn","Punjabi": "pa","Urdu": "ur"
    }
    code = lang_map.get(target_lang, "en")
    max_len = 5000
    chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]
    translated_chunks = []
    for chunk in chunks:
        translated_chunks.append(GoogleTranslator(source='auto', target=code).translate(chunk))
    return " ".join(translated_chunks)

# -----------------------------
# UI
# -----------------------------
st.title("Bhagavad Gita AI Chatbot")

col1, col2 = st.columns([3,1])

with col1:
    question = st.text_input("Ask a question about Bhagavad Gita:")

with col2:
    language = st.selectbox(
        "Select language:",
        ["English","Hindi","Telugu","Tamil","Kannada","Malayalam","Marathi","Gujarati","Bengali","Punjabi","Urdu"]
    )
    voice_gender = st.radio("Select voice gender:", ["Male","Female"])

# -----------------------------
# Inspirational quote if no question
# -----------------------------
if not question:
    sample_quote = df.sample(1)["EngMeaning"].values[0]
    st.markdown(f"💡 Inspirational Verse: *{sample_quote}*")

# -----------------------------
# Handle question
# -----------------------------
if question:
    results = search_gita(question)

    for i, row in results.iterrows():
        chapter = row["Chapter"]
        verse_num = row["Verse"]
        shloka = row["Shloka"]
        english_meaning = row["EngMeaning"]

        # Remove numbers at start for TTS
        cleaned_text = re.sub(r"^\d+(\.\d+)?\s*", "", english_meaning)
        translated_meaning = translate_text(cleaned_text, language)

        # Verse card
        st.markdown(f'<div class="verse-card">', unsafe_allow_html=True)
        st.subheader(f"Chapter {chapter}, Verse {verse_num}")
        st.markdown(f"**Shloka:** {shloka}")
        st.markdown(f"**English Meaning:** {english_meaning}")
        if language != "English":
            st.markdown(f"**{language} Meaning:** {translated_meaning}")

        # -----------------------------
        # Text-to-speech
        # -----------------------------
        tts_lang_map = {
            "English":"en","Hindi":"hi","Telugu":"te","Tamil":"ta","Kannada":"kn",
            "Malayalam":"ml","Marathi":"mr","Gujarati":"gu",
            "Bengali":"bn","Punjabi":"pa","Urdu":"ur"
        }
        tts_code = tts_lang_map.get(language, "en")
        tts = gTTS(text=translated_meaning, lang=tts_code)
        audio_file = f"verse_audio_{chapter}_{verse_num}.mp3"
        tts.save(audio_file)
        st.audio(audio_file, format="audio/mp3")
        with open(audio_file, "rb") as f:
            st.download_button(
                label="Download Audio",
                data=f,
                file_name=f"Chapter{chapter}_Verse{verse_num}_{language}.mp3",
                mime="audio/mp3",
                key=f"{chapter}_{verse_num}"
            )
        st.markdown("</div>", unsafe_allow_html=True)