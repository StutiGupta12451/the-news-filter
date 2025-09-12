import streamlit as st
import joblib
import numpy as np
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from tensorflow.keras.models import load_model
from transformers import MarianMTModel, MarianTokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Load ML components
model = load_model("news_classifier_model.h5")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load translation model
trans_model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(trans_model_name)
translator = MarianMTModel.from_pretrained(trans_model_name)

def translate_to_hindi(text):
    tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
    translated = translator.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def summarize_text(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

# --- Streamlit UI ---
st.set_page_config(page_title="News AI App", layout="wide")

st.title("📰 The News Filter")

tabs = st.tabs(["🧠 Categorize", "🌐 Translate to Hindi", "📝 Summarize"])

# Tab 1: Categorization
with tabs[0]:
    st.header("📌 Categorize the News")
    news_input = st.text_area("""Enter your news article:ARTS & CULTURE --> 0
BUSINESS --> 1
COMEDY --> 2
CRIME --> 3
EDUCATION --> 4
ENTERTAINMENT --> 5
ENVIRONMENT --> 6
MEDIA --> 7
POLITICS --> 8
RELIGION --> 9
SCIENCE --> 10
SPORTS --> 11
TECH --> 12
WOMEN --> 13""")
    if st.button("Categorize"):
        if news_input:
            vec = vectorizer.transform([news_input])
            prediction = model.predict(vec.toarray())
            predicted_class = np.argmax(prediction, axis=1)
            category = label_encoder.inverse_transform(predicted_class)[0]
            st.success(f"Predicted Category: *{category}*")
        else:
            st.warning("Please enter some text.")

# Tab 2: Translation
with tabs[1]:
    st.header("🌐 Translate News to Hindi")
    text_to_translate = st.text_area("Enter news to translate:")
    if st.button("Translate to Hindi"):
        if text_to_translate:
            hindi = translate_to_hindi(text_to_translate)
            st.success("Translated News:")
            st.write(hindi)
        else:
            st.warning("Enter some text to translate.")

# Tab 3: Summarization
with tabs[2]:
    st.header("📝 Summarize the News")
    text_to_summarize = st.text_area("Paste your news article:")
    if st.button("Summarize"):
        if text_to_summarize:
            summary = summarize_text(text_to_summarize)
            st.success("Summary:")
            st.write(summary)
        else:

            st.warning("Enter some text to summarize.")

