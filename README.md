📰 The News Filter

The **News Filter** is a Streamlit web app that:

* Classifies news articles into categories
* Translates content into English using Hugging Face MarianMT models
* Summarizes articles with **Sumy** + **NLTK**
* Provides an interactive, user-friendly interface

---

## 🚀 Features

* 🧠 Machine learning powered news classification (TensorFlow/Keras + scikit-learn)
* 🌍 Automatic translation (Hugging Face MarianMT + SentencePiece)
* ✂️ Summarization of long articles (Sumy + NLTK)
* 📊 Interactive web app built with **Streamlit**

---

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/StutiGupta12451/the-news-filter.git
   cd the-news-filter
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv env
   source env/bin/activate   # Linux/Mac
   env\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK data (only once):

   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

---

## ▶️ Usage

Run the Streamlit app locally:

```bash
streamlit run news.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📂 Repository Structure

```
the-news-filter/
│
├── news.py                 # Main Streamlit app
├── requirements.txt        # Python dependencies
├── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer
├── news_classifier_model.h5 # Trained classification model
└── README.md               # Project documentation
```

---

## ☁️ Deploying on Streamlit Cloud

1. Push your code + model files to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub repo and deploy
4. Streamlit will auto-install `requirements.txt`

---

## ⚡ Requirements

* Python 3.9+
* TensorFlow / Keras
* Hugging Face Transformers
* SentencePiece
* NLTK + Sumy
* Streamlit

---

