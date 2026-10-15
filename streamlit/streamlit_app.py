"""
Streamlit web app: Kinyarwanda tweet sentiment classifier (deployed on Streamlit Community Cloud).

How the app uses the trained models:
  - "AfroXLMR (T3)": the fine-tuned Transformer, downloaded once from the Hugging Face Hub
    (bellahug3/kin-sentiment-transformer) and cached. Text -> light_clean() -> sub-word
    tokenizer -> model -> softmax over 3 logits. To fit the free server's memory it is
    loaded in bfloat16 (half the memory of float32; predictions can differ very slightly).
  - "Linear SVM (E6)": the saved scikit-learn pipeline best_classical.joblib.
    Text -> clean() -> TF-IDF word+char n-grams -> Linear SVM -> softmax over decision scores.

Run locally:  streamlit run streamlit/streamlit_app.py
"""
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from preprocess import clean, light_clean

HERE = Path(__file__).resolve().parent
HF_MODEL = "bellahug3/kin-sentiment-transformer"

st.set_page_config(page_title="Kinyarwanda Sentiment", page_icon="💬")


@st.cache_resource(show_spinner="Loading the fine-tuned Transformer (first time takes ~1 minute)...")
def load_transformer():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    return tok, mdl


@st.cache_resource
def load_svm():
    return joblib.load(HERE / "best_classical.joblib")


def predict_transformer(text):
    import torch
    tok, mdl = load_transformer()
    enc = tok(light_clean(text), return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        p = torch.softmax(mdl(**enc).logits.float(), -1)[0].numpy()
    return {mdl.config.id2label[i]: float(p[i]) for i in range(len(p))}, light_clean(text)


def predict_svm(text):
    pipe = load_svm()["pipeline"]
    d = pipe.decision_function([clean(text)])[0]
    e = np.exp(d - d.max())  # softmax over SVM scores: relative confidence, not a calibrated probability
    p = e / e.sum()
    return {c: float(v) for c, v in zip(pipe.classes_, p)}, clean(text)


EXAMPLES = [
    "Urakoze cyane, umunsi mwiza kuri mwese 🙏",
    "Ibi ni agahomamunwa, ntabwo nshimishijwe n'iyi serivisi.",
    "Inama y'Abaminisitiri iraterana ejo saa tatu.",
    "Mwakoze cyane kuri iyi ndirimbo nziza, mukomeze mutere imbere!",
    "Impanuka yabereye ku muhanda yahitanye abantu batatu.",
    "Hahaha ubwo se ni iki kibaye? 🤔",
]

st.title("Kinyarwanda Tweet Sentiment Classifier")
st.write("Type a sentence or tweet in **Kinyarwanda**. The model predicts whether it is "
         "**positive**, **neutral** or **negative**. Trained on AfriSenti-Kinyarwanda (Muhammad et al., 2023).")

model_choice = st.radio(
    "Model", ["AfroXLMR-base, fine-tuned (T3, best: 67.5 macro-F1)", "TF-IDF + Linear SVM (E6, 62.3 macro-F1)"],
    horizontal=False)

example = st.selectbox("Or pick an example", ["(type your own)"] + EXAMPLES)
default = "" if example == "(type your own)" else example
text = st.text_area("Kinyarwanda text", value=default, height=110, placeholder="Andika hano...")

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please type a sentence in Kinyarwanda.")
    else:
        use_t3 = model_choice.startswith("AfroXLMR")
        try:
            scores, seen = predict_transformer(text) if use_t3 else predict_svm(text)
        except Exception as err:  # e.g. the Hub is unreachable -> fall back to the SVM
            st.error(f"The Transformer could not be loaded ({err}). Using the Linear SVM instead.")
            scores, seen = predict_svm(text)
        top = max(scores, key=scores.get)
        icon = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}[top]
        st.subheader(f"{icon} Prediction: {top}")
        for label in ["positive", "neutral", "negative"]:
            st.progress(scores[label], text=f"{label}: {scores[label]:.2f}")
        if max(scores.values()) < 0.5:
            st.info("Low confidence: the model is unsure (common for short, sarcastic or news-style texts).")
        st.caption(f"Preprocessed input seen by the model: `{seen}`")

st.divider()
st.caption("**Limitations:** trained on ~3.3k tweets; struggles with sarcasm, negation, news-style tweets whose "
           "sentiment depends on world knowledge, and the neutral/negative boundary, where human annotators also "
           "disagree. Source code: https://github.com/belladev250/kin-sentiment")
