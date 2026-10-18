
import os
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import LABELS, clean, light_clean  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "bellahug3/kin-sentiment-transformer").strip()
use_transformer = MODEL_ID.lower() != "classical"

if use_transformer:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    except Exception as err:  # missing libraries or no internet -> use the classical model
        print(f"Could not load {MODEL_ID} ({err}); falling back to the classical model.")
        use_transformer = False

if use_transformer:
    MODEL_NAME = f"Fine-tuned AfroXLMR-base Transformer [T3] ({MODEL_ID})"
    PREP = light_clean   # Transformers were trained on lightly cleaned text

    def scores(text):
        enc = tok(light_clean(text), return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            p = torch.softmax(mdl(**enc).logits, -1)[0].numpy()
        return {mdl.config.id2label[i]: float(p[i]) for i in range(len(p))}
else:
    path = next(p for p in [ROOT / "models" / "best_classical.joblib",
                            Path(__file__).resolve().parent / "best_classical.joblib"] if p.exists())
    bundle = joblib.load(path)
    pipe = bundle["pipeline"]
    MODEL_NAME = f"TF-IDF (word 1-2 + char 2-5 grams) + Linear SVM [{bundle['id']}]"
    PREP = clean

    def scores(text):
        # LinearSVC has no probabilities; softmax over decision scores gives a
        # relative confidence (not a calibrated probability).
        d = pipe.decision_function([clean(text)])[0]
        e = np.exp(d - d.max()); p = e / e.sum()
        return {c: float(v) for c, v in zip(pipe.classes_, p)}


def classify(text):
    if not text or not text.strip():
        return {}, "Please type a sentence in Kinyarwanda."
    s = scores(text)
    top = max(s, key=s.get)
    note = ""
    if max(s.values()) < 0.5:
        note = "Low confidence - the model is unsure (common for short or sarcastic texts)."
    return s, f"**Prediction:** {top}  \n**Preprocessed input seen by the model:** `{PREP(text)}`  \n{note}"


EXAMPLES = [
    "Urakoze cyane, umunsi mwiza kuri mwese 🙏",
    "Ibi ni agahomamunwa, ntabwo nshimishijwe n'iyi serivisi.",
    "Inama y'Abaminisitiri iraterana ejo saa tatu.",
    "Mwakoze cyane kuri iyi ndirimbo nziza, mukomeze mutere imbere!",
    "Impanuka yabereye ku muhanda yahitanye abantu batatu.",
    "Hahaha ubwo se ni iki kibaye? 🤔",
]

with gr.Blocks(title="Kinyarwanda Sentiment") as demo:
    gr.Markdown(
        "# Kinyarwanda Tweet Sentiment Classifier\n"
        "Type a sentence or tweet in **Kinyarwanda**. The model predicts whether it is "
        "**positive**, **neutral** or **negative**.\n\n"
        f"Model in use: *{MODEL_NAME}* · trained on AfriSenti-Kinyarwanda (Muhammad et al., 2023)."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Kinyarwanda text", lines=4, placeholder="Andika hano...")
            btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            out = gr.Label(label="Sentiment scores", num_top_classes=3)
            info = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=inp)
    gr.Markdown("**Limitations:** trained on ~3.3k tweets; struggles with sarcasm, very short texts, "
                "code-switching, and words it never saw. Neutral vs negative is the most common confusion.")
    btn.click(classify, inp, [out, info])
    inp.submit(classify, inp, [out, info])

if __name__ == "__main__":
    demo.launch()
