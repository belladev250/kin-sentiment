---
title: Kinyarwanda Sentiment
emoji: 💬
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-4.0
---

# Kinyarwanda Tweet Sentiment Classifier

Classifies Kinyarwanda text as positive / neutral / negative.
Trained on AfriSenti-Kinyarwanda (Muhammad et al., 2023, CC BY 4.0).

Default model: TF-IDF word + character n-grams with a Linear SVM (`best_classical.joblib`).
To serve the fine-tuned Transformer instead, add a Space variable `MODEL_ID=<your-hf-username>/<model-repo>`
and uncomment `torch` and `transformers` in `requirements.txt`.

Source code: see the GitHub repository linked in the report.
