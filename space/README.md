---
title: Kinyarwanda Sentiment
emoji: 💬
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
python_version: "3.11"
pinned: false
license: cc-by-4.0
---

# Kinyarwanda Tweet Sentiment Classifier

Classifies Kinyarwanda text as positive / neutral / negative.
Trained on AfriSenti-Kinyarwanda (Muhammad et al., 2023, CC BY 4.0).

Default model: fine-tuned AfroXLMR-base (T3), loaded from the Hugging Face Hub
(`bellahug3/kin-sentiment-transformer`). If it cannot be loaded, or the Space variable
`MODEL_ID=classical` is set, the app uses the TF-IDF word + character n-gram Linear SVM
(`best_classical.joblib`) instead.

Source code: https://github.com/belladev250/kin-sentiment
