# Kinyarwanda Tweet Sentiment Classification

Three-way sentiment classification (positive / neutral / negative) for **Kinyarwanda** tweets, comparing a TF-IDF baseline, classical n-gram models, BiLSTMs trained from scratch, and fine-tuned Transformers. The best model (fine-tuned AfroXLMR-base) is deployed as a Streamlit web app.

| | |
|---|---|
| **Live app** | https://kin-sentiment.streamlit.app |
| **Fine-tuned model** | https://huggingface.co/bellahug3/kin-sentiment-transformer |
| **Demo video** | https://YOUR-VIDEO-LINK |
| **Report (PDF)** | [`report/report.pdf`](report/report.pdf) |

## Problem

Almost all sentiment tools are built for English. Kinyarwanda, Rwanda's national language, has few labelled datasets and few pretrained models. It is also morphologically rich: *urakoze*, *murakoze* and *mwakoze* ("thank you") are different words sharing one stem. The target users are Rwandan organisations (telecoms, banks, media, public agencies) that want to track public opinion expressed in Kinyarwanda.

## Dataset

**AfriSenti, Kinyarwanda subset** (Muhammad et al., 2023), CC BY 4.0. The tweets are real and were labelled by three native speakers, with the final label decided by majority vote. We use the official splits.

| Split | Tweets | Negative | Neutral | Positive |
|---|---|---|---|---|
| Train | 3,302 | 1,146 | 1,257 | 899 |
| Dev   | 827   | 287   | 315   | 225 |
| Test  | 1,026 | 355   | 393   | 278 |

Findings about the data:
- The files are sorted by label, so shuffling is essential.
- 31% of test tokens never appear in the training data.
- Annotators agreed unanimously on only **35.8%** of tweets (Fleiss' κ = 0.32), so there is substantial label noise.

## Results (test set, scores ×100)

| ID | Approach | Test macro-F1 |
|---|---|---|
| E0 | Majority class | 18.5 |
| **E1** | **Baseline:** TF-IDF word unigrams + Logistic Regression | **57.9** |
| E3 | TF-IDF char 2–5-grams + LogReg | 61.3 |
| E4 | Word 1–2 + char 2–5 + LogReg | 62.4 |
| E6 | Word 1–2 + char 2–5 + Linear SVM (best classical, fallback in app) | 62.3 (McNemar vs E1, p = 0.0002) |
| N1 | BiLSTM + max-pool (from scratch, 3 seeds) | 57.2 ± 1.5 |
| N2 | BiLSTM + attention (3 seeds) | 56.1 ± 0.7 |
| T1 | mBERT fine-tuned (no Kinyarwanda in pretraining, 3 seeds) | 57.6 ± 0.7 |
| T2 | AfriBERTa-base fine-tuned (3 seeds) | 63.3 ± 0.3 |
| **T3** | **AfroXLMR-base fine-tuned (3 seeds, deployed)** | **67.5 ± 0.7** (McNemar vs E6, p = 0.002) |

The full table, including ablations, is in `results/*.csv` and the report.

Key lessons:
1. Character n-grams give the largest gain because they capture Kinyarwanda morphology.
2. The BiLSTM trained from scratch does *not* beat the baseline, because of the high out-of-vocabulary rate and small dataset.
3. Removing stopwords hurts, because the stopword list contains the negator *nta*.
4. Pretraining on the target language is decisive: mBERT (no Kinyarwanda) is no better than the baseline, while AfroXLMR beats every other model. Its biggest gains are on short tweets (+15 points accuracy) and tweets full of unseen words (+7).
5. Most errors of every model sit on the neutral/negative boundary, which is also where humans disagree: both models are far more accurate on tweets that all three annotators labelled the same way.

## Repository structure

```
data/                    AfriSenti-Kin train/dev/test TSVs, annotator labels, stopword list
src/preprocess.py        tweet cleaning (shared by training and the app)
src/baselines.py         baseline E1 + classical experiments E0-E6 + ablations A1-A3
src/bilstm.py            PyTorch BiLSTM (+attention) trained from scratch, 3 seeds
src/finetune_transformer.py  fine-tune mBERT / AfriBERTa / AfroXLMR (GPU)
src/error_analysis.py    confusion matrix, per-subset accuracy, error samples, figures
src/significance.py      bootstrap CIs + McNemar test vs baseline
src/transformer_analysis.py  T3 vs E6: per-class, subsets, annotator agreement, examples
src/annotator_agreement.py   inter-annotator agreement (Fleiss' kappa)
notebooks/finetune_transformers_colab.ipynb   one-click Colab run for the Transformers
streamlit/               deployed Streamlit app (streamlit_app.py + requirements + SVM fallback)
app/app.py               Gradio version of the app
space/                   ready-to-upload Hugging Face Space (Gradio)
report/build_report.py   regenerates report.docx / report.pdf from results/
results/                 all metrics, predictions, figures
models/                  saved models
```

## Reproduce

```bash
pip install -r requirements.txt
python src/baselines.py            # ~1-2 min on CPU
python src/bilstm.py               # ~3 min on CPU
python src/annotator_agreement.py
python src/error_analysis.py
# Transformers (GPU): open notebooks/finetune_transformers_colab.ipynb in Colab,
# then copy results/transformer_*.csv into results/ and run:
python src/transformer_analysis.py
python src/significance.py
python report/build_report.py      # rebuild the PDF with the latest numbers
```

## Run the web app

**Deployed (Streamlit Community Cloud):** https://kin-sentiment.streamlit.app

```bash
pip install -r streamlit/requirements.txt
streamlit run streamlit/streamlit_app.py     # choose AfroXLMR (T3) or the Linear SVM in the app
```

The app downloads the fine-tuned model from the Hugging Face Hub (`bellahug3/kin-sentiment-transformer`) on first use.
If it can't, it falls back to the saved SVM.

**Deploy:** on share.streamlit.io, choose *Create app → Deploy a public app from GitHub*. Pick this repository, set the main file
to `streamlit/streamlit_app.py`, and choose Python 3.12 under *Advanced settings*.

A Gradio version is also included: `python app/app.py` locally, or upload `space/` to a Hugging Face Space.

## Acknowledgements

- **Data:** AfriSenti (Muhammad et al., 2023, CC BY 4.0).
- **Stopword list:** KINNEWS (Niyongabo et al., 2020).
- **Pretrained models:** mBERT, AfriBERTa (Ogueji et al., 2021) and AfroXLMR (Alabi et al., 2022).
- **Libraries:** scikit-learn, PyTorch, Hugging Face Transformers and Gradio.
- **AI assistance:** AI tools were used to help with parts of the code, debugging and report editing; all experiments were run and verified by the author (see the report).

The full references are in the report.
