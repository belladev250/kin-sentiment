# Kinyarwanda Tweet Sentiment Classification

Three-way sentiment classification (positive / neutral / negative) for **Kinyarwanda** tweets, comparing a TF-IDF baseline, classical n-gram models, BiLSTMs trained from scratch, and fine-tuned African-centric Transformers. The best model is deployed as a Gradio web app.

| | |
|---|---|
| **Live app** | https://huggingface.co/spaces/YOUR_HF_USERNAME/kin-sentiment |
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
| **E6** | **Word 1–2 + char 2–5 + Linear SVM (deployed)** | **62.3** (McNemar vs E1, p = 0.0002) |
| N1 | BiLSTM + max-pool (from scratch, 3 seeds) | 57.2 ± 1.5 |
| N2 | BiLSTM + attention (3 seeds) | 56.1 ± 0.7 |
| T1–T3 | mBERT / AfriBERTa / AfroXLMR fine-tuned | run the Colab notebook |

The full table, including ablations, is in `results/*.csv` and the report.

Key lessons:
1. Character n-grams give the largest gain because they capture Kinyarwanda morphology.
2. The BiLSTM trained from scratch does *not* beat the baseline, because of the high out-of-vocabulary rate and small dataset.
3. Removing stopwords hurts, because the stopword list contains the negator *nta*.
4. Most errors sit on the neutral boundary, which is also where humans disagree.

## Repository structure

```
data/                    AfriSenti-Kin train/dev/test TSVs, annotator labels, stopword list
src/preprocess.py        tweet cleaning (shared by training and the app)
src/baselines.py         baseline E1 + classical experiments E0-E6 + ablations A1-A3
src/bilstm.py            PyTorch BiLSTM (+attention) trained from scratch, 3 seeds
src/finetune_transformer.py  fine-tune mBERT / AfriBERTa / AfroXLMR (GPU)
src/error_analysis.py    confusion matrix, per-subset accuracy, error samples, figures
src/significance.py      bootstrap CIs + McNemar test vs baseline
src/annotator_agreement.py   inter-annotator agreement (Fleiss' kappa)
notebooks/finetune_transformers_colab.ipynb   one-click Colab run for the Transformers
app/app.py               Gradio web app
space/                   ready-to-upload Hugging Face Space (app + model)
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
python src/significance.py
# Transformers (GPU): open notebooks/finetune_transformers_colab.ipynb in Colab
python report/build_report.py      # rebuild the PDF with the latest numbers
```

## Run the web app

```bash
python app/app.py                                     # uses the trained SVM
MODEL_ID=YOUR_HF_USERNAME/kin-sentiment-transformer python app/app.py   # uses the fine-tuned Transformer
```

**Deploy:** create a Gradio Space on huggingface.co, then upload the four files in `space/`.

## Acknowledgements

- **Data:** AfriSenti (Muhammad et al., 2023, CC BY 4.0).
- **Stopword list:** KINNEWS (Niyongabo et al., 2020).
- **Pretrained models:** mBERT, AfriBERTa (Ogueji et al., 2021) and AfroXLMR (Alabi et al., 2022).
- **Libraries:** scikit-learn, PyTorch, Hugging Face Transformers and Gradio.

The full references are in the report.
