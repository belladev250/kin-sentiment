# Demo video script (aim for ~10 minutes) + viva prep

Record your screen with OBS or Loom. The rubric requires you to *explain* each part, not just show it, so talk through the "why" in every section.

## 1. Problem and language (1 min)
- Explain that sentiment tools exist for English but barely exist for Kinyarwanda.
- Name the users: telecoms, banks, media houses and government agencies watching public opinion.
- Explain why Kinyarwanda is hard:
  - It is agglutinative: urakoze / murakoze / mwakoze all mean "thank you" and share the stem -koze.
  - Tweets add slang, emojis and code-switching.

## 2. Dataset (1.5 min)
Show `data/train.tsv`.
- **Source:** AfriSenti, real tweets labelled by 3 native speakers by majority vote, CC BY 4.0.
- **Size:** 3,302 / 827 / 1,026 tweets, using the official splits.
- **Problems found:**
  - Long @user chains, elongated words (hhhhhh) and emojis.
  - The files are sorted by label, so the data must be shuffled.
- **Key finding:** annotators agreed unanimously on only 35.8% of tweets (κ = 0.32).
- **Preprocessing:** open `src/preprocess.py` and explain each rule and why it exists.

## 3. Baseline and why (1 min)
Show `src/baselines.py`.
- **Baseline E1:** TF-IDF word unigrams + logistic regression.
  - It is simple, transparent and the standard starting point.
  - It was also used for the KINNEWS dataset.
- **TF-IDF** gives high weight to words that are frequent in a tweet but rare across tweets.
- **Logistic regression** learns one weight per word per class, then applies a softmax.
- **C** is the inverse regularisation strength, tuned on dev only.

## 4. Experiments: what changed and what I learned (2.5 min)
Show Table 3 and `experiments.png`.

**E2 (bigrams):** almost no gain, because most bigrams appear only once in 3.3k tweets.

**E3 (character 2–5-grams):** the biggest jump, +3.4 points.
- Character pieces like "koze" are shared across word forms.
- They also help with unseen words and spelling variation.

**E4–E6 (word + char features, then LogReg / NB / SVM):**
- The classifiers are within 0.4 points of each other, so the representation matters more than the classifier.
- E6 beats the baseline with McNemar p = 0.0002, so the gain is real, not luck.

**Ablations:**
- Removing emojis hurts.
- Removing stopwords hurts, because the list deletes "nta" (negation).

**BiLSTM (`src/bilstm.py`):**
- Walk through the pipeline: embedding → BiLSTM (both directions) → max-pool or attention → linear layer.
- Training: Adam, early stopping, 3 seeds.
- It was worse than the baseline.
- Why: 31% of test words are unknown, so a third of each tweet becomes the same <unk> vector. There is also too little data to learn embeddings, and the larger model over-fit faster.

**Transformers (Colab):**
- AfroXLMR and AfriBERTa were pretrained on Kinyarwanda; mBERT was not.
- Sub-word tokens fix the unknown-word problem.
- Report your numbers: did African pretraining beat mBERT? Did it beat the SVM?

## 5. Evaluation and errors (2 min)
**Metric choice:**
- Macro-F1 is the main metric because the classes are imbalanced and all three matter equally.
- Accuracy alone would reward predicting the majority class.
- Weighted-F1 is the official AfriSenti metric, reported for comparison.

**Confusion matrix:**
- Most errors are neutral↔negative.
- Positive↔negative flips are rare.

**Table 5:**
- Accuracy drops as unseen words increase.
- Questions and short tweets are harder.

**Walk through 3–4 errors from the report:**
- Negation ("ntacyo itwaye ikibazo").
- Implicit news (masks no longer mandatory).
- Humour with 😭.
- The Bible verse.

**Link to label noise:** humans disagree on the same kinds of tweets.

## 6. Live app (1.5 min)
**How the app connects to the model:**
- It loads the saved pipeline (`best_classical.joblib`).
- It runs the SAME `clean()` function used in training.
- It shows scores per class and the preprocessed text.

**Try inputs:**
- A clear positive, e.g. "Murakoze cyane…".
- A clear negative.
- A neutral news sentence.
- A failure case (sarcasm or a question). Explain why it fails.

## 7. Limitations and wrap-up (0.5 min)
- Small, noisy data from one domain (tweets).
- Negation and sarcasm remain unsolved.
- Next steps: KinyaBERT, soft labels from the 3 annotators, and more data.

---

## Viva questions you should be able to answer
- **Why macro-F1 and not accuracy?** The classes are imbalanced, and macro-F1 weights every class equally.
- **What does TF-IDF do?** It computes term frequency × log(N / document frequency). Sublinear TF uses 1 + log(tf).
- **What is `char_wb`?** Character n-grams taken only inside word boundaries, with padding spaces.
- **Why `class_weight="balanced"`?** It up-weights the smaller positive class in the loss.
- **What is the difference between LinearSVC and logistic regression?** SVC uses hinge loss and maximises the margin. LR uses log loss and gives probabilities.
- **Why does the app softmax the SVM scores?** To show relative confidence. These are not calibrated probabilities.
- **What does bidirectional mean?** One LSTM reads left→right and another right→left, and their states are concatenated (64 + 64 = 128).
- **How does attention work here?** Each timestep gets a score vᵀ tanh(W h_t). A softmax turns the scores into weights, and the sentence vector is the weighted sum of the states.
- **Why early stopping, and why 3 seeds?** Early stopping prevents over-fitting. Small data makes single runs noisy, so we average over seeds.
- **How do you prevent test leakage?** All tuning and model selection happen on dev. The vocabulary is built from train only.
- **What is fine-tuning?** A classification head is added on the [CLS]/first token, and all weights are updated with a small learning rate plus warm-up.
- **What is MAFT (AfroXLMR)?** Continued masked-LM pretraining of XLM-R on African languages.
- **What is McNemar's test?** It counts tweets where only one of the two models is right and tests whether that split is 50/50.
- **What is Fleiss' κ?** Agreement among several raters, corrected for chance: 0.32 is "fair".
