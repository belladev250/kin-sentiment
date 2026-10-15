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

**Transformers (`src/finetune_transformer.py`, Colab T4):**
- Pipeline: tweet → light cleaning → sub-word tokenizer → 12-layer Transformer encoder → first-token vector → linear layer → 3 logits → softmax.
- Full fine-tuning: AdamW, lr 3e-5 (5e-5 AfriBERTa), weight decay 0.01, 10% warm-up, 5 epochs, batch 32, fp16; keep the epoch with the best dev macro-F1; 3 seeds.
- Why three models: they differ mainly in *pretraining data*, so the comparison isolates that factor.
- Results (test macro-F1, mean of 3 seeds):
  - T1 mBERT (no Kinyarwanda): 57.6 ± 0.7, no better than the baseline.
  - T2 AfriBERTa: 63.3 ± 0.3.
  - T3 AfroXLMR-base: 67.5 ± 0.7, +5.1 over the SVM; McNemar vs E6 p = 0.002.
- Lesson: pretraining on the language matters more than size or architecture. mBERT has to learn Kinyarwanda from 3,302 tweets.

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

**Transformer vs SVM (Tables 6–7, Figure 4):**
- T3 gains most on short tweets (+15 points accuracy), code-switched tweets (+8.5) and tweets with >30% unseen words (+7).
- T3 raises negative and positive recall, but neutral recall stays at about 53%. Neutral→negative is still the largest error.
- Accuracy on tweets all three annotators agreed on: about 75%. On 2-vs-1 tweets: about 64%.
- All five failure examples in the report are still wrong for T3, because they need world knowledge or consistent labels.

## 6. Live app (1.5 min)
**How the app connects to the model:**
- On start-up it downloads the fine-tuned T3 model from the Hugging Face Hub (`bellahug3/kin-sentiment-transformer`).
- It runs the same `light_clean()` used in fine-tuning, then the tokenizer, the model and a softmax over 3 logits.
- It shows the probability of each class and the cleaned text.
- Fallback: if the Transformer can't load, it uses the saved SVM (`best_classical.joblib`).

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
- **Why is mBERT so weak?** Kinyarwanda is not in its pretraining data, so its sub-word pieces carry no learned meaning for Kinyarwanda.
- **Why AfroXLMR over AfriBERTa?** It is XLM-R (trained on 100 languages, including English and French) plus extra pretraining on African languages. That helps with code-switching, and it has a larger vocabulary.
- **Why report the mean of 3 seeds but deploy one?** The mean shows how stable the method is. The deployed checkpoint is the seed with the best *dev* score, chosen without looking at test.
- **What is a logit / what does softmax do?** Logits are the raw class scores. Softmax turns them into probabilities that sum to 1.
- **What does warm-up do?** It raises the learning rate gradually over the first 10% of steps, so the new random head doesn't wreck the pretrained weights early on.
- **What does bidirectional mean?** One LSTM reads left→right and another right→left, and their states are concatenated (64 + 64 = 128).
- **How does attention work here?** Each timestep gets a score vᵀ tanh(W h_t). A softmax turns the scores into weights, and the sentence vector is the weighted sum of the states.
- **Why early stopping, and why 3 seeds?** Early stopping prevents over-fitting. Small data makes single runs noisy, so we average over seeds.
- **How do you prevent test leakage?** All tuning and model selection happen on dev. The vocabulary is built from train only.
- **What is fine-tuning?** A classification head is added on the [CLS]/first token, and all weights are updated with a small learning rate plus warm-up.
- **What is MAFT (AfroXLMR)?** Continued masked-LM pretraining of XLM-R on African languages.
- **What is McNemar's test?** It counts tweets where only one of the two models is right and tests whether that split is 50/50.
- **What is Fleiss' κ?** Agreement among several raters, corrected for chance: 0.32 is "fair".
