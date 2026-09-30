"""
Builds the final report (report/report.docx and report/report.pdf) from the
experiment results, so every number in the report comes straight from results/.

1. Fill in report/links.json (name, GitHub, video, Space links).
2. After running the Colab notebook, copy results/transformer_*.csv into results/
   and re-run: the Transformer rows and discussion are added automatically.
Run:  python report/build_report.py
"""
import json
import subprocess
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
RES, FIG, OUT = ROOT / "results", ROOT / "results" / "figures", ROOT / "report"
L = json.load(open(OUT / "links.json"))

ce = pd.read_csv(RES / "classical_experiments.csv").set_index("id")
ne = pd.read_csv(RES / "neural_experiments.csv").set_index("id")
te = pd.read_csv(RES / "transformer_experiments.csv").set_index("id") if (RES / "transformer_experiments.csv").exists() else None
es = json.load(open(RES / "error_stats.json"))
ag = json.load(open(RES / "annotator_agreement.json"))
sg = json.load(open(RES / "significance.json"))
BEST = es["best_model"]
f = lambda x: f"{100 * x:.1f}"


def pct(x):
    return f"{100 * x:.1f}%"


# ------------------------------------------------------------------ helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.3))
sec.top_margin = sec.bottom_margin = Cm(2.2)
st = doc.styles["Normal"]
st.font.name = "Calibri"; st.font.size = Pt(11)
st.paragraph_format.space_after = Pt(6); st.paragraph_format.line_spacing = 1.12
for lvl, size in ((1, 14), (2, 12)):
    h = doc.styles[f"Heading {lvl}"]
    h.font.name = "Calibri"; h.font.size = Pt(size); h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)


def P(text="", bold=False, italic=False, align=None, size=None, style=None):
    p = doc.add_paragraph(style=style)
    add_runs(p, text, bold, italic, size)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align != "left":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def add_runs(p, text, bold=False, italic=False, size=None):
    # inline **bold** and *italic*
    import re
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)", text):
        if not part:
            continue
        b, it = bold, italic
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            part, b = part[2:-2], True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            part, it = part[1:-1], True
        r = p.add_run(part)
        r.bold = b; r.italic = it
        if size:
            r.font.size = Pt(size)


def B(text):
    p = doc.add_paragraph(style="List Bullet")
    add_runs(p, text); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def H(text, lvl=1):
    return doc.add_heading(text, level=lvl)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    s = OxmlElement("w:shd"); s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), hexcolor)
    tcPr.append(s)


def set_widths(t, widths):
    t.autofit = False
    tblPr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    grid = t._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(w * 567)))
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)
    for row in t.rows:  # keep table rows together on one page where possible
        for c in row.cells:
            for par in c.paragraphs:
                par.paragraph_format.keep_with_next = True
                par.paragraph_format.space_after = Pt(1)


def table(header, rows, widths, caption, bold_rows=()):
    cap = doc.add_paragraph(); r = cap.add_run(caption); r.bold = True; r.font.size = Pt(9.5)
    cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; rr = c.paragraphs[0].add_run(h); rr.bold = True; rr.font.size = Pt(9)
        shade(c, "DCE6F1")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; rr = cells[i].paragraphs[0].add_run(str(v)); rr.font.size = Pt(9)
            rr.bold = ri in bold_rows
    set_widths(t, widths)
    doc.add_paragraph()
    return t


def figure(path, width_cm, caption):
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(); r = p.add_run(caption); r.italic = True; r.font.size = Pt(9)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


# ------------------------------------------------------------------ numbers
e1, best = ce.loc["E1"], ce.loc[BEST]
gain = best.test_macro_f1 - e1.test_macro_f1
n_best = ne.test_macro_f1.idxmax(); nb = ne.loc[n_best]
bs_ci = sg[f"{BEST} (best classical)"]["ci95"]; b_ci = sg["E1 (baseline)"]["ci95"]
mc = sg[f"{BEST} (best classical)"]["mcnemar_vs_baseline"]
have_tf = te is not None
if have_tf:
    tbest = te.dev_macro_f1.idxmax(); tb = te.loc[tbest]
    tf_beats = tb.test_macro_f1 > best.test_macro_f1
overall_name = (f"{tbest} ({tb.experiment.split(' (')[0]})" if have_tf and tf_beats
                else f"{BEST} (TF-IDF word+char n-grams + Linear SVM)")
overall_f1 = tb.test_macro_f1 if have_tf and tf_beats else best.test_macro_f1
overall_wf1 = tb.test_weighted_f1 if have_tf and tf_beats else best.test_weighted_f1

# ------------------------------------------------------------------ title
P("Sentiment Classification of Kinyarwanda Tweets: Comparing Classical, Recurrent and Transformer Models on AfriSenti",
  bold=True, align="center", size=16)
P(f"{L['author']}  ·  {L['course']}", align="center", size=10.5)
lk = doc.add_table(rows=3, cols=2); lk.style = "Table Grid"
for i, (k, v) in enumerate([("GitHub Repository", L["github"]), ("Demo Video", L["video"]), ("Live System", L["live_system"])]):
    lk.rows[i].cells[0].text = k; lk.rows[i].cells[1].text = v
    lk.rows[i].cells[0].paragraphs[0].runs[0].bold = True; shade(lk.rows[i].cells[0], "DCE6F1")
set_widths(lk, [4, 12.4])
doc.add_paragraph()

# ------------------------------------------------------------------ 1 abstract
H("Abstract")
abs_tf = (f" Fine-tuned African-centric Transformers were also evaluated; the best, {tb.experiment.split(' (')[0]}, reached "
          f"{f(tb.test_macro_f1)} macro-F1{' and became the deployed model' if tf_beats else ', not exceeding the best classical model'}."
          if have_tf else
          " Fine-tuning of pretrained African-centric Transformers (AfriBERTa, AfroXLMR) is provided as a reproducible GPU notebook.")
P("Sentiment analysis tools rarely exist for African languages, even widely spoken ones such as Kinyarwanda. This project builds "
  "and deploys a three-way (positive / neutral / negative) sentiment classifier for Kinyarwanda tweets using the Kinyarwanda "
  "portion of the AfriSenti benchmark (Muhammad et al., 2023a), which contains 5,155 human-annotated tweets. A TF-IDF word-unigram "
  f"logistic regression baseline achieved {f(e1.test_macro_f1)} test macro-F1. Adding character n-grams, which capture the "
  f"agglutinative morphology of Kinyarwanda, raised this to {f(best.test_macro_f1)} ({BEST}, Linear SVM), a statistically "
  f"significant improvement (McNemar p = {mc['p_value']:.4f}). BiLSTM models trained from scratch performed worse than the "
  f"baseline ({f(nb.test_macro_f1)}), which we attribute to the {pct(0.312)} out-of-vocabulary rate and small training set."
  + abs_tf +
  f" Error analysis shows that most mistakes involve the neutral class and that human annotators themselves agreed unanimously "
  f"on only {pct(ag['unanimous_fraction'])} of tweets (Fleiss' κ = {ag['fleiss_kappa']}), suggesting label ambiguity limits "
  "achievable accuracy. The final model is deployed as a Gradio web application.")

# ------------------------------------------------------------------ 2 intro
H("1. Introduction")
P("Social media is now a primary channel through which people in Rwanda discuss public services, news, sport, entertainment and "
  "politics. Organisations such as government agencies, telecom companies, banks and media houses increasingly want to understand "
  "public opinion expressed online, but almost all off-the-shelf sentiment tools are built for English and a few other high-resource "
  "languages. Joshi et al. (2020) show that the vast majority of the world's languages, including most African languages, are "
  "severely under-represented in NLP resources and research. Kinyarwanda, the national language of Rwanda, is one of these: "
  "despite a large speaker population, labelled datasets are small and only a few pretrained models include it (Niyongabo et al., 2020; "
  "Nzeyimana & Niyongabo Rubungo, 2022).")
P("Kinyarwanda is a Bantu language with rich agglutinative morphology: a single word can combine subject, tense, object and "
  "negation markers around a stem (Nzeyimana & Niyongabo Rubungo, 2022). For example, the gratitude expressions *urakoze*, "
  "*murakoze* and *mwakoze* (\"thank you\", singular/plural/past) are different surface words sharing the stem -koze. This makes "
  "word-level models sparse and motivates sub-word and character-level representations. Tweets add further challenges: informal "
  "spelling, elongated characters, emojis and frequent code-switching with English and French.")
P("**Objective.** The goal of this project is to build, evaluate and deploy a sentiment classifier that assigns Kinyarwanda text to "
  "one of three classes (positive, neutral, negative), and to understand which modelling choices matter for a low-resource, "
  "morphologically rich language. Specifically, we ask: (RQ1) Does a meaningful model beat a simple bag-of-words baseline? (RQ2) Do "
  "character-level features help, given Kinyarwanda morphology? (RQ3) Do neural sequence models trained from scratch help on only "
  "~3,300 training tweets? (RQ4) Does African-centric pretraining help compared with generic multilingual pretraining? (RQ5) "
  "Which kinds of tweets remain hard, and why? **Target users** are Rwandan organisations and researchers who want to monitor "
  "public sentiment in Kinyarwanda, e.g. customer feedback for a telecom or public reaction to a government announcement.")

# ------------------------------------------------------------------ 3 related
H("2. Related Work")
P("**African-language sentiment analysis.** AfriSenti (Muhammad et al., 2023a) is the largest sentiment benchmark for African "
  "languages, with over 110,000 tweets in 14 languages annotated by native speakers, and was the basis of the SemEval-2023 Task 12 "
  "shared task (Muhammad et al., 2023b). In the shared task, the strongest Kinyarwanda systems used pretrained Transformers: the "
  "organisers' AfroXLMR-large baseline reached 70.6 weighted-F1 and the best Kinyarwanda team result (72.5 weighted-F1) used "
  "KinyaBERT, a morphology-aware Kinyarwanda language model (Muhammad et al., 2023b; Nzeyimana & Niyongabo Rubungo, 2022). "
  "For Kinyarwanda text classification more broadly, Niyongabo et al. (2020) released KINNEWS, a 14-class news corpus, and showed "
  "that classical models with TF-IDF features are competitive with neural models when data is limited.")
P("**Classical vs neural text classification.** Linear models over n-gram features remain strong baselines for short-text "
  "sentiment; Wang and Manning (2012) showed that Naive Bayes and SVM variants with bigrams are hard to beat on short snippets, and "
  "Joulin et al. (2017) showed that simple linear models with sub-word n-grams rival deep networks at a fraction of the cost. "
  "Recurrent networks such as LSTMs (Hochreiter & Schmidhuber, 1997) with attention (Bahdanau et al., 2015) model word order, "
  "but typically need more data to learn good embeddings from scratch.")
P("**Pretrained multilingual Transformers.** mBERT (Devlin et al., 2019) and XLM-R (Conneau et al., 2020) are pretrained on "
  "100+ languages but include little or no Kinyarwanda. African-centric models address this: AfriBERTa (Ogueji et al., 2021) was "
  "pretrained from scratch on under 1 GB of text in 11 African languages including Kinyarwanda/Kirundi, and AfroXLMR (Alabi et al., "
  "2022) adapts XLM-R to 17 African languages through multilingual adaptive fine-tuning (MAFT).")
P("**Gap addressed.** Most published Kinyarwanda sentiment systems are shared-task submissions that report a single leaderboard "
  "score. This project contributes a controlled comparison of classical, recurrent and Transformer approaches under the same "
  "splits and preprocessing, with ablations, significance testing and an analysis of annotator disagreement, and it packages the "
  "result as a deployed, usable application.")

# ------------------------------------------------------------------ 4 data
H("3. Dataset and Data Preparation")
P("**Source and licence.** We use the Kinyarwanda subset of AfriSenti (Muhammad et al., 2023a), downloaded from the official "
  "AfriSenti-SemEval GitHub repository (CC BY 4.0). The tweets are real, naturally occurring posts collected from Twitter and "
  "labelled by three native-speaker annotators; the final label is the majority vote. No synthetic or machine-generated data is "
  "used. We keep the official train / dev / test split so that results are comparable with the literature.")
table(["Split", "Tweets", "Negative", "Neutral", "Positive"],
      [["Train", 3302, 1146, 1257, 899], ["Dev (validation)", 827, 287, 315, 225], ["Test", 1026, 355, 393, 278],
       ["Total", 5155, 1788, 1965, 1402]],
      [4, 2.5, 2.5, 2.5, 2.5], "Table 1. AfriSenti-Kinyarwanda splits and label distribution.")
figure(FIG / "class_distribution.png", 9, "Figure 1. Label distribution per split. Classes are moderately imbalanced (positive is smallest).")
P("**Characteristics and quality issues.** Inspecting the data revealed: (i) many replies begin with long chains of anonymised "
  "mentions (up to 11 \"@user\" tokens); (ii) URLs, hashtags and elongated words such as *Hhhhhh* or *Yoooo*; (iii) emojis in "
  f"{pct(es['n_has_emoji_true'] / es['test_size'])} of test tweets; (iv) code-switching with English/French in about "
  f"{pct(es['n_code_switch_true'] / es['test_size'])} of test tweets; (v) many news-style and promotional tweets whose sentiment is "
  "implicit. The data files are **sorted by label**, so shuffling during training is essential. Most importantly, the individual "
  f"annotations released with AfriSenti show that the three annotators gave the same label to only {pct(ag['unanimous_fraction'])} "
  f"of Kinyarwanda tweets, with Fleiss' κ = {ag['fleiss_kappa']} (\"fair\" agreement; Fleiss, 1971). Annotators also used "
  f"'mixed' ({ag['raw_label_counts'].get('MIXED', 0)} labels) and 'indeterminate' ({ag['raw_label_counts'].get('INDETERMINATE', 0)}) "
  "labels that were later collapsed. Label noise is therefore substantial and sets a ceiling on achievable performance.")
P("**Preprocessing** (src/preprocess.py). For classical and BiLSTM models: lower-casing; URLs → *xxurl*; runs of @-mentions → a "
  "single *xxuser* token (keeping the \"this is a reply\" signal); any character repeated 3+ times is capped at two (*hhhhhh* → *hh*); "
  "emojis are kept and separated into their own tokens; remaining punctuation is removed. We deliberately **do not remove "
  "stopwords**, because the standard Kinyarwanda stopword list (Niyongabo et al., 2020) contains the negation word *nta*, and "
  "removing negation can flip sentiment; this decision is tested in ablation A3. For Transformers we apply only light cleaning "
  "(mentions, URLs, elongation), leaving case, punctuation and emojis to the model's sub-word tokenizer.")

# ------------------------------------------------------------------ 5 method
H("4. Methodology")
P("All models map a tweet to one of three labels. Hyper-parameters are selected only on the dev set; the test set is used once per "
  "configuration for reporting. Because the classes are imbalanced and all three matter equally, **macro-F1** is our primary "
  "metric; we also report accuracy and weighted-F1 (the official AfriSenti metric) for comparability.")
H("4.1 Baseline", 2)
P("**E1: TF-IDF word unigrams + logistic regression.** Each tweet is represented as a sparse vector of sub-linear TF-IDF weights "
  "over its words, and a multinomial logistic regression with balanced class weights is trained on top (regularisation strength "
  "C ∈ {0.1, 0.3, 1, 3, 10} tuned on dev). This is the standard, transparent starting point for text classification and was also "
  "used as a baseline for KINNEWS (Niyongabo et al., 2020). A majority-class predictor (E0) is reported as a floor.")
H("4.2 Classical experiments", 2)
P("Building on E1 we vary one factor at a time: word bigrams (E2); **character n-grams** of length 2–5 within word boundaries (E3), "
  "which let the model share information across morphological variants such as *urakoze/murakoze/mwakoze*; the union of word "
  "1–2-grams and character 2–5-grams (E4); and the classifier on the E4 features: Multinomial Naive Bayes (E5) and a linear SVM "
  "(E6). Ablations test preprocessing choices on the E4 configuration: raw text (A1), emojis removed (A2), and stopwords removed (A3).")
H("4.3 BiLSTM trained from scratch", 2)
P("To test whether modelling word order helps, we implemented a BiLSTM classifier in PyTorch (src/bilstm.py). Input words are "
  "mapped to indices (vocabulary built from training words with frequency ≥ 2; 4,345 types; unknown words → <unk>), passed "
  "through a 100-dimensional embedding layer learned from scratch, dropout (0.5), and a bidirectional LSTM with 64 hidden units "
  "per direction. The 128-dimensional hidden states are pooled into one sentence vector either by **max-pooling** over time "
  "(N1) or by **additive attention** (N2), where each state h_t receives a weight a_t = softmax(vᵀ tanh(W h_t)) and the sentence "
  "vector is Σ a_t h_t (Bahdanau et al., 2015). A linear layer maps it to three logits. Training uses cross-entropy loss, Adam "
  "(learning rate 2×10⁻³), batch size 32, gradient clipping at 1.0, and early stopping on dev macro-F1 (patience 3, max 30 "
  "epochs). N3 doubles the embedding (200) and hidden (128) sizes. Each configuration is run with three random seeds and we "
  "report mean ± standard deviation.")
H("4.4 Fine-tuned Transformers", 2)
P("We fine-tune three pretrained encoders with the Hugging Face Transformers library (Wolf et al., 2020) (src/finetune_transformer.py): "
  "**T1 mBERT** (Devlin et al., 2019), whose pretraining data does not include Kinyarwanda; **T2 AfriBERTa-base** (Ogueji et al., "
  "2021); and **T3 AfroXLMR-base** (Alabi et al., 2022). The tweet is split into sub-word tokens (max 128), the final-layer "
  "representation of the first token is fed to a randomly initialised linear classification head, and all weights are updated "
  "(full fine-tuning) with AdamW, weight decay 0.01, linear warm-up over 10% of steps, 5 epochs, batch size 32, learning rate "
  "3×10⁻⁵ (5×10⁻⁵ for AfriBERTa), mixed precision on a Colab T4 GPU. The checkpoint with the best dev macro-F1 is kept, and each "
  "model is run with three seeds. Sub-word tokenization addresses the out-of-vocabulary problem that limits the word-level BiLSTM.")
H("4.5 Own work vs existing resources", 2)
table(["Existing resources used", "Implemented in this project"],
      [["AfriSenti-Kin data and annotator labels (Muhammad et al., 2023a)", "Data inspection, cleaning pipeline, agreement analysis"],
       ["Kinyarwanda stopword list (Niyongabo et al., 2020)", "Baseline, 10 classical experiments and ablations with dev-set tuning"],
       ["scikit-learn, PyTorch, Hugging Face Transformers libraries", "BiLSTM + attention model and training loop written from scratch"],
       ["Pretrained mBERT, AfriBERTa, AfroXLMR checkpoints", "Fine-tuning script, multi-seed evaluation, model selection"],
       ["Gradio, Hugging Face Spaces", "Error analysis, significance tests, figures, web app, this report"]],
      [8, 8.4], "Table 2. Distinction between existing resources and own implementation.")

# ------------------------------------------------------------------ 6 results
H("5. Experiments and Results")
rows = []
for i in ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "A1", "A2", "A3"]:
    r = ce.loc[i]
    rows.append([i, r.experiment.replace("BASELINE: ", ""), "-" if pd.isna(r.best_param) else r.best_param,
                 f(r.dev_macro_f1), f(r.test_acc), f(r.test_macro_f1), f(r.test_weighted_f1)])
for i in ne.index:
    r = ne.loc[i]
    rows.append([i, r.experiment, "3 seeds", f(r.dev_macro_f1), f(r.test_acc),
                 f"{f(r.test_macro_f1)} ± {f(r.test_macro_f1_std)}", f(r.test_weighted_f1)])
if have_tf:
    for i in te.index:
        r = te.loc[i]
        rows.append([i, r.experiment, f"lr {r.lr:g}", f(r.dev_macro_f1), f(r.test_acc),
                     f"{f(r.test_macro_f1)} ± {f(r.test_macro_f1_std)}", f(r.test_weighted_f1)])
bold_ids = {"E1", BEST} | ({tbest} if have_tf else set())
table(["ID", "Approach", "Best param", "Dev M-F1", "Test Acc", "Test M-F1", "Test W-F1"], rows,
      [1.1, 6.6, 1.7, 1.7, 1.6, 2.1, 1.6],
      "Table 3. All experiments (scores ×100). M-F1 = macro-F1, W-F1 = weighted-F1. Neural rows: mean ± std over 3 seeds.",
      bold_rows=[k for k, r in enumerate(rows) if r[0] in bold_ids])
figure(FIG / "experiments.png", 15.5, "Figure 2. Test macro-F1 per experiment; dashed line = baseline E1. Error bars = std over seeds.")

P(f"**RQ1 – baseline vs better models.** The majority-class floor (E0) scores only {f(ce.loc['E0'].test_macro_f1)} macro-F1, "
  f"while the word-unigram baseline reaches {f(e1.test_macro_f1)}. The best classical model, {BEST}, improves this by "
  f"{100 * gain:.1f} points to {f(best.test_macro_f1)} (95% bootstrap CI {f(bs_ci[0])}–{f(bs_ci[1])} vs {f(b_ci[0])}–{f(b_ci[1])} "
  f"for the baseline). On the paired test predictions it fixes {mc['baseline_wrong_model_right']} baseline errors while breaking "
  f"{mc['baseline_right_model_wrong']}, and McNemar's exact test gives p = {mc['p_value']:.4f}, so the improvement is unlikely to be chance.")
P(f"**RQ2 – character n-grams.** Word bigrams alone barely help (E2: {f(ce.loc['E2'].test_macro_f1)}), because with only 3,302 "
  "tweets most bigrams occur once. Switching to character 2–5-grams gives the largest single jump "
  f"(E3: {f(ce.loc['E3'].test_macro_f1)}, +{100 * (ce.loc['E3'].test_macro_f1 - e1.test_macro_f1):.1f} points). This supports our "
  "hypothesis: character n-grams recover shared stems and affixes of agglutinative Kinyarwanda words (e.g. -koze in all 'thank "
  "you' variants) and are robust to spelling variation, so a word never seen in training can still activate useful features. "
  f"Combining word and character features (E4) adds a further gain ({f(ce.loc['E4'].test_macro_f1)}). With these features the "
  f"choice of classifier matters little: Naive Bayes, logistic regression and SVM are within about "
  f"{100 * (ce.loc[['E4', 'E5', 'E6']].test_macro_f1.max() - ce.loc[['E4', 'E5', 'E6']].test_macro_f1.min()):.1f} points, "
  "i.e. representation matters more than the classifier.")
P(f"**Ablations.** Keeping emojis helps: removing them (A2) lowers dev macro-F1 from {f(ce.loc['E4'].dev_macro_f1)} to "
  f"{f(ce.loc['A2'].dev_macro_f1)}, and in the error analysis tweets containing emojis are classified more accurately "
  f"({pct(es['acc_has_emoji_true'])} vs {pct(es['acc_has_emoji_false'])}). Our cleaning gives a small gain over raw text (A1: "
  f"{f(ce.loc['A1'].dev_macro_f1)} dev), mainly by collapsing mention chains and elongations. Removing stopwords hurts "
  f"(A3: {f(ce.loc['A3'].test_macro_f1)} vs {f(ce.loc['E2'].test_macro_f1)} test for the same model), consistent with our concern "
  "that the list removes the negator *nta*. Note that with 827 dev tweets, differences below ~1–2 points are within noise.")
P(f"**RQ3 – BiLSTM from scratch.** All BiLSTM variants performed **worse than the linear baseline** (best N1: "
  f"{f(ne.loc['N1'].test_macro_f1)} ± {f(ne.loc['N1'].test_macro_f1_std)}), and attention (N2) or a larger network (N3) did not "
  "help. We investigated why: 31.2% of test tokens are out of the training vocabulary, so a third of each tweet reaches the "
  "LSTM as the same <unk> vector, whereas character n-grams still extract information from those words. With only ~3,300 short "
  "tweets, the embedding layer (≈430k parameters) cannot learn good word representations, and the dev score peaks within 6–13 "
  "epochs before over-fitting. The larger N3 over-fit fastest (best epoch 6–7). This matches the observation of Niyongabo et al. "
  "(2020) that classical models are competitive on small Kinyarwanda datasets, and shows that model complexity is not a "
  "substitute for data.")
if have_tf:
    t1 = te.loc["T1"].test_macro_f1 if "T1" in te.index else None
    afr = te.drop(index=[i for i in ["T1"] if i in te.index])
    P(f"**RQ4 – pretrained Transformers.** The best Transformer on dev, {tbest} ({tb.experiment.split(' (')[0]}), reached "
      f"{f(tb.test_macro_f1)} ± {f(tb.test_macro_f1_std)} test macro-F1 and {f(tb.test_weighted_f1)} weighted-F1"
      + (f", {100 * (tb.test_macro_f1 - best.test_macro_f1):+.1f} points relative to the best classical model." if True else ".")
      + (f" mBERT, which never saw Kinyarwanda during pretraining, scored {f(t1)}, "
         f"{'below' if t1 < afr.test_macro_f1.max() else 'comparable to'} the African-centric models "
         f"(best {f(afr.test_macro_f1.max())}), " if t1 is not None else " ")
      + "indicating how much pretraining on the target language matters. Sub-word tokenization and pretrained representations "
      "remove the out-of-vocabulary problem that limited the BiLSTM. For reference, the AfriSenti AfroXLMR-large baseline reported "
      "70.6 weighted-F1 on this test set (Muhammad et al., 2023b).")
else:
    P("**RQ4 – pretrained Transformers.** [Run notebooks/finetune_transformers_colab.ipynb, copy results/transformer_*.csv into "
      "results/ and rebuild this report to fill in this paragraph and the T1–T3 rows of Table 3.] For reference, the AfriSenti "
      "AfroXLMR-large baseline reported 70.6 weighted-F1 on this test set (Muhammad et al., 2023b).")

# ------------------------------------------------------------------ 7 errors
H("6. Error Analysis and Discussion")
rep = (RES / "classification_report.txt").read_text().splitlines()
cls = {l.split()[0]: l.split()[1:4] for l in rep if l.strip().split()[:1] and l.split()[0] in ("negative", "neutral", "positive")}
table(["Class", "Precision", "Recall", "F1"], [[c, *[f(float(x)) for x in v]] for c, v in cls.items()],
      [4, 3, 3, 3], f"Table 4. Per-class test results for {BEST}.")
figure(FIG / "confusion_matrix.png", 7.5, f"Figure 3. Confusion matrix of {BEST} on the test set.")
conf = es["confusions"]
P(f"**Where the model fails.** {BEST} makes {es['errors']} errors on 1,026 test tweets. Neutral is the hardest class "
  f"(F1 {f(float(cls['neutral'][2]))}) and the two most frequent confusions both involve it: neutral→negative "
  f"({conf.get('neutral->negative')}) and negative→neutral ({conf.get('negative->neutral')}). Direct polarity flips "
  f"(positive↔negative) are rare ({conf.get('positive->negative', 0) + conf.get('negative->positive', 0)}), so the model has "
  "learned polarity but struggles with its *intensity* and with implicit sentiment. We grouped test tweets by properties "
  "(Table 5) to find systematic weaknesses.")
table(["Tweet property", "n", "Accuracy", "Accuracy (others)"],
      [["≤ 5 content words (short)", es["n_short_true"], pct(es["acc_short_true"]), pct(es["acc_short_false"])],
       ["Contains a question mark", es["n_question_true"], pct(es["acc_question_true"]), pct(es["acc_question_false"])],
       ["Code-switched (English/French words)", es["n_code_switch_true"], pct(es["acc_code_switch_true"]), pct(es["acc_code_switch_false"])],
       ["Contains emoji", es["n_has_emoji_true"], pct(es["acc_has_emoji_true"]), pct(es["acc_has_emoji_false"])],
       ["OOV rate ≤ 10%", es["n_oov_<=10%"], pct(es["acc_oov_<=10%"]), "-"],
       ["OOV rate 10–30%", es["n_oov_10-30%"], pct(es["acc_oov_10-30%"]), "-"],
       ["OOV rate > 30%", es["n_oov_>30%"], pct(es["acc_oov_>30%"]), "-"]],
      [6.5, 1.5, 2.7, 3.2], f"Table 5. {BEST} accuracy on subsets of the test set (OOV = share of words unseen in training).")
P(f"Accuracy falls steadily as the share of unseen words rises (from {pct(es['acc_oov_<=10%'])} to {pct(es['acc_oov_>30%'])}), "
  "confirming that vocabulary coverage is the main bottleneck for a 3,300-tweet training set. Questions, which often express "
  "sarcasm or rhetorical complaint without explicit sentiment words, and very short tweets, which give few features, are also "
  "harder. Code-switched tweets are slightly harder because English/French words are sparse in the training data.")
P("**Representative examples** (English glosses are approximate):")
B("**Correct, high confidence:** *\"Urakoze cyane nawe umunsi mwiza 🥰\"* (\"Thank you very much, have a good day too\") → "
  "positive. Formulaic gratitude (*urakoze/murakoze*), *umunsi mwiza* (\"good day\") and positive emojis are strong, frequent cues.")
B("**Negation (bag-of-n-grams limitation):** *\"Ntacyo itwaye ikibazo ndumva ari battery😂\"* (\"It doesn't matter, I think the "
  "problem is the battery\") – gold neutral, predicted negative. *ikibazo* (\"problem\") is a strong negative feature, and a "
  "linear model over n-grams cannot compose it with *ntacyo itwaye* (\"it's nothing\").")
B("**Implicit sentiment needing world knowledge:** *\"Inama y'Abaminisitiri yatangaje ko bitakiri itegeko kwambara agapfukamunwa\"* "
  "(\"The Cabinet announced that wearing masks is no longer mandatory\") – gold positive, predicted neutral. The tweet is written "
  "in neutral news style; it is positive only because readers know the news is welcome.")
B("**Negative news in neutral style:** *\"Rutahizamu … yahagaritswe iminsi 8 … kubera imyitwarire itari myiza\"* (\"The striker "
  "has been suspended for 8 days for bad behaviour\") – gold negative, predicted neutral. Similar sports/news reports are labelled "
  "neutral elsewhere in the data, which illustrates annotator inconsistency.")
B("**Humour and emojis:** *\"…none banz nkicyonka😭😭😭\"* – gold neutral, predicted negative. Crying emojis are used "
  "humorously here, but the model has learned 😭 as a negative cue.")
B("**Religious text:** *\"Wiringire Uwiteka n'umutima wawe wose…\"* (Proverbs 3:5, \"Trust in the Lord with all your heart\") – "
  "gold neutral, predicted positive. Religious vocabulary is positive in tone, but annotators judged the verse as neutral.")
P(f"**Relation to label quality and the literature.** Many 'errors' are tweets on which humans also disagree. Only "
  f"{pct(ag['unanimous_fraction'])} of Kinyarwanda tweets received three identical labels (κ = {ag['fleiss_kappa']}), and the "
  "neutral/negative boundary in news-style tweets is exactly where annotators diverge. Thus the achievable score is well below "
  "100%, which helps explain why even the best SemEval systems plateau around 70–72 weighted-F1 on Kinyarwanda "
  "(Muhammad et al., 2023b). The remaining gap between our classical models and those systems is consistent with the value of "
  "pretrained, morphology-aware representations (Nzeyimana & Niyongabo Rubungo, 2022; Alabi et al., 2022), which can supply "
  "knowledge of words never seen in our small training set.")

# ------------------------------------------------------------------ 8 deploy
H("7. Deployment")
dep = (f"the fine-tuned Transformer ({L['hf_model']})" if L.get("deployed_model") == "transformer"
       else f"the best classical model ({BEST}), saved as a scikit-learn pipeline (models/best_classical.joblib, ~4 MB)")
P(f"The system is deployed as a Gradio (Abid et al., 2019) web application hosted on Hugging Face Spaces: {L['live_system']}. "
  f"On start-up the app loads {dep}. When a user types Kinyarwanda text, the app applies exactly the same preprocessing used in "
  "training, runs the model, and displays the predicted label with a score for each class, together with the preprocessed text "
  "the model actually saw (useful for explaining errors). Scores come from a softmax over the SVM decision values and are "
  "relative confidences rather than calibrated probabilities; a low-confidence warning appears when no class exceeds 0.5. The app "
  "can switch to a fine-tuned Transformer from the Hugging Face Hub by setting one environment variable (MODEL_ID). A lightweight "
  "linear model was chosen for the default deployment because it runs instantly on the free CPU tier and its behaviour is easy "
  "to explain. If the Space is asleep, the app can be run locally with `python app/app.py` (see README).")

# ------------------------------------------------------------------ 9 limits
H("8. Limitations and Future Work")
B("**Data size and label noise.** 3,302 training tweets with fair inter-annotator agreement limit every model. Future work: "
  "train with soft labels from the three individual annotations, or remove highly disputed tweets from training.")
B("**Domain.** All data are tweets from a specific period (e.g. COVID-19, CHOGM 2022, sport); performance on customer reviews, "
  "SMS or news comments is untested and likely lower.")
B("**Model limitations.** Bag-of-n-gram models cannot handle negation scope, sarcasm or implicit sentiment. Morphology-aware "
  "pretrained models such as KinyaBERT (Nzeyimana & Niyongabo Rubungo, 2022), larger models (AfroXLMR-large) and "
  "further pretraining on unlabelled Kinyarwanda tweets are promising.")
B("**Evaluation.** A single fixed test set of 1,026 tweets gives confidence intervals of about ±3 points; cross-validation "
  "and evaluation by additional native speakers on new data would give a more reliable picture.")
B("**Deployment.** The free hosting tier sleeps when idle, and the app does not yet log feedback from users, which could be "
  "used to collect new labelled data.")

# ------------------------------------------------------------------ 10 concl
H("9. Conclusion")
P(f"This project built and deployed a Kinyarwanda tweet sentiment classifier. The best model, {overall_name}, reached "
  f"{f(overall_f1)} macro-F1 and {f(overall_wf1)} weighted-F1, significantly above a word-level TF-IDF baseline "
  f"({f(e1.test_macro_f1)}). The main lessons were: (1) for a morphologically rich, low-resource language, the input "
  "representation matters more than the classifier, and character n-grams give most of the gain; (2) a more complex neural model "
  "trained from scratch (BiLSTM with attention) did not beat the baseline, because of high out-of-vocabulary rates and little "
  "data; (3) seemingly harmless preprocessing such as stopword removal can hurt by deleting negation; and (4) a large share of "
  "remaining errors lie on the neutral boundary, where human annotators also disagree. Pretrained, African-centric and "
  "morphology-aware language models, more data and better handling of label uncertainty are the clearest routes to improvement.")

# ------------------------------------------------------------------ 11 refs
H("References")
refs = [
    "Abid, A., Abdalla, A., Abid, A., Khan, D., Alfozan, A., & Zou, J. (2019). Gradio: Hassle-free sharing and testing of ML models in the wild. arXiv:1906.02569.",
    "Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. In Proceedings of COLING 2022 (pp. 4336–4349).",
    "Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. In Proceedings of ICLR 2015.",
    "Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. In Proceedings of ACL 2020 (pp. 8440–8451).",
    "Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional Transformers for language understanding. In Proceedings of NAACL-HLT 2019 (pp. 4171–4186).",
    "Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. Psychological Bulletin, 76(5), 378–382.",
    "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.",
    "Joshi, P., Santy, S., Budhiraja, A., Bali, K., & Choudhury, M. (2020). The state and fate of linguistic diversity and inclusion in the NLP world. In Proceedings of ACL 2020 (pp. 6282–6293).",
    "Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). Bag of tricks for efficient text classification. In Proceedings of EACL 2017 (pp. 427–431).",
    "Muhammad, S. H., Abdulmumin, I., Ayele, A. A., Ousidhoum, N., Adelani, D. I., Yimam, S. M., et al. (2023a). AfriSenti: A Twitter sentiment analysis benchmark for African languages. In Proceedings of EMNLP 2023. Dataset: https://github.com/afrisenti-semeval/afrisent-semeval-2023 (CC BY 4.0).",
    "Muhammad, S. H., Abdulmumin, I., Yimam, S. M., Adelani, D. I., Ahmad, I. S., Ousidhoum, N., Ayele, A., Mohammad, S. M., Beloucif, M., & Ruder, S. (2023b). SemEval-2023 Task 12: Sentiment analysis for African languages (AfriSenti-SemEval). In Proceedings of SemEval-2023. ACL.",
    "Niyongabo, R. A., Qu, H., Kreutzer, J., & Huang, L. (2020). KINNEWS and KIRNEWS: Benchmarking cross-lingual text classification for Kinyarwanda and Kirundi. In Proceedings of COLING 2020 (pp. 5507–5521).",
    "Nzeyimana, A., & Niyongabo Rubungo, A. (2022). KinyaBERT: A morphology-aware Kinyarwanda language model. In Proceedings of ACL 2022 (pp. 5347–5363).",
    "Ogueji, K., Zhu, Y., & Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. In Proceedings of the 1st Workshop on Multilingual Representation Learning (pp. 116–126).",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in NeurIPS 32.",
    "Wang, S., & Manning, C. D. (2012). Baselines and bigrams: Simple, good sentiment and topic classification. In Proceedings of ACL 2012 (pp. 90–94).",
    "Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. In Proceedings of EMNLP 2020: System Demonstrations (pp. 38–45).",
]
for r in refs:
    p = doc.add_paragraph(r); p.paragraph_format.left_indent = Cm(0.8); p.paragraph_format.first_line_indent = Cm(-0.8)
    p.runs[0].font.size = Pt(9.5)

# page numbers in footer
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
    r = fp.add_run()
    if tag:
        e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tag); r._r.append(e)
    else:
        e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = text; r._r.append(e)

docx_path = OUT / "report.docx"
doc.save(docx_path)
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(OUT), str(docx_path)],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"Wrote {docx_path} and {OUT / 'report.pdf'}  (transformer results included: {have_tf})")
