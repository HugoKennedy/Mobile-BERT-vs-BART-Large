# MobileBERT vs BART-Large on Yelp Sentiment

This repo compares a **small fine-tuned model** (MobileBERT) with a **large zero-shot model** (BART-Large) on a binary sentiment task using Yelp reviews. 

---

## Goal

- Fine-tune **MobileBERT** on Yelp reviews and evaluate its accuracy.
- Use **BART-Large MNLI** as a **zero-shot** sentiment classifier on the same test set.
- Compare performance, speed, and practical trade-offs between:
  - *Small + task-specific fine-tuning* vs.
  - *Large + zero-shot general model*.

---

## Models

- **MobileBERT**: `google/mobilebert-uncased`  
  - Fine-tuned with **LoRA** (parameter-efficient fine-tuning). :contentReference[oaicite:1]{index=1}  
- **BART-Large**: `facebook/bart-large-mnli`  
  - Used via Hugging Face `pipeline("zero-shot-classification")` with labels `["positive", "negative"]`. :contentReference[oaicite:2]{index=2}  

Both models reach around **95–96% accuracy** on a 5k test sample (exact numbers are printed at the end of the notebook). :contentReference[oaicite:3]{index=3}  

---

## Dataset

- **Yelp Review Polarity** from Kaggle  
- Automatic download in the notebook via the Kaggle API. :contentReference[oaicite:4]{index=4}  
- Labels are converted to:
  - `0` = Negative  
  - `1` = Positive  

To stay within runtime limits, the notebook uses:
- **50,000** training reviews  
- **5,000** test reviews (random sample, `random_state=42`). :contentReference[oaicite:5]{index=5}  

---

## What the Notebook Does

In `Mobile_BERT_vs_BART_Large.ipynb`:

1. Installs dependencies (Transformers, Datasets, Evaluate, Kaggle, scikit-learn, PEFT). :contentReference[oaicite:6]{index=6}  
2. Downloads and unzips the Yelp dataset from Kaggle.  
3. Cleans and re-labels the data.  
4. Converts Pandas DataFrames to Hugging Face `Dataset` objects.  
5. Tokenizes text (with truncation) for MobileBERT.  
6. Fine-tunes MobileBERT using **LoRA** on the 50k training subset. :contentReference[oaicite:7]{index=7}  
7. Evaluates MobileBERT on the 5k test subset (accuracy + report).  
8. Runs BART-Large MNLI as a **zero-shot** classifier on the same test subset. :contentReference[oaicite:8]{index=8}  
9. Prints final accuracy and a short conclusion comparing both approaches. :contentReference[oaicite:9]{index=9}  

---

## Requirements

- Python 3.10+
- GPU recommended (training + BART-Large zero-shot is slow on CPU)
- Main libraries (installed in the notebook):

```bash
pip install "transformers[torch]" datasets evaluate kaggle scikit-learn peft
