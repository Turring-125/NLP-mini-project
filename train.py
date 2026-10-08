"""
train.py - Step 2: Text Preprocessing, Morphology & Classification
Cleans KCC farmer queries, compares Stemming vs Lemmatization, trains
TF-IDF + Naive Bayes baseline & Logistic Regression, and saves best model.
"""

import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
DATA_PATH = os.path.join("data", "kcc_queries.csv")

def demonstrate_morphology():
    """Demonstrate NLTK Stemming vs Lemmatization on real agricultural query terms."""
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    sample_words = [
        "infestation", "varieties", "dropping", "fertilizers", "spraying",
        "weedicides", "borers", "cultivation", "germinating", "diseases"
    ]
    print("\n" + "=" * 70)
    print("MORPHOLOGY ANALYSIS: STEMMING VS LEMMATIZATION (10 Real Farm Terms)")
    print("=" * 70)
    print(f"{'Original Term':16s} | {'Porter Stemmer':16s} | {'WordNet Lemmatizer':16s}")
    print("-" * 70)
    for w in sample_words:
        s = stemmer.stem(w)
        l = lemmatizer.lemmatize(w)
        print(f"{w:16s} | {s:16s} | {l:16s}")
    print("-" * 70)
    print("Note: Stemming chops affixes heuristically (e.g., 'varieti'), while")
    print("lemmatization uses a lexicon/WordNet to produce valid root words ('variety').")
    print("=" * 70 + "\n")

def clean_text(text: str, stop_words: set) -> str:
    """Lowercase, strip punctuation, digits, extra spaces, and stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

def main():
    demonstrate_morphology()
    
    print("Loading and cleaning KCC dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["questions", "answers", "category"]).drop_duplicates(subset=["questions"])
    
    # Drop categories with fewer than 50 examples
    counts = df["category"].value_counts()
    valid_cats = counts[counts >= 50].index
    df = df[df["category"].isin(valid_cats)].copy()
    print(f"Dataset size after cleaning: {len(df):,} queries across {len(valid_cats)} categories.")

    stop_words = set(stopwords.words("english"))
    df["clean_query"] = df["questions"].apply(lambda q: clean_text(q, stop_words))
    
    X = df["clean_query"]
    y = df["category"]
    
    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split sizes: Train={len(X_train):,}, Test={len(X_test):,}")

    # TF-IDF Feature Extraction
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_tr_vec = tfidf.fit_transform(X_train)
    X_te_vec = tfidf.transform(X_test)

    # 1. Baseline: Multinomial Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_tr_vec, y_train)
    nb_preds = nb.predict(X_te_vec)
    nb_acc = accuracy_score(y_test, nb_preds)
    nb_f1 = f1_score(y_test, nb_preds, average="macro")

    # 2. Main Model: Logistic Regression
    lr = LogisticRegression(max_iter=300, random_state=42)
    lr.fit(X_tr_vec, y_train)
    lr_preds = lr.predict(X_te_vec)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds, average="macro")

    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(f"Multinomial Naive Bayes : Accuracy = {nb_acc:.4f} ({nb_acc*100:.2f}%) | Macro-F1 = {nb_f1:.4f}")
    print(f"Logistic Regression     : Accuracy = {lr_acc:.4f} ({lr_acc*100:.2f}%) | Macro-F1 = {lr_f1:.4f}")

    print("\n--- DETAILED CLASSIFICATION REPORT (Logistic Regression) ---")
    print(classification_report(y_test, lr_preds))

    # Save Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y_test, lr_preds, ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix: Farmer Query Classification (Logistic Regression)")
    plt.tight_layout()
    cm_path = os.path.join("results", "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

    # Save metrics table
    res_df = pd.DataFrame([
        {"model": "Multinomial Naive Bayes", "accuracy": round(nb_acc, 4), "macro_f1": round(nb_f1, 4)},
        {"model": "Logistic Regression", "accuracy": round(lr_acc, 4), "macro_f1": round(lr_f1, 4)}
    ])
    res_df.to_csv(os.path.join("results", "classifier_results.csv"), index=False)

    # Save best model, vectorizer, and test split indices
    bundle = {
        "vectorizer": tfidf,
        "model": lr,
        "classes": list(lr.classes_),
        "stop_words": stop_words
    }
    model_path = os.path.join("models", "classifier_bundle.joblib")
    joblib.dump(bundle, model_path)
    
    # Save train & test splits for retrieval and pipeline evaluation
    train_df = df.loc[idx_tr, ["questions", "answers", "category", "clean_query"]]
    test_df = df.loc[idx_te, ["questions", "answers", "category", "clean_query"]]
    train_df.to_csv(os.path.join("data", "kcc_train.csv"), index=False)
    test_df.to_csv(os.path.join("data", "kcc_test.csv"), index=False)
    print(f"Saved trained classifier bundle to {model_path}")
    print("Saved train/test splits to data/kcc_train.csv and data/kcc_test.csv")

if __name__ == "__main__":
    main()
