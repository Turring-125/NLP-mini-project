"""
retrieve.py - Step 3: TF-IDF Knowledge Retrieval Engine
Indexes training queries with TF-IDF, retrieves top-3 matching past queries
with expert advisory answers and cosine similarity, and evaluates Precision@1.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_PATH = os.path.join("data", "kcc_train.csv")
TEST_PATH = os.path.join("data", "kcc_test.csv")
BUNDLE_PATH = os.path.join("models", "retrieval_bundle.joblib")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

class FarmerQueryRetriever:
    """TF-IDF Cosine Similarity Search Engine over historical KCC queries."""

    def __init__(self, train_df: pd.DataFrame = None):
        if train_df is not None:
            self.train_df = train_df.reset_index(drop=True)
            self.vectorizer = TfidfVectorizer(max_features=15000, stop_words="english", ngram_range=(1, 2))
            self.matrix = self.vectorizer.fit_transform(self.train_df["questions"])
        else:
            self.train_df = None
            self.vectorizer = None
            self.matrix = None

    def save(self, path: str = BUNDLE_PATH):
        bundle = {
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "train_df": self.train_df
        }
        joblib.dump(bundle, path)
        print(f"Saved retrieval index bundle to {path}")

    @classmethod
    def load(cls, path: str = BUNDLE_PATH):
        bundle = joblib.load(path)
        retriever = cls()
        retriever.vectorizer = bundle["vectorizer"]
        retriever.matrix = bundle["matrix"]
        retriever.train_df = bundle["train_df"]
        return retriever

    def search(self, query: str, top_k: int = 3):
        """Return top_k similar past queries, expert answers, categories, and similarity scores."""
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "query": self.train_df.iloc[idx]["questions"],
                "answer": self.train_df.iloc[idx]["answers"],
                "category": self.train_df.iloc[idx]["category"],
                "similarity": float(sims[idx])
            })
        return results

    def evaluate_precision_at_1(self, test_df: pd.DataFrame, batch_size: int = 2000) -> float:
        """Precision@1: Fraction of test queries whose top retrieved query shares the same category."""
        print(f"Evaluating Precision@1 on full test set ({len(test_df):,} queries)...")
        correct = 0
        total = len(test_df)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = test_df.iloc[start:end]
            test_vecs = self.vectorizer.transform(batch["questions"])
            batch_sims = cosine_similarity(test_vecs, self.matrix)
            top1_idx = batch_sims.argmax(axis=1)

            retrieved_cats = self.train_df["category"].iloc[top1_idx].values
            true_cats = batch["category"].values
            correct += np.sum(retrieved_cats == true_cats)

        prec_1 = correct / total
        return float(prec_1)

def main():
    print("=" * 70)
    print("STEP 3: TF-IDF RETRIEVAL ENGINE")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Loaded {len(train_df):,} training QA pairs and {len(test_df):,} test QA pairs.")

    # Build and save retriever index
    retriever = FarmerQueryRetriever(train_df)
    retriever.save()

    # Evaluate Precision@1
    prec_1 = retriever.evaluate_precision_at_1(test_df)
    print(f"\nPrecision@1 on Test Set: {prec_1:.4f} ({prec_1*100:.2f}%)")
    
    # Save results
    res_df = pd.DataFrame([{"metric": "Precision@1", "score": round(prec_1, 4), "test_size": len(test_df)}])
    res_df.to_csv(os.path.join(RESULTS_DIR, "retrieval_results.csv"), index=False)

    # Demo queries (including sugarcane queries for irrigation advisory context)
    demo_queries = [
        "asking about control of red rot in sugarcane",
        "asking about recommended irrigation interval for sugarcane crops during summer",
        "asking about fertilizer dose for coconut tree",
        "how to get kisan credit card loan"
    ]

    print("\n" + "=" * 70)
    print("DEMO RETRIEVAL FOR SAMPLE FARMER QUERIES")
    print("=" * 70)
    for q in demo_queries:
        print(f"\nFarmer Spoken Query: '{q}'")
        matches = retriever.search(q, top_k=3)
        for rank, m in enumerate(matches, 1):
            print(f"  Rank {rank} [Score: {m['similarity']:.3f} | Cat: {m['category']}]:")
            print(f"    Matched Query : {m['query']}")
            print(f"    Expert Answer : {m['answer'][:100]}...")
    print("=" * 70)

if __name__ == "__main__":
    main()
