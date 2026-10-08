"""
pipeline_eval.py - Step 4: Full Pipeline Evaluation (ASR -> Classifier)
Evaluates end-to-end impact of speech recognition on farmer query classification.
Transcribes user recordings, classifies transcript vs reference, and detects
ASR-induced misclassifications and frequently corrupted farming terms.
"""

import os
import re
import joblib
import pandas as pd
import whisper
from collections import Counter

# Ensure ffmpeg binary is found by Whisper subprocess on Windows
try:
    import imageio_ffmpeg
    ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.path.abspath(".") + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

REC_DIR = "my_recordings"
REC_CSV = os.path.join(REC_DIR, "my_recordings.csv")
MODEL_PATH = os.path.join("models", "classifier_bundle.joblib")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

AGRICULTURAL_TERMS = {
    "sugarcane", "borer", "urea", "dap", "ssp", "mop", "rot", "blight", "wilt",
    "infestation", "weedicide", "herbicide", "irrigation", "pesticide", "fungicide",
    "aphid", "chlorpyrifos", "malathion", "bavistin", "dithane", "bordeaux",
    "manure", "compost", "kisan", "kharif", "rabi", "transplanting", "germination"
}

def clean_text(text: str, stop_words: set) -> str:
    """Preprocess text identically to training pipeline."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

def main():
    print("=" * 70)
    print("STEP 4: END-TO-END PIPELINE EVALUATION (Speech -> Text -> Intent)")
    print("=" * 70)

    if not os.path.exists(REC_DIR) or not os.path.exists(REC_CSV):
        print("\n[STATUS: SKIPPED]")
        print(f"Directory '{REC_DIR}' or '{REC_CSV}' was not found.")
        print("As specified in project instructions, Step 4 is skipped until user recordings are provided.")
        print("\nTo test with your own recordings:")
        print("1. Create folder 'my_recordings/'.")
        print("2. Record WAV audio files of real KCC test queries (from data/kcc_test.csv).")
        print("3. Create 'my_recordings/my_recordings.csv' with columns: filename, reference_text")
        print("4. Re-run 'python pipeline_eval.py'.")
        print("=" * 70)
        return

    # Load recorded queries and classifier bundle
    df = pd.read_csv(REC_CSV)
    print(f"Found {len(df)} user recordings in {REC_CSV}.")
    bundle = joblib.load(MODEL_PATH)
    vectorizer = bundle["vectorizer"]
    classifier = bundle["model"]
    stop_words = bundle["stop_words"]

    models = ["tiny", "base", "small"]
    results = []

    for model_name in models:
        print(f"\n--- Transcribing & Classifying with Whisper '{model_name}' ---")
        whisper_model = whisper.load_model(model_name, device="cpu")
        
        cat_changes = 0
        misrec_terms = Counter()

        for _, row in df.iterrows():
            audio_path = os.path.join(REC_DIR, row["filename"])
            ref_raw = row["reference_text"]
            
            # 1. Transcribe audio
            trans = whisper_model.transcribe(audio_path, language="en", fp16=False)["text"]
            
            # 2. Preprocess reference & hypothesis
            ref_clean = clean_text(ref_raw, stop_words)
            hyp_clean = clean_text(trans, stop_words)

            # 3. Classify reference intent vs speech intent
            ref_pred = classifier.predict(vectorizer.transform([ref_clean]))[0]
            hyp_pred = classifier.predict(vectorizer.transform([hyp_clean]))[0]

            if ref_pred != hyp_pred:
                cat_changes += 1

            # 4. Check agricultural term corruption
            ref_words = set(ref_clean.split())
            hyp_words = set(hyp_clean.split())
            for term in AGRICULTURAL_TERMS:
                if term in ref_words and term not in hyp_words:
                    misrec_terms[term] += 1

        change_rate = cat_changes / len(df)
        print(f"Model: {model_name:6s} | Category Change Rate: {change_rate:.2%} ({cat_changes}/{len(df)})")
        print(f"Top Misrecognized Farming Terms: {misrec_terms.most_common(5)}")

        results.append({
            "model_size": model_name,
            "total_clips": len(df),
            "category_mismatches": cat_changes,
            "category_change_rate": round(change_rate, 4),
            "top_misrecognized_terms": str(dict(misrec_terms.most_common(5)))
        })

    out_csv = os.path.join(RESULTS_DIR, "pipeline_results.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nSaved pipeline evaluation to {out_csv}")
    print("=" * 70)

if __name__ == "__main__":
    main()
