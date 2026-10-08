"""
nlp_eval.py - Step 7: Evaluation Suite for New NLP Components
Evaluates Named Entity Recognition (Precision, Recall, F1), Word Sense
Disambiguation (Accuracy), and demonstrates Shallow Parsing outputs.
"""

import os
import pandas as pd
from typing import Set, Tuple
from ner import extract_entities
from shallow_parser import parse_chunks
from wsd import disambiguate_sentence

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Curated benchmark of 10 diverse agricultural test sentences with ground truth
BENCHMARK_SUITE = [
    {
        "sentence": "My sugarcane crop has red rot and I applied urea yesterday.",
        "true_entities": {("CROP", "sugarcane"), ("DISEASE", "red rot"), ("FERTILIZER", "urea"), ("TIME", "yesterday")},
        "true_wsd": {}
    },
    {
        "sentence": "I have 2 acres of sugarcane and the soil is very dry.",
        "true_entities": {("QUANTITY", "2"), ("UNIT", "acres"), ("CROP", "sugarcane"), ("CONDITION", "soil is very dry")},
        "true_wsd": {}
    },
    {
        "sentence": "Plant 50 kg of paddy seeds in the nursery field tomorrow.",
        "true_entities": {("QUANTITY", "50"), ("UNIT", "kg"), ("CROP", "paddy"), ("LOCATION", "nursery"), ("LOCATION", "field"), ("TIME", "tomorrow")},
        "true_wsd": {"plant": "planting / agricultural action", "field": "farm plot / agricultural land"}
    },
    {
        "sentence": "The plant is affected by wilt in maharashtra.",
        "true_entities": {("DISEASE", "wilt"), ("LOCATION", "maharashtra")},
        "true_wsd": {"plant": "crop / botanical organism"}
    },
    {
        "sentence": "Spray 2 litres of chlorpyrifos to kill stem borer.",
        "true_entities": {("QUANTITY", "2"), ("UNIT", "litres"), ("CHEMICAL", "chlorpyrifos"), ("PEST", "stem borer")},
        "true_wsd": {"spray": "chemical application action"}
    },
    {
        "sentence": "He bought 10 bags of dap fertilizer for the wheat farm.",
        "true_entities": {("QUANTITY", "10"), ("UNIT", "bags"), ("FERTILIZER", "dap"), ("CROP", "wheat"), ("LOCATION", "farm")},
        "true_wsd": {}
    },
    {
        "sentence": "Sugarcane yield was 75 tons in summer season.",
        "true_entities": {("CROP", "sugarcane"), ("QUANTITY", "75"), ("UNIT", "tons"), ("TIME", "summer")},
        "true_wsd": {"yield": "harvest output / production quantity"}
    },
    {
        "sentence": "Cane will rot in standing water with moisture stress.",
        "true_entities": {("CROP", "cane"), ("CONDITION", "moisture stress")},
        "true_wsd": {"rot": "decomposition / decay action"}
    },
    {
        "sentence": "The crop shows leaf curl disease and severe wilting.",
        "true_entities": {("DISEASE", "leaf curl"), ("CONDITION", "wilting")},
        "true_wsd": {}
    },
    {
        "sentence": "Drip irrigation was installed across 5 hectares in punjab.",
        "true_entities": {("IRRIGATION", "drip irrigation"), ("QUANTITY", "5"), ("UNIT", "hectares"), ("LOCATION", "punjab")},
        "true_wsd": {}
    }
]

def evaluate_ner():
    """Calculate Precision, Recall, and F1 score for agricultural NER."""
    print("=" * 70)
    print("1. NAMED ENTITY RECOGNITION (NER) EVALUATION")
    print("=" * 70)
    total_pred = 0
    total_true = 0
    correct_matches = 0

    for sample in BENCHMARK_SUITE:
        extracted = extract_entities(sample["sentence"])
        pred_set = {(e["label"], e["text"].lower()) for e in extracted}
        true_set = {(l, t.lower()) for l, t in sample["true_entities"]}

        total_pred += len(pred_set)
        total_true += len(true_set)
        correct_matches += len(pred_set & true_set)

    precision = correct_matches / total_pred if total_pred > 0 else 0.0
    recall = correct_matches / total_true if total_true > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Total Ground Truth Entities : {total_true}")
    print(f"Total Extracted Entities    : {total_pred}")
    print(f"Correct True Positives      : {correct_matches}")
    print(f"NER Precision               : {precision:.4f} ({precision*100:.2f}%)")
    print(f"NER Recall                  : {recall:.4f} ({recall*100:.2f}%)")
    print(f"NER F1-Score                : {f1:.4f} ({f1*100:.2f}%)")
    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_wsd():
    """Calculate accuracy of Word Sense Disambiguation on target words."""
    print("\n" + "=" * 70)
    print("2. WORD SENSE DISAMBIGUATION (WSD) EVALUATION")
    print("=" * 70)
    total_ambiguous = 0
    correct_senses = 0

    for sample in BENCHMARK_SUITE:
        if sample["true_wsd"]:
            wsd_results = disambiguate_sentence(sample["sentence"])
            pred_map = {r["word"].lower(): r["sense_label"] for r in wsd_results}

            for target_word, expected_substr in sample["true_wsd"].items():
                total_ambiguous += 1
                pred_sense = pred_map.get(target_word, "")
                if expected_substr in pred_sense:
                    correct_senses += 1
                    status = "CORRECT"
                else:
                    status = "INCORRECT"
                print(f"Word: '{target_word:6s}' | Expected: '{expected_substr[:30]:30s}' | Got: '{pred_sense[:30]:30s}' | [{status}]")

    accuracy = correct_senses / total_ambiguous if total_ambiguous > 0 else 0.0
    print(f"\nWSD Evaluated Cases: {total_ambiguous} | Correct: {correct_senses}")
    print(f"WSD Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return {"accuracy": accuracy}

def demonstrate_shallow_parsing():
    """Display chunking outputs for all benchmark sentences."""
    print("\n" + "=" * 70)
    print("3. SHALLOW PARSING / CHUNKING REPRESENTATIVE OUTPUTS")
    print("=" * 70)
    for idx, sample in enumerate(BENCHMARK_SUITE, 1):
        parsed = parse_chunks(sample["sentence"])
        print(f"[{idx}] Sentence: {sample['sentence']}")
        print(f"    Chunks  : {parsed}\n")

def main():
    ner_metrics = evaluate_ner()
    wsd_metrics = evaluate_wsd()
    demonstrate_shallow_parsing()

    # Save summary metrics
    summary_df = pd.DataFrame([
        {"Component": "Agricultural NER", "Metric": "F1-Score", "Score": round(ner_metrics["f1"], 4)},
        {"Component": "Agricultural NER", "Metric": "Precision", "Score": round(ner_metrics["precision"], 4)},
        {"Component": "Agricultural NER", "Metric": "Recall", "Score": round(ner_metrics["recall"], 4)},
        {"Component": "Word Sense Disambiguation", "Metric": "Accuracy", "Score": round(wsd_metrics["accuracy"], 4)}
    ])
    out_path = os.path.join(RESULTS_DIR, "nlp_components_results.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"Saved evaluation metrics to {out_path}")

if __name__ == "__main__":
    main()
