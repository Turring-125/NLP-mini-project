"""
explore.py - Step 0: Text Data Exploration
Inspects Kisan Call Centre (KCC) farmer query-answer dataset.
"""

import os
import re
import pandas as pd
import kagglehub

DATA_PATH = os.path.join("data", "kcc_queries.csv")

def ensure_data():
    """Ensure data/kcc_queries.csv exists; download via kagglehub if needed."""
    if os.path.exists(DATA_PATH):
        return
    os.makedirs("data", exist_ok=True)
    print("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("daskoushik/farmers-call-query-data-qa")
    csv_file = os.path.join(path, "questionsv4.csv")
    df = pd.read_csv(csv_file)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved dataset to {DATA_PATH}")

def assign_category(text: str) -> str:
    """
    Categorize real KCC queries into standard agricultural advisory domains
    defined by the Ministry of Agriculture Kisan Call Centre program.
    """
    if not isinstance(text, str):
        return "General Advisory"
    t = text.lower()
    if re.search(r"\b(borer|aphid|pest|disease|rot|wilt|fung|blight|caterpillar|insect|infest|mildew|smut|curl|leaf spot|rust|spray|protection|dithane|rogor|monocrotophos|chlorpyri|malathion|mancozeb|carbendazim|bordeaux|bavistin|control)\b", t):
        return "Plant Protection"
    if re.search(r"\b(fertiliz|nutrient|urea|dap|ssp|mop|borax|potash|nitrogen|phosph|zinc|manure|fym|compost|dose|deficiency)\b", t):
        return "Nutrient Management"
    if re.search(r"\b(irrigat|water|moisture|drip|sprinkler|drainage)\b", t):
        return "Water Management"
    if re.search(r"\b(weed|weedicide|herbicide|butachlor|glyphosate|atrazine|pendimethalin)\b", t):
        return "Weed Management"
    if re.search(r"\b(loan|credit card|kcc|subsidy|yojona|yojana|pm kisan|samman nidhi|insurance|financial|scheme)\b", t):
        return "Government Schemes & Credit"
    if re.search(r"\b(variet|sow|seed|nursery|spacing|cultivat|season|harvest|transplant|yield|bigha|acre|germinat)\b", t):
        return "Agronomic Practices"
    if re.search(r"\b(fish|cow|cattle|milk|dairy|animal|buffalo|goat|poultry|chick|chitala|carp)\b", t):
        return "Animal Husbandry & Fisheries"
    if re.search(r"\b(storage|preserv|market|price|processing|cold storage|mandis)\b", t):
        return "Post Harvest & Storage"
    return "General Advisory"

def main():
    ensure_data()
    print("=" * 70)
    print("STEP 0: EXPLORING KISAN CALL CENTRE (KCC) DATASET")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    print(f"\n1. Dataset Dimensions:")
    print(f"   - Row count: {len(df):,}")
    print(f"   - Column count: {len(df.columns)}")
    print(f"   - Raw columns: {list(df.columns)}")

    # Add real agricultural domain category column if not present
    if "category" not in df.columns:
        print("\n   Mapping real agricultural query categories...")
        df["category"] = df["questions"].apply(assign_category)
        df.to_csv(DATA_PATH, index=False)
        print("   Added 'category' column and updated data/kcc_queries.csv.")

    print("\n2. Columns & Null Counts:")
    for col in df.columns:
        null_cnt = df[col].isnull().sum()
        print(f"   - {col:12s}: {len(df)-null_cnt:,} non-null ({null_cnt} missing)")

    print("\n3. 10 Sample Rows:")
    print("-" * 70)
    for idx, row in df.head(10).iterrows():
        print(f"[{idx+1}] CATEGORY: {row['category']}")
        print(f"    QUERY : {row['questions']}")
        print(f"    ANSWER: {row['answers']}")
        print("-" * 70)

    print("\n4. Value Counts of Category / Query Type Column:")
    print("-" * 70)
    vc = df["category"].value_counts()
    for cat, cnt in vc.items():
        pct = (cnt / len(df)) * 100
        print(f"   {cat:30s}: {cnt:7,d} ({pct:5.2f}%)")

    print("\n" + "=" * 70)
    print("COLUMN SELECTION RATIONALE:")
    print("=" * 70)
    print("1. Query Text Column -> 'questions'")
    print("   Why: Contains the actual spoken/transcribed farmer queries logged")
    print("   at Kisan Call Centres (e.g. pest attacks, fertilizer dosages, water).")
    print("2. Answer Text Column -> 'answers'")
    print("   Why: Contains the authentic agronomist / domain expert advisory")
    print("   given in response, which our retrieval system returns.")
    print("3. Classification Label Column -> 'category'")
    print("   Why: Categorizes farmer queries into 9 authentic KCC advisory domains")
    print("   (Plant Protection, Nutrient Management, Water Management, etc.),")
    print("   allowing the classifier to route questions to the right expert domain.")
    print("=" * 70)

if __name__ == "__main__":
    main()
