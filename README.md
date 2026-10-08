# 🌾 Voice-Enabled Farmer Query Assistant
### Advisory & Chatbot Layer for an AI Sugarcane Irrigation & Agronomy Decision System

This project implements an end-to-end, voice-enabled query assistant designed for agricultural advisory, specifically tailored as the conversational decision-support layer of an AI-driven irrigation advisory system for sugarcane farmers. Built using 100% real-world data and open-source models running locally on CPU.

---

## 📋 Extended NLP Pipeline Architecture

```text
Farmer Voice (Spoken Audio Clip)
         │
         ▼
[ Audio Processing & Ingestion ] ──► Dual Path: direct soundfile PCM or imageio-ffmpeg
         │
         ▼
[ Whisper ASR Engine ]           ──► Raw Transcribed Query Text (tiny / base / small)
         │
         ▼
[ Text Normalization ]           ──► Lowercased, stripped punctuation, filtered stopwords
         │
         ▼
[ Morphological Processing ]     ──► Porter Stemming & WordNet Lemmatization (Module 2.1)
         │
         ▼
[ Named Entity Recognition ]     ──► CROP, DISEASE, PEST, FERTILIZER, QUANTITY, etc. (Module 2.3)
         │
         ▼
[ Shallow Parsing / Chunking ]   ──► [NP ...], [VP ...], [PP ...] Syntactic Chunks (Module 2.3 / 3)
         │
         ▼
[ Word Sense Disambiguation ]    ──► Contextual disambiguation of 'plant', 'rot', etc. (Module 4.3)
         │
         ├───────────────────────────────────────────────────────┐
         ▼                                                       ▼
[ Intent Classifier ]                                   [ TF-IDF Knowledge Retriever ]
(Multinomial Logistic Regression)                       (Cosine Similarity Search)
         │                                                       │
         ▼                                                       ▼
Predicted Category + Confidence %                       Top-3 Historical KCC Expert Advisories
         │                                                       │
         └───────────────────────────┬───────────────────────────┘
                                     ▼
                    [ Low-Confidence Safety Guardrail ]
             (Prompts Agronomist Review if confidence < 45% or sim < 0.30)
                                     │
                                     ▼
                     Streamlit User Interface (app.py)
```

---

## 🎓 NLP Course Syllabus Mapping

This project maps directly to core Natural Language Processing syllabus modules:

| Syllabus Module | Topic Area | Project Component & Implementation File |
| :--- | :--- | :--- |
| **Module 2.1** | **Morphology** | Stemming vs Lemmatization comparison ([train.py](file:///d:/clg/LY/NLP/mini%20project/train.py)) |
| **Module 2.3** | **Named Entities** | Agricultural Domain NER ([ner.py](file:///d:/clg/LY/NLP/mini%20project/ner.py)) |
| **Module 2.3 / 3** | **Structures & Parsing** | Shallow Parsing & Chunking ([shallow_parser.py](file:///d:/clg/LY/NLP/mini%20project/shallow_parser.py)) |
| **Module 4.1** | **Lexical Semantics & WordNet** | Princeton WordNet synset hierarchy & lemmas ([train.py](file:///d:/clg/LY/NLP/mini%20project/train.py), [wsd.py](file:///d:/clg/LY/NLP/mini%20project/wsd.py)) |
| **Module 4.3** | **Word Sense Disambiguation** | Contextual Lesk & syntactic mood disambiguation ([wsd.py](file:///d:/clg/LY/NLP/mini%20project/wsd.py)) |
| **Module 5.3** | **Sequence-to-Sequence Models**| OpenAI Whisper Transformer encoder-decoder ([asr_eval.py](file:///d:/clg/LY/NLP/mini%20project/asr_eval.py), [app.py](file:///d:/clg/LY/NLP/mini%20project/app.py)) |

---

## 📂 Project Structure

```text
├── explore.py              # Step 0: Data inspection and domain category mapping
├── asr_eval.py             # Step 1: Whisper ASR benchmark on Google FLEURS (WER, CER, Latency)
├── train.py                # Step 2: Morphology (stemming/lemmatization) & classifier training
├── retrieve.py             # Step 3: TF-IDF vector retrieval engine & Precision@1 evaluation
├── pipeline_eval.py        # Step 4: End-to-end pipeline evaluation on user recordings
├── app.py                  # Step 5: Interactive Streamlit web app with NLP Analysis
├── ner.py                  # Module 2.3: Agricultural Named Entity Recognition
├── shallow_parser.py       # Module 2.3/3: Part-of-Speech tagging & Regexp Chunk Parser
├── wsd.py                  # Module 4.3: Word Sense Disambiguation for agricultural terms
├── nlp_eval.py             # Step 7: Evaluation suite for NER, WSD, and Chunking
├── requirements.txt        # Python dependency manifest (with FFmpeg note)
├── data/
│   ├── kcc_queries.csv     # Full raw Kisan Call Centre Q&A dataset (178,939 records)
│   ├── kcc_train.csv       # Training split (72,708 queries)
│   ├── kcc_test.csv        # Stratified test split (18,178 queries)
│   └── fleurs_50.joblib    # 50 Google FLEURS en_us audio benchmark test clips
├── models/
│   ├── classifier_bundle.joblib  # Trained TF-IDF vectorizer + Logistic Regression model
│   └── retrieval_bundle.joblib   # TF-IDF sparse index over training knowledge base
└── results/
    ├── asr_results.csv           # Whisper WER, CER, and CPU latency metrics
    ├── asr_wer_cer.png           # ASR performance bar chart
    ├── asr_errors.txt            # 15 qualitative speech transcription error examples
    ├── classifier_results.csv    # Naive Bayes vs Logistic Regression comparison
    ├── confusion_matrix.png      # 9-class confusion matrix plot
    ├── retrieval_results.csv     # Precision@1 retrieval metrics
    └── nlp_components_results.csv# NER (P/R/F1) and WSD accuracy evaluation metrics
```

---

## 🧩 Deep Dive: Extended NLP Components

### 1. Named Entity Recognition (`ner.py`)
- **Purpose**: Identifies agricultural entities in conversational farmer queries.
- **Entity Labels**:
  - `CROP`: *sugarcane, paddy, wheat, cotton, maize, coconut, etc.*
  - `DISEASE`: *red rot, smut, rust, blast, blight, wilt, leaf curl, etc.*
  - `PEST`: *stem borer, top borer, pyrilla, aphid, whitefly, thrips, etc.*
  - `FERTILIZER`: *urea, dap, mop, ssp, npk, zinc sulphate, compost, etc.*
  - `CHEMICAL`: *chlorpyrifos, carbendazim, bavistin, atrazine, malathion, etc.*
  - `QUANTITY` & `UNIT`: *2 acres, 50 kg, 5 litres, 10 bags, 75 tons, etc.*
  - `CONDITION`: *soil is very dry, moisture stress, yellowing leaves, cracked soil, wilting, etc.*
  - `TIME`: *yesterday, tomorrow, summer, june, kharif, etc.*
  - `LOCATION`: *maharashtra, punjab, field, nursery, farm, etc.*
  - `IRRIGATION`: *drip irrigation, flood irrigation, furrow, drip lateral, etc.*
- **Methodology**: Multi-token greedy gazetteer matching + specialized regex patterns for quantities, temporal expressions, and environmental conditions.

### 2. Shallow Parsing & Chunking (`shallow_parser.py`)
- **Purpose**: Demonstrates syntactic phrase structure without the overhead of full dependency trees.
- **Chunk Grammar**:
  ```python
  CHUNK_GRAMMAR = r"""
    NP: {<DT|PRP\$|POS>?<JJ.*|CD>*<NN.*>+}   # Noun Phrase
    VP: {<MD>?<VB.*>+(<RB.*>)?}              # Verb Phrase
    PP: {<IN>+<NP>}                          # Prepositional Phrase
  """
  ```
- **Example**:
  - Input: `"my sugarcane leaves are curling because of water stress"`
  - Output: `[NP my sugarcane leaves] [VP are curling] [PP because of water stress]`

### 3. Word Sense Disambiguation (`wsd.py`)
- **Purpose**: Resolves the exact semantic meaning of polysemous agricultural words based on contextual syntactic clues and WordNet synsets:
  - `"Plant the sugarcane seedlings tomorrow."` $\to$ **`plant`**: *planting / agricultural action (sowing seeds into ground)* (`plant.v.01`).
  - `"The plant is affected by red rot."` $\to$ **`plant`**: *crop / botanical organism (living plant)* (`plant.n.02`).
  - `"Working in the sugarcane field."` $\to$ **`field`**: *farm plot / agricultural land* (`field.n.01`).
  - `"Expert in the field of agronomy."` $\to$ **`field`**: *domain / discipline of study* (`field.n.04`).
  - `"The fallen cane will rot in water."` $\to$ **`rot`**: *decomposition / decay action* (`rot.v.01`).
  - `"The stalks have severe red rot."` $\to$ **`rot`**: *plant disease / fungal decay symptom* (`rot.n.01`).

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- **Python**: 3.10+ (tested on Python 3.12)
- **FFmpeg**: Required on the system PATH for Whisper audio loading.
  - Windows: `winget install Gyan.FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org).
  - Ubuntu/Debian: `sudo apt install ffmpeg`

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Required NLTK Corpora
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

---

## 🚀 Execution Guide

```bash
# Step 0: Inspect KCC dataset and category distribution
python explore.py

# Step 1: Evaluate Whisper (tiny, base, small) on Google FLEURS
python asr_eval.py

# Step 2: Run morphology analysis, train classifier, and save artifacts
python train.py

# Step 3: Index knowledge base and evaluate Precision@1
python retrieve.py

# Step 4: Pipeline evaluation on user recordings (skips gracefully if my_recordings/ is missing)
python pipeline_eval.py

# Step 7: Evaluate new NLP components (NER, WSD, Shallow Parsing)
python nlp_eval.py

# Step 5: Launch Streamlit web app
streamlit run app.py
```

---

## 📊 Experimental Results

### NLP Course Components Evaluation ([nlp_eval.py](file:///d:/clg/LY/NLP/mini%20project/nlp_eval.py))

| Component | Metric | Score | Evaluation Methodology |
| :--- | :--- | :---: | :--- |
| **Agricultural NER** | **Precision** | **97.37%** | Strict span-and-label exact match on benchmark suite |
| **Agricultural NER** | **Recall** | **100.00%**| Comprehensive multi-token gazetteer coverage |
| **Agricultural NER** | **F1-Score** | **98.67%** | Harmonic mean of precision and recall |
| **Word Sense Disambiguation** | **Accuracy** | **100.00%**| Contextual sense resolution on ambiguous target words |
| **Shallow Parsing** | **Coverage** | **100.00%**| Correct extraction of NP, VP, and PP structures |

---

### Step 1: Automatic Speech Recognition (Google FLEURS `en_us` 50 Clips)

| Model Size | WER (%) | CER (%) | Latency (s / clip on CPU) | Recommended Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **Whisper tiny** | 15.21% | 6.74% | 0.45s | Ultra-low latency edge devices |
| **Whisper base** | 10.52% | 4.97% | 0.79s | **Optimal real-time balance for laptops** |
| **Whisper small**| 6.76%  | 2.81% | 1.93s | High-accuracy batch processing |

---

### Step 2: Text Classification (90,886 KCC Queries)

| Model Architecture | Accuracy | Macro-F1 | Notes |
| :--- | :---: | :---: | :--- |
| **Multinomial Naive Bayes (Baseline)** | 92.32% | 0.7876 | Fast baseline, lower recall on minority classes |
| **Logistic Regression (Best Model)** | **99.15%** | **0.9740** | Robust linear decision boundary on n-gram TF-IDF |

---

### Step 3: Retrieval Engine Evaluation

- **Knowledge Base Size**: 72,708 historical farmer queries with agronomist answers.
- **Test Set Size**: 18,178 unseen test queries.
- **Top-1 Categorical Precision (`Precision@1`)**: **90.20%**
- **Average Query Retrieval Latency**: < 1.5 ms per query on CPU.

---

## 🌾 Context: AI Sugarcane Irrigation Advisory System

In commercial sugarcane cultivation, water management directly dictates stalk elongation, internode formation, sucrose content, and cane yield:
- **Formative Stage (60–130 days)**: Critical period where water deficit reduces cane tonnage drastically.
- **Maturity / Ripening Stage**: Moderate moisture withholding enhances sucrose concentration.

The Farmer Query Assistant serves as the human-interaction frontend:
1. Translates spoken farmer concerns (e.g. soil crack appearance, leaf rolling, yellowing, drip dripper clogging) into structured agricultural intent and recognized entities.
2. Cross-references live IoT soil tension / weather sensors before generating definitive irrigation schedule advisories.
3. Automatically triggers an agronomist review warning whenever retrieval similarity or classification confidence drops below safety thresholds.
