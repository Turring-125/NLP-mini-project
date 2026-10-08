# 🌾 Voice-Enabled Farmer Query Assistant
### Advisory & Chatbot Layer for an AI Sugarcane Irrigation & Agronomy Decision System

This project implements an end-to-end, voice-enabled query assistant designed for agricultural advisory, specifically tailored as the conversational layer of an AI-driven irrigation advisory system for sugarcane farmers. Built using 100% real-world data and open-source models running locally on CPU.

---

## 📋 System Architecture

```text
Spoken Query (WAV/MP3)
         │
         ▼
[ Whisper ASR Engine ]  ──► Normalized Transcribed Text
(tiny / base / small)                │
                                     ▼
                      [ Text Preprocessing & Cleaning ]
                      (Lowercasing, Punctuation, Stopwords)
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
[ Intent Classifier ]                                   [ TF-IDF Knowledge Retriever ]
(Logistic Regression)                                   (Cosine Similarity Search)
         │                                                       │
         ▼                                                       ▼
Predicted Category + Confidence                         Top-3 Historical KCC Expert Advisories
         │                                                       │
         └───────────────────────────┬───────────────────────────┘
                                     ▼
                    [ Low-Confidence Safety Guardrail ]
             (Prompts Agronomist Review if confidence < 45% or sim < 0.30)
                                     │
                                     ▼
                         Streamlit User Interface
```

---

## 📂 Project Structure

```text
├── explore.py              # Step 0: Data inspection and domain category mapping
├── asr_eval.py             # Step 1: Whisper ASR benchmark on Google FLEURS (WER, CER, Latency)
├── train.py                # Step 2: Morphology (stemming/lemmatization) & classifier training
├── retrieve.py             # Step 3: TF-IDF vector retrieval engine & Precision@1 evaluation
├── pipeline_eval.py        # Step 4: End-to-end pipeline evaluation on user recordings
├── app.py                  # Step 5: Interactive Streamlit web interface
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
    └── confusion_matrix.png      # 9-class confusion matrix plot
```

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
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 🚀 Execution Guide

Run each step sequentially:

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

# Step 5: Launch Streamlit web app
streamlit run app.py
```

---

## 📊 Experimental Results

### Step 1: Automatic Speech Recognition (Google FLEURS `en_us` 50 Clips)

| Model Size | WER (%) | CER (%) | Latency (s / clip on CPU) | Recommended Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **Whisper tiny** | 15.21% | 6.74% | 0.45s | Ultra-low latency edge devices |
| **Whisper base** | 10.52% | 4.97% | 0.79s | **Optimal real-time balance for laptops** |
| **Whisper small**| 6.76%  | 2.81% | 1.93s | High-accuracy batch processing |

#### Qualitative ASR Error Patterns:
1. **Phonetic substitutions / homophones**: *'sintra'* recognized as *'sinatra'*; *'tracking'* recognized as *'trafficking'*; *'pools'* as *'poles'*.
2. **Number formats**: Spoken textual numbers transcribed as digits (*'twentieth century'* vs *'20th century'*).
3. **Proper nouns**: Non-standard geographical entities and technical vocabulary incur substitution errors.

---

### Step 2: NLP Morphology & Classification

#### Morphological Comparison: Stemming vs Lemmatization

| Term | Porter Stemmer | WordNet Lemmatizer | Linguistic Distinction |
| :--- | :--- | :--- | :--- |
| **infestation** | `infest` | `infestation` | Stemmer chops derivational suffix `-ation` |
| **varieties** | `varieti` | `variety` | Lemmatizer maps plural `-ies` to valid lexical root |
| **dropping** | `drop` | `dropping` | Stemmer strips inflectional `-ing` |
| **fertilizers** | `fertil` | `fertilizer` | Lemmatizer handles plural noun inflection |
| **spraying** | `spray` | `spraying` | Stemmer reduces verb to base root |
| **weedicides** | `weedicid` | `weedicides` | Stemmer truncates trailing `-e` heuristically |
| **borers** | `borer` | `borer` | Lemmatizer maps plural `-s` to noun singular |
| **cultivation** | `cultiv` | `cultivation` | Stemmer removes suffix `-ation` |
| **germinating** | `germin` | `germinating` | Stemmer trims `-ating` |
| **diseases** | `diseas` | `disease` | Lemmatizer restores canonical vocabulary entry |

#### Text Classification Performance (Stratified 80/20 Split on 90,886 KCC Queries)

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
1. Translates spoken farmer concerns (e.g. soil crack appearance, leaf rolling, yellowing, drip dripper clogging) into structured agricultural intent.
2. Cross-references live IoT soil tension / weather sensors before generating definitive irrigation schedule advisories.
3. Automatically triggers an agronomist review warning whenever retrieval similarity or classification confidence drops below safety thresholds.

---

## ⚠️ Limitations & Future Directions

1. **Acoustic and Regional Dialects**: While Whisper handles standard English well, regional rural accents and code-switching (Hinglish/rural colloquial terms) require domain-adapted acoustic fine-tuning.
2. **Out-of-Vocabulary Agricultural Chemicals**: Novel chemical trade names and regional bio-fertilizer formulations may not be present in static TF-IDF vocabulary.
3. **Multi-turn Dialogue Context**: Current implementation operates in single-turn intent-and-retrieve mode; extending to conversational state tracking (DST) will enable follow-up questions.
4. **Sensor & Actuator Integration**: Linking retrieved recommendations directly to field soil moisture probes (capacitive/tensiometers) and solenoid drip valves for closed-loop autonomous irrigation.
