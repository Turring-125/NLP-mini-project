# Technical Implementation Specification & Architecture Reference
## Project: Voice-Enabled Farmer Query Assistant (Advisory Layer for AI Sugarcane Irrigation)

---

### Table of Contents
1. [Executive Summary & Domain Objectives](#1-executive-summary--domain-objectives)
2. [End-to-End System Architecture & Dataflow](#2-end-to-end-system-architecture--dataflow)
3. [NLP Course Syllabus Mapping](#3-nlp-course-syllabus-mapping)
4. [Dataset Profiles & Preprocessing Pipeline](#4-dataset-profiles--preprocessing-pipeline)
5. [Mathematical Formulations & Model Mechanics](#5-mathematical-formulations--model-mechanics)
   - 5.1 [Automatic Speech Recognition (ASR): OpenAI Whisper (Module 5.3)](#51-automatic-speech-recognition-asr-openai-whisper-module-53)
   - 5.2 [Linguistic Morphology: Stemming vs Lemmatization (Module 2.1)](#52-linguistic-morphology-stemming-vs-lemmatization-module-21)
   - 5.3 [Agricultural Named Entity Recognition (Module 2.3)](#53-agricultural-named-entity-recognition-module-23)
   - 5.4 [Shallow Parsing & Chunking Grammar (Module 2.3 / 3)](#54-shallow-parsing--chunking-grammar-module-23--3)
   - 5.5 [Word Sense Disambiguation: Lesk & Syntax (Module 4.3)](#55-word-sense-disambiguation-lesk--syntax-module-43)
   - 5.6 [Feature Representation: TF-IDF Vector Space](#56-feature-representation-tf-idf-vector-space)
   - 5.7 [Intent Classification: Multinomial Naive Bayes & Logistic Regression](#57-intent-classification-multinomial-naive-bayes--logistic-regression)
   - 5.8 [Information Retrieval Engine: Vector Space Cosine Similarity](#58-information-retrieval-engine-vector-space-cosine-similarity)
6. [Empirical Experimental Results & Benchmark Tables](#6-empirical-experimental-results--benchmark-tables)
7. [Qualitative Linguistic & ASR Error Analysis](#7-qualitative-linguistic--asr-error-analysis)
8. [Codebase Anatomy & Component Walkthrough](#8-codebase-anatomy--component-walkthrough)
9. [Audio Ingestion Pipeline & OS Compatibility Hardening](#9-audio-ingestion-pipeline--os-compatibility-hardening)
10. [Integration with the Sugarcane Irrigation Decision System](#10-integration-with-the-sugarcane-irrigation-decision-system)
11. [Known Limitations & Future Technical Trajectory](#11-known-limitations--future-technical-trajectory)

---

### 1. Executive Summary & Domain Objectives

This project builds a real-time, voice-enabled agricultural advisory assistant that converts spoken farmer queries into structured agricultural domains and retrieves expert agronomist recommendations from authentic Kisan Call Centre (KCC) records.

The primary application domain is serving as the conversational decision-support layer for an **AI-driven Sugarcane Irrigation Scheduling System**. In sugarcane (*Saccharum officinarum*), water management during the **formative stage (60–130 days post-planting)** directly dictates shoot elongation, tillering capacity, internode elongation, and eventual sucrose accumulation. Farmer queries during this phase frequently describe symptoms of water stress (leaf rolling, leaf scorch, soil cracking), nutrient deficiencies exacerbated by moisture deficits, or disease outbreaks (such as red rot, *Colletotrichum falcatum*). 

The assistant provides an instant, accurate voice-to-advisory pipeline that:
1. Ingests raw voice input (WAV/MP3/M4A/OGG) or text queries.
2. Transcribes acoustic speech using OpenAI Whisper models benchmarked on CPU.
3. Preprocesses and normalizes agricultural text using NLTK morphological analysis (Stemming vs Lemmatization).
4. Extracts domain-specific entities (crops, diseases, pests, fertilizers, chemicals, quantities, units, conditions, time, locations) via a hybrid Named Entity Recognition module.
5. Performs shallow parsing to extract Noun Phrases (NP), Verb Phrases (VP), and Prepositional Phrases (PP).
6. Disambiguates polysemous agricultural terms (e.g., `plant`, `field`, `rot`, `yield`, `spray`) using syntactic mood and WordNet Lesk overlap.
7. Classifies query intent into 9 specialized agricultural domains using a calibrated Logistic Regression model (achieving **99.15% accuracy** and **0.9740 Macro-F1**).
8. Queries a 72,708-document TF-IDF vector index to retrieve the top-3 historically validated expert recommendations (achieving **90.20% Precision@1** on unseen test queries).
9. Implements safety guardrails: warns the farmer (`"Low confidence - please consult an agronomist"`) when classifier probability $< 0.45$ or retrieval similarity $< 0.30$.

---

### 2. End-to-End System Architecture & Dataflow

```
Farmer Spoken Audio (WAV / MP3 / M4A)
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ Audio Decoupling & Ingestion Pipeline                  │
│  - Method 1 (Primary): Direct soundfile PCM decoding   │
│    -> 16 kHz Mono float32 NumPy array                  │
│  - Method 2 (Fallback): imageio-ffmpeg subprocess      │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ OpenAI Whisper ASR (tiny / base / small)               │
│  - Log-Mel Spectrogram extraction (80 channels)        │
│  - 2x 1D Convolution downsampling                      │
│  - Transformer Encoder-Decoder (Greedy decoding)       │
└────────────────────────────────────────────────────────┘
                 │
                 ▼ Raw Transcript Text
┌────────────────────────────────────────────────────────┐
│ NLP Preprocessing & Normalization                      │
│  - Lowercasing                                         │
│  - Regex punctuation & symbol removal: [^\w\s]         │
│  - NLTK English stopword filtering                     │
│  - Morphological normalization (Porter vs WordNet)     │
└────────────────────────────────────────────────────────┘
                 │
                 ▼ Cleaned Query String
┌────────────────────────────────────────────────────────┐
│ Named Entity Recognition (NER - Module 2.3)            │
│  - CROP, DISEASE, PEST, FERTILIZER, CHEMICAL           │
│  - QUANTITY, UNIT, CONDITION, TIME, LOCATION           │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ Shallow Parsing / Chunking (Module 2.3 / 3)            │
│  - Noun Phrases (NP), Verb Phrases (VP), PPs           │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ Word Sense Disambiguation (WSD - Module 4.3)           │
│  - Disambiguates 'plant', 'field', 'rot', 'yield', etc.│
│  - Syntactic mood + WordNet Lesk overlap               │
└────────────────────────────────────────────────────────┘
                 │
       ┌─────────┴───────────────────────────────────────┐
       │                                                 │
       ▼                                                 ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│ Intent Classification Branch │        │ Knowledge Retrieval Branch   │
│ - TF-IDF (10,000 features,   │        │ - TF-IDF (15,000 features,   │
│   unigrams + bigrams)        │        │   unigrams + bigrams)        │
│ - Multinomial Logistic Reg.  │        │ - Sparse CSR Matrix Dot Prod │
│ - Class probability via      │        │ - Cosine Similarity Ranking  │
│   Softmax function           │        │   over 72,708 Train QA pairs │
└──────────────────────────────┘        └──────────────────────────────┘
       │                                                 │
       ▼                                                 ▼
  Predicted Domain & Conf %                        Top-3 Expert Answers & Scores
       │                                                 │
       └───────────────────────┬─────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│ Confidence & Similarity Safety Guardrail               │
│  IF confidence < 0.45 OR top_similarity < 0.30:       │
│     Display Agronomist Consultation Warning Banner    │
└────────────────────────────────────────────────────────┘
                               │
                               ▼
               Streamlit Web Interface (app.py)
```

---

### 3. NLP Course Syllabus Mapping

| Syllabus Module | Topic Area | Code File | Purpose & Role in Pipeline |
| :--- | :--- | :--- | :--- |
| **Module 2.1** | **Morphology** | [train.py](file:///d:/clg/LY/NLP/mini%20project/train.py) | Porter Stemmer vs WordNet Lemmatizer comparison on 10 farm terms |
| **Module 2.3** | **Named Entities** | [ner.py](file:///d:/clg/LY/NLP/mini%20project/ner.py) | Agricultural NER for crops, chemicals, pests, quantities, and units |
| **Module 2.3 / 3** | **Structures & Parsing** | [shallow_parser.py](file:///d:/clg/LY/NLP/mini%20project/shallow_parser.py) | Part-of-Speech tagging & Regexp Parser for NP, VP, and PP chunking |
| **Module 4.1** | **Lexical Semantics & WordNet** | [train.py](file:///d:/clg/LY/NLP/mini%20project/train.py), [wsd.py](file:///d:/clg/LY/NLP/mini%20project/wsd.py) | WordNet lexical database, lemmas, and synset hierarchies |
| **Module 4.3** | **Word Sense Disambiguation** | [wsd.py](file:///d:/clg/LY/NLP/mini%20project/wsd.py) | Contextual disambiguation of polysemous farm words (`plant`, `rot`, etc.) |
| **Module 5.3** | **Sequence-to-Sequence Models**| [asr_eval.py](file:///d:/clg/LY/NLP/mini%20project/asr_eval.py), [app.py](file:///d:/clg/LY/NLP/mini%20project/app.py) | Whisper Transformer encoder-decoder architecture with 80-channel mel input |

---

### 4. Dataset Profiles & Preprocessing Pipeline

#### 4.1 Kisan Call Centre (KCC) Query-Answer Dataset
- **Origin**: Government of India, Ministry of Agriculture & Farmers Welfare Kisan Call Centre records (via Kaggle Hub).
- **Raw Volume**: 178,939 total rows.
- **Columns**: `questions` (raw farmer query), `answers` (expert agricultural extension officer response), `category` (domain category).
- **Deduplication & Hygiene**:
  - Null query / answer / category rows dropped.
  - Duplicate questions dropped to prevent data leakage between train and test splits.
  - Categories with fewer than 50 total examples dropped (all retained categories exceed 380 records).
  - Cleaned volume: **90,886 unique, verified records**.

#### Category Distribution:
| Category Label | Record Count | Proportion (%) | Typical Farmer Queries |
| :--- | :---: | :---: | :--- |
| **General Advisory** | 37,738 | 41.52% | Weather forecast, market prices, general crop cultivation |
| **Plant Protection** | 31,492 | 34.65% | Red rot, stem borer, pyrilla, leaf spot, pesticide dosage |
| **Nutrient Management** | 7,949 | 8.75% | Urea, DAP, potash application, zinc deficiency, composting |
| **Animal Husbandry & Fisheries** | 5,332 | 5.87% | Cattle feed, milk yield, vaccination, fish pond treatment |
| **Agronomic Practices** | 3,615 | 3.98% | Sowing depth, spacing, ratoon management, seed rate |
| **Government Schemes & Credit** | 3,180 | 3.50% | Kisan Credit Card (KCC), PM-KISAN subsidy, crop insurance |
| **Water Management** | 741 | 0.82% | Irrigation interval, drip lateral clogging, summer watering |
| **Weed Management** | 459 | 0.50% | Atrazine herbicide dosage, broadleaf weed eradication |
| **Post Harvest & Storage** | 380 | 0.42% | Cane jaggery storage, grain moisture, godown protection |

#### Train/Test Splitting Strategy:
- **Partition Ratio**: 80% Train, 20% Test.
- **Methodology**: `train_test_split(..., test_size=0.2, random_state=42, stratify=y)`.
- **Training Records**: 72,708 samples saved to `data/kcc_train.csv`.
- **Testing Records**: 18,178 samples saved to `data/kcc_test.csv`.

---

### 5. Mathematical Formulations & Model Mechanics

#### 5.1 Automatic Speech Recognition (ASR): OpenAI Whisper (Module 5.3)

Whisper processes audio $x(t)$ by converting the raw 16 kHz waveform into an 80-channel log-magnitude Mel spectrogram:
1. Audio divided into 25 ms frames (400 samples) with a 10 ms hop size (160 samples) using a Hanning window.
2. Fast Fourier Transform (FFT) computes the frequency spectrum.
3. Spectrum mapped onto 80 Mel-scale filter banks spaced logarithmically from 0 Hz to 8,000 Hz:
$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$
4. Natural logarithm applied: $S = \log(\text{MelSpectrum} + 10^{-5})$.
5. Transformer Encoder downsamples via two 1D convolutions (stride 2) to 50 Hz representations.
6. Transformer Decoder autoregressively predicts Byte-Pair Encoding tokens with cross-attention.

#### 5.2 Linguistic Morphology: Stemming vs Lemmatization (Module 2.1)
- **Porter Stemmer**: Algorithmic affix stripping using 5 cascading heuristic rule sets.
- **WordNet Lemmatizer**: Morphological reduction using lexical database lookup, preserving syntactic validity.

#### 5.3 Agricultural Named Entity Recognition (Module 2.3)
Implements a hybrid multi-token gazetteer and regular expression system:
$$\text{Entities} = \text{GazetteerMatch}(\mathcal{G}_{\text{Agri}}) \cup \text{RegexMatch}(\mathcal{P}_{\text{Qty/Unit/Time/Cond}})$$
Prioritizes greedy longest-match first to resolve multi-word entities (e.g. `red rot` over `rot`, `drip irrigation` over `irrigation`).

#### 5.4 Shallow Parsing & Chunking Grammar (Module 2.3 / 3)
Constructs chunk trees over Part-of-Speech tags using regular expression grammar:
```python
CHUNK_GRAMMAR = r"""
  NP: {<DT|PRP\$|POS>?<JJ.*|CD>*<NN.*>+}   # Noun Phrase
  VP: {<MD>?<VB.*>+(<RB.*>)?}              # Verb Phrase
  PP: {<IN>+<NP>}                          # Prepositional Phrase
"""
```

#### 5.5 Word Sense Disambiguation: Lesk & Syntax (Module 4.3)
Combines sentence-level syntactic mood detection with WordNet gloss overlap:
- **Imperative / Action Detection**: If word occurs at sentence-initial position without subject pronoun or follows modal verbs/infinitives (`to`, `should`, `please`), it is treated as a transitive verb action (e.g. `Plant 50 kg...` $\to$ `plant.v.01`).
- **Nominal Context Detection**: If preceded by determiners/possessives (`the`, `my`, `this`), it is treated as a botanical noun organism (`The plant is affected...` $\to$ `plant.n.02`).
- **Domain Clue Matching**: Resolves `field` as agricultural land if agricultural lexical context is present, otherwise academic discipline.

#### 5.6 Feature Representation: TF-IDF Vector Space
Given term $t$, document $d$, and corpus $D$:
$$\text{TF-IDF}(t, d, D) = f_{t,d} \times \left[\log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1\right]$$
Normalized using Euclidean $L_2$ norm.

#### 5.7 Intent Classification: Logistic Regression
Multinomial Softmax regression with $L_2$ penalty:
$$P(Y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$

#### 5.8 Information Retrieval Engine
Computes cosine similarity between unit-norm query vector $\mathbf{q}$ and sparse CSR training matrix $\mathbf{A}$:
$$\mathbf{s} = \mathbf{q} \mathbf{A}^T$$
Evaluated using Precision@1 on 18,178 unseen test queries.

---

### 6. Empirical Experimental Results & Benchmark Tables

#### 6.1 Extended NLP Components Evaluation ([nlp_eval.py](file:///d:/clg/LY/NLP/mini%20project/nlp_eval.py))

| Component | Metric | Score | Evaluation Methodology |
| :--- | :--- | :---: | :--- |
| **Agricultural NER** | **Precision** | **97.37%** | Strict span-and-label exact match on benchmark suite |
| **Agricultural NER** | **Recall** | **100.00%**| Comprehensive multi-token gazetteer coverage |
| **Agricultural NER** | **F1-Score** | **98.67%** | Harmonic mean of precision and recall |
| **Word Sense Disambiguation** | **Accuracy** | **100.00%**| Contextual sense resolution on ambiguous target words |
| **Shallow Parsing** | **Coverage** | **100.00%**| Correct extraction of NP, VP, and PP structures |

#### 6.2 Speech Recognition Evaluation (FLEURS 50 Clips)

| Model | WER (%) | CER (%) | Latency / Clip | Total Time (50 clips) | Optimal Environment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Whisper `tiny`** | **15.21%** | **6.74%** | **0.45s** | 22.5s | Raspberry Pi / Low-power Edge |
| **Whisper `base`** | **10.52%** | **4.97%** | **0.79s** | 39.5s | **Standard Laptop / Real-time Production** |
| **Whisper `small`**| **6.76%**  | **2.81%** | **1.93s** | 96.5s | Server-side / Batch Processing |

#### 6.3 Intent Classifier Benchmark (18,178 Unseen Test Queries)

| Architecture | Overall Accuracy | Macro-Average F1 | Weighted-Average F1 | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (Baseline)** | 92.32% | 0.7876 | 0.9184 | 0.42s |
| **Logistic Regression (Main Model)** | **99.15%** | **0.9740** | **0.9914** | 2.56s |

#### 6.4 Retrieval Engine Benchmark
- **Indexed Documents**: 72,708 historical farmer queries + agronomist answers.
- **Evaluation Set**: 18,178 unseen test queries.
- **Categorical Precision@1**: **90.20%**.
- **Inference Speed**: $< 1.5\text{ ms}$ per query across 72,708 documents via CSR matrix vectorization.

---

### 7. Qualitative Linguistic & ASR Error Analysis

Inspection of errors logged in [`results/asr_errors.txt`](file:///d:/clg/LY/NLP/mini%20project/results/asr_errors.txt) reveals three distinct linguistic failure categories:
1. **Phonetic Homophones & Substitutions**: Acoustic confusion leads to common conversational words (`sintra` $\to$ `sinatra`, `tracking` $\to$ `trafficking`, `pools` $\to$ `poles`).
2. **Number Orthography**: Spoken numbers transcribed as digits (`twentieth century` $\to$ `20th century`, `35mm` $\to$ `35 millimeter`).
3. **Proper Nouns**: Low-frequency regional names incur character substitutions (`nyiragongo` $\to$ `nairaiganggo`).

---

### 8. Codebase Anatomy & Component Walkthrough

```
d:/clg/LY/NLP/mini project/
├── explore.py              (106 lines) - Step 0: Data ingestion & category mapping
├── asr_eval.py             (130 lines) - Step 1: FLEURS Whisper ASR benchmark & error logging
├── train.py                (143 lines) - Step 2: Morphology analysis, TF-IDF & classifier training
├── retrieve.py             (130 lines) - Step 3: TF-IDF search engine & Precision@1 evaluation
├── pipeline_eval.py        (123 lines) - Step 4: End-to-end user recording pipeline test
├── app.py                  (183 lines) - Step 5: Streamlit web app with NLP Analysis UI
├── ner.py                  (112 lines) - Module 2.3: Agricultural Named Entity Recognition
├── shallow_parser.py       (81 lines)  - Module 2.3/3: POS tagging & Regexp Chunk Parser
├── wsd.py                  (135 lines) - Module 4.3: Word Sense Disambiguation
├── nlp_eval.py             (156 lines) - Step 7: Evaluation suite for NER, WSD, and Chunking
├── requirements.txt        (14 lines)  - Dependency manifest with FFmpeg notes
└── README.md               (215 lines) - High-level project documentation
```

---

### 9. Audio Ingestion Pipeline & OS Compatibility Hardening

On Windows environments, OpenAI Whisper's native `load_audio()` executes a subprocess call to `ffmpeg`.
To prevent `FileNotFoundError: [WinError 2]`:
1. **Dynamic Binary Registration**: Locates bundled `imageio-ffmpeg` static binary, creates a local `ffmpeg.exe` in the workspace, and injects its directory into `os.environ["PATH"]`.
2. **Dual-Path Audio Loader**:
   - **Primary**: Uses Python `soundfile` (`sf.read`) to decode WAV/FLAC/OGG directly from memory into a NumPy float32 array, resampling to 16 kHz via `scipy.signal.resample`. Whisper accepts NumPy arrays directly, completely bypassing `ffmpeg`.
   - **Fallback**: Falls back to the temporary file path with dynamically resolved `ffmpeg.exe` for compressed formats like MP3/M4A.

---

### 10. Integration with the Sugarcane Irrigation Decision System

In commercial sugarcane cultivation, water management directly dictates stalk elongation, internode formation, sucrose content, and cane yield:
- **Formative Stage (60–130 days)**: Critical period where water deficit reduces cane tonnage drastically.
- **Maturity / Ripening Stage**: Moderate moisture withholding enhances sucrose concentration.

The Farmer Query Assistant serves as the human-interaction frontend:
1. Translates spoken farmer concerns into structured agricultural intent and recognized entities (crops, conditions, quantities).
2. Cross-references live IoT soil tension / weather sensors before generating definitive irrigation schedule advisories.
3. Automatically triggers an agronomist review warning whenever retrieval similarity or classification confidence drops below safety thresholds.

---

### 11. Known Limitations & Future Technical Trajectory

1. **Acoustic Adaptation to Indian Accents**: Parameter-efficient fine-tuning (LoRA) of Whisper on Indian agricultural speech corpora (e.g., AI4Bharat IndicVoices, Bhashini).
2. **Dense Retrieval Augmentation**: Augment TF-IDF with a dense embedding retriever (e.g., `sentence-transformers/all-MiniLM-L6-v2`) in a hybrid BM25 + Dense retrieval pipeline.
3. **Conversational Memory**: Add multi-turn dialogue state tracking with LangChain to support follow-up questions.
