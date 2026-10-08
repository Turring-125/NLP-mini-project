# Technical Implementation Specification & Architecture Reference
## Project: Voice-Enabled Farmer Query Assistant (Advisory Layer for AI Sugarcane Irrigation)

---

### Table of Contents
1. [Executive Summary & Domain Objectives](#1-executive-summary--domain-objectives)
2. [End-to-End System Architecture & Dataflow](#2-end-to-end-system-architecture--dataflow)
3. [Dataset Profiles & Preprocessing Pipeline](#3-dataset-profiles--preprocessing-pipeline)
4. [Mathematical Formulations & Model Mechanics](#4-mathematical-formulations--model-mechanics)
   - 4.1 [Automatic Speech Recognition (ASR): OpenAI Whisper](#41-automatic-speech-recognition-asr-openai-whisper)
   - 4.2 [Linguistic Morphology: Stemming vs Lemmatization](#42-linguistic-morphology-stemming-vs-lemmatization)
   - 4.3 [Feature Representation: TF-IDF Vector Space](#43-feature-representation-tf-idf-vector-space)
   - 4.4 [Intent Classification: Multinomial Naive Bayes & Logistic Regression](#44-intent-classification-multinomial-naive-bayes--logistic-regression)
   - 4.5 [Information Retrieval Engine: Vector Space Cosine Similarity](#45-information-retrieval-engine-vector-space-cosine-similarity)
5. [Empirical Experimental Results & Benchmark Tables](#5-empirical-experimental-results--benchmark-tables)
6. [Qualitative Linguistic & ASR Error Analysis](#6-qualitative-linguistic--asr-error-analysis)
7. [Codebase Anatomy & Component Walkthrough](#7-codebase-anatomy--component-walkthrough)
8. [Audio Ingestion Pipeline & OS Compatibility Hardening](#8-audio-ingestion-pipeline--os-compatibility-hardening)
9. [Integration with the Sugarcane Irrigation Decision System](#9-integration-with-the-sugarcane-irrigation-decision-system)
10. [Known Limitations & Future Technical Trajectory](#10-known-limitations--future-technical-trajectory)

---

### 1. Executive Summary & Domain Objectives

This project builds a real-time, voice-enabled agricultural advisory assistant that converts spoken farmer queries into structured agricultural domains and retrieves expert agronomist recommendations from authentic Kisan Call Centre (KCC) records.

The primary application domain is serving as the conversational decision-support layer for an **AI-driven Sugarcane Irrigation Scheduling System**. In sugarcane (*Saccharum officinarum*), water management during the **formative stage (60–130 days post-planting)** directly dictates shoot elongation, tillering capacity, internode elongation, and eventual sucrose accumulation. Farmer queries during this phase frequently describe symptoms of water stress (leaf rolling, leaf scorch, soil cracking), nutrient deficiencies exacerbated by moisture deficits, or disease outbreaks (such as red rot, *Colletotrichum falcatum*). 

The assistant provides an instant, accurate voice-to-advisory pipeline that:
1. Ingests raw voice input (WAV/MP3/M4A/OGG) or text queries.
2. Transcribes acoustic speech using OpenAI Whisper models benchmarked on CPU.
3. Preprocesses and normalizes agricultural text using NLTK morphological analysis.
4. Classifies query intent into 9 specialized agricultural domains using a calibrated Logistic Regression model (achieving **99.15% accuracy** and **0.9740 Macro-F1**).
5. Queries a 72,708-document TF-IDF vector index to retrieve the top-3 historically validated expert recommendations (achieving **90.20% Precision@1** on unseen test queries).
6. Implements safety guardrails: warns the farmer (`"Low confidence - please consult an agronomist"`) when classifier probability $< 0.45$ or retrieval similarity $< 0.30$.

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
│  - Transformer Encoder-Decoder (Beam search / Greedy)  │
└────────────────────────────────────────────────────────┘
                 │
                 ▼ Raw Transcript Text
┌────────────────────────────────────────────────────────┐
│ NLP Preprocessing & Normalization                      │
│  - Lowercasing                                         │
│  - Regex punctuation & symbol removal: [^\w\s]         │
│  - NLTK English stopword filtering                     │
│  - Morphological normalization                         │
└────────────────────────────────────────────────────────┘
                 │
                 ▼ Cleaned Query String
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

### 3. Dataset Profiles & Preprocessing Pipeline

#### 3.1 Kisan Call Centre (KCC) Query-Answer Dataset
- **Origin**: Government of India, Ministry of Agriculture & Farmers Welfare Kisan Call Centre records (distributed via Kaggle Hub).
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
- **Stratification**: Guarantees exact class proportions across both splits down to minority classes (e.g., Post Harvest: 304 train / 76 test).

#### 3.2 Google FLEURS Speech Dataset (`en_us`)
- **Origin**: Google FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech), curated 50 audio clips of native/non-native spoken English saved in `data/fleurs_50.joblib`.
- **Audio Format**: 16,000 Hz, 1-channel mono PCM.
- **Ground Truth**: Manually transcribed reference sentences used to calculate exact Word Error Rate (WER) and Character Error Rate (CER).

---

### 4. Mathematical Formulations & Model Mechanics

#### 4.1 Automatic Speech Recognition (ASR): OpenAI Whisper

##### Acoustic Feature Extraction
Whisper processes audio $x(t)$ by converting the raw 16 kHz waveform into an 80-channel log-magnitude Mel spectrogram:
1. Audio divided into 25 ms frames (400 samples) with a 10 ms hop size (160 samples) using a Hanning window.
2. Fast Fourier Transform (FFT) computes the frequency spectrum.
3. Spectrum mapped onto 80 Mel-scale filter banks spaced logarithmically from 0 Hz to 8,000 Hz:
$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$
4. Natural logarithm applied: $S = \log(\text{MelSpectrum} + 10^{-5})$.
5. The resulting matrix is normalized to $[-1, 1]$.

##### Transformer Encoder-Decoder Architecture
- **Stem**: Two 1D convolutional layers with filter width 3 and stride 2 compress the temporal dimension by a factor of 4 (reducing 100 frames/sec to 50 Hz feature representations).
- **Encoder**: Pre-activation residual blocks with multi-head self-attention and GELU feed-forward networks.
- **Decoder**: Standard autoregressive sequence-to-sequence Transformer with causal multi-head self-attention and cross-attention over encoder outputs. Uses a 51,865-token byte-level Byte-Pair Encoding (BPE) vocabulary.

##### Model Specifications Compared:
| Parameter | `tiny` | `base` | `small` |
| :--- | :---: | :---: | :---: |
| Layers (Encoder / Decoder) | 4 / 4 | 6 / 6 | 12 / 12 |
| Hidden Dimension ($d_{model}$) | 384 | 512 | 768 |
| Attention Heads | 6 | 8 | 12 |
| Total Parameters | 39 Million | 74 Million | 244 Million |
| Relative CPU Compute Overhead | $1\times$ | $\sim 1.75\times$ | $\sim 4.3\times$ |

##### ASR Evaluation Metrics
1. **Word Error Rate (WER)**:
$$WER = \frac{S + D + I}{N} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Reference Word Count}}$$
Calculated using Levenshtein distance dynamic programming after text normalization (lowercasing, punctuation stripping).
2. **Character Error Rate (CER)**:
$$CER = \frac{S_c + D_c + I_c}{N_c}$$
Measured at character granularity, penalizing acoustic spelling deviations.

---

#### 4.2 Linguistic Morphology: Stemming vs Lemmatization

Natural Language Processing in agricultural advisory requires standardizing highly inflected terminology (e.g., *infestation*, *infesting*, *infested*).

1. **Porter Stemming Algorithm (`nltk.stem.PorterStemmer`)**:
   - A cascading heuristic rule-engine based on 5 sequential transformation phases.
   - Evaluates word endings against measure $m$ (vowel-consonant sequence count $[C](VC)^m[V]$).
   - Example rule: `(*v*) ING ->` reduces `dropping` to `drop`.
   - Characteristic: Fast, purely syntactic affix truncation that often yields non-lexical stems (e.g., `varieties` $\to$ `varieti`, `weedicides` $\to$ `weedicid`).

2. **WordNet Lemmatization (`nltk.stem.WordNetLemmatizer`)**:
   - Leverages the Princeton WordNet lexical database.
   - Applies morphological analysis (Morphy) to strip inflectional affixes and match words against canonical dictionary lemmas.
   - Characteristic: Computationally heavier, guarantees valid English base words (e.g., `varieties` $\to$ `variety`, `diseases` $\to$ `disease`).

---

#### 4.3 Feature Representation: TF-IDF Vector Space

Both the intent classifier and retrieval engine utilize Term Frequency - Inverse Document Frequency (TF-IDF) feature weighting.

Given term $t$, document $d$, and corpus $D$:
$$\text{TF}(t, d) = f_{t,d} \quad (\text{frequency of } t \text{ in } d)$$
$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

To prevent document length bias, vectors are normalized using the Euclidean ($L_2$) norm:
$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum_{i=1}^{M} v_i^2}}$$

- **Classifier Vectorizer**: $M = 10,000$ features, $n$-gram range $(1, 2)$ (captures phrases like *"red rot"*, *"urea dose"*, *"drip irrigation"*).
- **Retrieval Vectorizer**: $M = 15,000$ features, $n$-gram range $(1, 2)$.

---

#### 4.4 Intent Classification: Multinomial Naive Bayes & Logistic Regression

##### 1. Multinomial Naive Bayes (Baseline)
Assumes conditional independence between feature tokens given the class label $c_k$:
$$P(c_k \mid \mathbf{x}) \propto P(c_k) \prod_{j=1}^{M} P(w_j \mid c_k)^{x_j}$$
With Laplace (add-1) smoothing:
$$P(w_j \mid c_k) = \frac{N_{k,j} + 1}{N_k + M}$$
Where $N_{k,j}$ is the count of term $j$ in class $k$, and $N_k$ is the total token count in class $k$.

##### 2. Multinomial Logistic Regression (Main Classifier)
Models class posterior probabilities directly using the Softmax function over a linear combination of features:
$$P(Y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$

Objective function minimized using the L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) quasi-Newton algorithm with $L_2$ weight regularization:
$$\mathcal{L}(\mathbf{W}) = -\sum_{i=1}^{N} \sum_{k=1}^{K} \mathbb{I}(y_i = k) \log P(Y = k \mid \mathbf{x}_i) + \frac{1}{2C} \sum_{k=1}^{K} \|\mathbf{w}_k\|_2^2$$
Where $C = 1.0$, `max_iter = 300`.
Confidence score is computed as $\max_{k} P(Y=k \mid \mathbf{x})$.

---

#### 4.5 Information Retrieval Engine: Vector Space Cosine Similarity

The retrieval index encodes all $N = 72,708$ training questions into a sparse Compressed Sparse Row (CSR) matrix $\mathbf{A} \in \mathbb{R}^{72708 \times 15000}$.

When an input query $q$ is received:
1. $q$ is vectorized into unit-norm sparse vector $\mathbf{q} \in \mathbb{R}^{1 \times 15000}$.
2. Cosine similarities across all indexed documents are computed via single sparse matrix-vector multiplication:
$$\mathbf{s} = \mathbf{q} \mathbf{A}^T \in \mathbb{R}^{1 \times 72708}$$
Since both $\mathbf{q}$ and rows of $\mathbf{A}$ are $L_2$-normalized:
$$\text{sim}(q, d_i) = \mathbf{q} \cdot \mathbf{d}_i^T = \cos(\theta)$$
3. The top-$k$ ($k=3$) indices are extracted using partial quickselect (`np.argsort`):
$$\text{Top-}k = \operatorname{arg\,top-}k_{i} (\mathbf{s}_i)$$

##### Metric: Precision@1
Precision@1 measures whether the single most similar retrieved query belongs to the exact same agricultural domain as the test query:
$$\text{Precision@1} = \frac{1}{|Q_{\text{test}}|} \sum_{j=1}^{|Q_{\text{test}}|} \mathbb{I}\left(\operatorname{Category}(\operatorname{Top1}(q_j)) == \operatorname{Category}(q_j)\right)$$

---

### 5. Empirical Experimental Results & Benchmark Tables

#### 5.1 Speech Recognition Evaluation (FLEURS 50 Clips)
Benchmarked locally on Windows Intel CPU:

| Model | WER (%) | CER (%) | Latency / Clip | Total Time (50 clips) | Optimal Environment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Whisper `tiny`** | **15.21%** | **6.74%** | **0.45s** | 22.5s | Raspberry Pi / Low-power Edge |
| **Whisper `base`** | **10.52%** | **4.97%** | **0.79s** | 39.5s | **Standard Laptop / Real-time Production** |
| **Whisper `small`**| **6.76%**  | **2.81%** | **1.93s** | 96.5s | Server-side / Batch Processing |

*Trade-off analysis*: Whisper `base` reduces Word Error Rate by nearly **31% relative to `tiny`** (15.21% $\to$ 10.52%) while remaining sub-second per clip ($0.79\text{s}$), making it the recommended default.

---

#### 5.2 Morphology Benchmark: Stemming vs Lemmatization (10 Real Agricultural Words)

| Input Word | Porter Stemmer Output | WordNet Lemmatizer Output | Linguistic Mechanism Explained |
| :--- | :--- | :--- | :--- |
| `infestation` | `infest` | `infestation` | Stemmer drops derivational noun suffix `-ation` |
| `varieties` | `varieti` | `variety` | Lemmatizer resolves plural `-ies` to canonical noun root |
| `dropping` | `drop` | `dropping` | Stemmer removes inflectional participle suffix `-ing` |
| `fertilizers` | `fertil` | `fertilizer` | Lemmatizer handles plural noun inflection |
| `spraying` | `spray` | `spraying` | Stemmer reduces action verb to base root |
| `weedicides` | `weedicid` | `weedicides` | Stemmer drops trailing `-e` heuristically |
| `borers` | `borer` | `borer` | Lemmatizer maps plural `-s` to singular agent noun |
| `cultivation` | `cultiv` | `cultivation` | Stemmer aggressively truncates `-ation` |
| `germinating` | `germin` | `germinating` | Stemmer removes continuous suffix `-ating` |
| `diseases` | `diseas` | `disease` | Lemmatizer restores canonical vocabulary entry |

---

#### 5.3 Intent Classifier Benchmark (18,178 Unseen Test Queries)

| Architecture | Overall Accuracy | Macro-Average F1 | Weighted-Average F1 | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (Baseline)** | 92.32% | 0.7876 | 0.9184 | 0.42s |
| **Logistic Regression (Main Model)** | **99.15%** | **0.9740** | **0.9914** | 2.56s |

##### Detailed Classification Report (Logistic Regression on Test Set):
```
                              precision    recall  f1-score   support

         Agronomic Practices       0.97      0.96      0.97       723
Animal Husbandry & Fisheries       0.98      0.99      0.99      1066
            General Advisory       0.99      1.00      0.99      7548
 Government Schemes & Credit       1.00      0.98      0.99       636
         Nutrient Management       0.99      0.98      0.99      1590
            Plant Protection       1.00      0.99      1.00      6299
      Post Harvest & Storage       0.99      0.95      0.97        76
            Water Management       0.97      0.95      0.96       148
             Weed Management       0.99      0.86      0.92        92

                    accuracy                           0.99     18178
                   macro avg       0.99      0.96      0.97     18178
                weighted avg       0.99      0.99      0.99     18178
```

---

#### 5.4 Retrieval Engine Benchmark

- **Indexed Documents**: 72,708 historical farmer queries + agronomist answers.
- **Evaluation Set**: 18,178 unseen test queries.
- **Categorical Precision@1**: **90.20%** (16,396 / 18,178 queries correctly matched to the same domain at rank #1).
- **Inference Speed**: $< 1.5\text{ ms}$ per query across 72,708 documents via CSR matrix vectorization.

---

### 6. Qualitative Linguistic & ASR Error Analysis

Inspection of errors logged in [`results/asr_errors.txt`](file:///d:/clg/LY/NLP/mini%20project/results/asr_errors.txt) reveals four distinct linguistic failure categories:

#### 1. Phonetic Homophones & Substitutions
When phonetic acoustic representations overlap, smaller Whisper models bias toward conversational English rather than specialized vocabulary:
- Reference: `town of sintra and which was made famous`
  $\to$ Hypothesis: `town of sinatra which was made famous` (*sintra* $\to$ *sinatra*)
- Reference: `mountain gorilla tracking in africa`
  $\to$ Hypothesis: `mountain gorilla trafficking in africa` (*tracking* $\to$ *trafficking*)
- Reference: `two pools of genetic variation`
  $\to$ Hypothesis: `two poles of genetic variation` (*pools* $\to$ *poles*)
- Reference: `dont think about them as dinosaurs because`
  $\to$ Hypothesis: `dont think about them as senators because` (*dinosaurs* $\to$ *senators*)

#### 2. Number & Measurement Orthography
Whisper automatically normalizes spoken numerical expressions to digits, producing technical string mismatch:
- Reference: `twentieth century research has shown`
  $\to$ Hypothesis: `20th century research has shown`
- Reference: `simplest wholenumber ratio is therefore said to be 32`
  $\to$ Hypothesis: `simplest whole number ratio is therefore said to be free to 2`
- Reference: `photography format in the world is 35mm`
  $\to$ Hypothesis: `photography format in the world is 35 millimeter`

#### 3. Regional Proper Nouns & Agricultural Entities
Low-frequency proper nouns suffer character corruption:
- Reference: `climb the nyiragongo volcano`
  $\to$ Hypothesis: `climb that nairaiganggo volcano`
- Reference: `dunlap broadsides`
  $\to$ Hypothesis: `the net broad sides`

---

### 7. Codebase Anatomy & Component Walkthrough

Every script in the codebase is modular, self-contained, and constrained to $< 150$ lines.

```
d:/clg/LY/NLP/mini project/
├── explore.py              (106 lines) - Data ingestion & category mapping
├── asr_eval.py             (130 lines) - FLEURS Whisper ASR benchmark & error logging
├── train.py                (143 lines) - Morphology analysis, TF-IDF & classifier training
├── retrieve.py             (130 lines) - TF-IDF search engine & Precision@1 evaluation
├── pipeline_eval.py        (115 lines) - End-to-end user recording pipeline test
├── app.py                  (128 lines) - Streamlit web app with dual audio decoding
├── requirements.txt         (14 lines) - Dependency manifest with FFmpeg notes
├── README.md               (184 lines) - High-level project documentation
├── data/
│   ├── kcc_queries.csv     (178,939 records raw dataset)
│   ├── kcc_train.csv       (72,708 training QA pairs)
│   ├── kcc_test.csv        (18,178 test QA pairs)
│   └── fleurs_50.joblib    (50 audio clips + references)
├── models/
│   ├── classifier_bundle.joblib  (Vectorizer, LogisticRegression, Classes, Stopwords)
│   └── retrieval_bundle.joblib   (Vectorizer, CSR Matrix, Training DataFrame)
└── results/
    ├── asr_results.csv           (WER, CER, Latency table)
    ├── asr_wer_cer.png           (ASR performance chart)
    ├── asr_errors.txt            (15 qualitative error breakdowns)
    ├── classifier_results.csv    (NB vs LogReg performance)
    ├── confusion_matrix.png      (9-class confusion matrix plot)
    └── retrieval_results.csv     (Precision@1 results)
```

#### Detailed Script Breakdown:

1. **`explore.py`**:
   - Downloads/loads `kcc_queries.csv`.
   - Inspects missing values, shapes, and duplicate queries.
   - Maps raw call center query types into clean agricultural category taxonomy.
   - Prints sample queries across categories.

2. **`asr_eval.py`**:
   - Loads 50 audio clips and reference texts from `data/fleurs_50.joblib`.
   - Evaluates Whisper `tiny`, `base`, and `small` sequentially on CPU.
   - Computes WER and CER via `jiwer`.
   - Generates and saves `results/asr_results.csv`, `results/asr_wer_cer.png`, and `results/asr_errors.txt`.

3. **`train.py`**:
   - Executes `demonstrate_morphology()`: prints side-by-side table of Porter Stemmer vs WordNet Lemmatizer on 10 farm terms.
   - Cleans text (lowercase, regex punctuation strip, NLTK stopwords).
   - Generates stratified 80/20 train/test split.
   - Fits $N$-gram (1,2) TF-IDF vectorizer ($10,000$ features).
   - Trains Multinomial Naive Bayes baseline and Logistic Regression.
   - Computes and exports confusion matrix plot to `results/confusion_matrix.png`.
   - Saves model bundle to `models/classifier_bundle.joblib`.
   - Exports `data/kcc_train.csv` and `data/kcc_test.csv`.

4. **`retrieve.py`**:
   - Implements `FarmerQueryRetriever` class.
   - Builds TF-IDF sparse index ($15,000$ features) over training queries.
   - Evaluates batch `Precision@1` across all 18,178 test queries.
   - Exposes `search(query, top_k=3)` returning matched query, answer, category, and cosine score.
   - Serializes index to `models/retrieval_bundle.joblib`.

5. **`pipeline_eval.py`**:
   - Verifies existence of `my_recordings/` and `my_recordings/my_recordings.csv`.
   - If missing, logs explicit graceful skip message per project specification.
   - If present, transcribes user audio with Whisper, compares classified intent against reference text, and measures category change rate and corrupted farming terms.

6. **`app.py`**:
   - Streamlit interactive interface.
   - Features dual audio loading (direct `soundfile` in-memory decoding + `ffmpeg` fallback).
   - Dropdown for Whisper model size (`tiny`, `base`, `small`).
   - Displays recognized text, predicted category, confidence score, and top-3 retrieved historical answers in expandable cards.
   - Implements safety guardrail warning if confidence $< 0.45$ or similarity $< 0.30$.
   - Sidebar displays live benchmark metrics from CSV files.

---

### 8. Audio Ingestion Pipeline & OS Compatibility Hardening

On Windows environments, OpenAI Whisper's native `load_audio()` executes a subprocess call:
```python
cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-"]
subprocess.Popen(cmd, ...)
```
If `ffmpeg` is not in Windows system `PATH`, this raises `FileNotFoundError: [WinError 2] The system cannot find the file specified`.

#### Solution Implemented in `app.py`:
1. **Dynamic Binary Registration**:
   Locates the bundled `imageio-ffmpeg` static binary, creates a local `ffmpeg.exe` in the workspace, and injects its directory into `os.environ["PATH"]`.
2. **Dual-Path Audio Loader**:
   - **Primary (Zero-Subprocess)**: Uses Python `soundfile` (`sf.read`) to read WAV/FLAC/OGG directly from memory into a NumPy float32 array, resampling to 16 kHz via `scipy.signal.resample` if necessary. Whisper's `transcribe()` accepts NumPy arrays directly, completely bypassing `ffmpeg`.
   - **Fallback**: Falls back to the temporary file path with the dynamically resolved `ffmpeg.exe` for compressed formats like MP3/M4A.

---

### 9. Integration with the Sugarcane Irrigation Decision System

This voice assistant serves as the conversational intelligence layer for a broader IoT-enabled Sugarcane Irrigation Decision System:

```
┌────────────────────────────────────────────────────────┐
│               Farmer Voice Interaction                 │
│        "The sugarcane leaves are curling and           │
│         the soil is drying up, should I irrigate?"     │
└────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Voice-Enabled Farmer Query Assistant (This System)     │
│  - Recognizes audio query                              │
│  - Categorizes intent: "Water Management"              │
│  - Extracts matched past agronomist recommendation     │
└────────────────────────────────────────────────────────┘
                            │
                            ▼ Contextual Advisory Intent
┌────────────────────────────────────────────────────────┐
│ Sugarcane Irrigation Decision Engine                   │
│  - Query IoT Soil Moisture Sensors (Tension in kPa)   │
│  - Check Phenological Stage: Formative (Day 60-130)    │
│  - Query Weather Forecast API (Rainfall probability)   │
│  - Calculate Crop Evapotranspiration (ETc = Kc x ETo)  │
└────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Closed-Loop Action & Spoken Feedback                   │
│  - Automated Action: Trigger Solenoid Drip Valve       │
│  - Voice Output: "Soil moisture is at 45% depletion.   │
│    Formative stage requires irrigation. Initiating     │
│    drip cycle for 3 hours."                            │
└────────────────────────────────────────────────────────┘
```

#### Agronomic Irrigation Rules for Sugarcane:
- **Germination Phase (0–30 days)**: Frequent light irrigations to establish root zone.
- **Formative Phase (60–130 days)**: **Peak water requirement**. Soil moisture depletion must not exceed 50% available water capacity. Irrigation interval typically 7–10 days in summer. Deficit here leads to irreversible internode stunting.
- **Grand Growth Phase (130–250 days)**: Heavy vegetative growth; irrigation interval 10–12 days.
- **Ripening Phase (250–365 days)**: Moderate water deficit withheld 15–20 days prior to harvest to promote sucrose synthesis and prevent lodging.

---

### 10. Known Limitations & Future Technical Trajectory

1. **Acoustic Adaptation to Indian Accents & Rural Dialects**:
   - Whisper standard models were trained predominantly on standard English. When rural farmers speak with regional accents or code-switch (e.g., Hinglish, Kannada-English, Tamil-English), Word Error Rate increases.
   - *Future Work*: Parameter-efficient fine-tuning (LoRA) of Whisper on Indian agricultural speech corpora (e.g., AI4Bharat IndicVoices, Bhashini).

2. **Out-of-Vocabulary (OOV) Agrochemicals**:
   - Novel commercial pesticide formulations and bio-fertilizer trade names may not exist in the TF-IDF vocabulary.
   - *Future Work*: Augment TF-IDF with a dense embedding retriever (e.g., `sentence-transformers/all-MiniLM-L6-v2` or `BGE-small-en-v1.5`) in a hybrid BM25 + Dense retrieval pipeline.

3. **Multi-Turn Dialogue State Tracking**:
   - The current architecture operates in single-turn query $\to$ advisory mode.
   - *Future Work*: Add conversational memory with LangChain/LlamaIndex to support follow-up questions (e.g., *"How much does that fertilizer cost?"* or *"Can I mix it with insecticide?"*).
