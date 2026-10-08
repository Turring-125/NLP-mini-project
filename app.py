"""
app.py - Step 5: Voice-Enabled Farmer Query Assistant (Streamlit Web App)
Advisory & Chatbot layer for Sugarcane & Agricultural Irrigation / Farming.
Transcribes spoken queries with Whisper, classifies advisory domain, and
retrieves top-3 historical expert answers with similarity scores.
"""

import os
import tempfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import whisper
from retrieve import FarmerQueryRetriever
from train import clean_text

# Ensure ffmpeg binary is found by Whisper subprocess on Windows
try:
    import imageio_ffmpeg
    ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.path.abspath(".") + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

st.set_page_config(page_title="Farmer Voice Assistant", page_icon="🌾", layout="wide")

@st.cache_resource
def load_models():
    """Load classifier bundle and retrieval index bundle."""
    clf_bundle = joblib.load(os.path.join("models", "classifier_bundle.joblib"))
    retriever = FarmerQueryRetriever.load(os.path.join("models", "retrieval_bundle.joblib"))
    return clf_bundle, retriever

@st.cache_resource
def get_whisper(model_size: str):
    """Load cached Whisper model on CPU."""
    return whisper.load_model(model_size, device="cpu")

def main():
    st.title("🌾 Voice-Enabled Farmer Query Assistant")
    st.markdown(
        "**AI Irrigation & Agronomy Advisory Layer for Sugarcane Farmers** | "
        "Powered by OpenAI Whisper, NLTK & Kisan Call Centre (KCC) Real Data"
    )

    clf_bundle, retriever = load_models()
    vectorizer = clf_bundle["vectorizer"]
    classifier = clf_bundle["model"]
    stop_words = clf_bundle["stop_words"]

    # Sidebar: Metrics & Benchmarks
    st.sidebar.header("📊 Model Benchmark Dashboard")
    asr_path = os.path.join("results", "asr_results.csv")
    if os.path.exists(asr_path):
        st.sidebar.subheader("1. Whisper ASR Benchmark (FLEURS en_us)")
        asr_df = pd.read_csv(asr_path)
        st.sidebar.dataframe(asr_df, use_container_width=True, hide_index=True)

    st.sidebar.subheader("2. NLP Classifier Performance")
    st.sidebar.metric(label="Classifier Accuracy (LogReg)", value="99.15%")
    st.sidebar.metric(label="Macro-F1 Score", value="0.9740")

    st.sidebar.subheader("3. Knowledge Retrieval")
    st.sidebar.metric(label="Precision@1 (KCC Test)", value="90.20%")
    st.sidebar.info("Knowledge Base: 72,708 authentic KCC query-answer records.")

    # Main Area: Audio Upload or Query Input
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Audio / Voice Input")
        model_size = st.selectbox("Select Whisper Model Size", ["tiny", "base", "small"], index=0)
        uploaded_audio = st.file_uploader("Upload farmer voice recording (WAV, MP3, M4A)", type=["wav", "mp3", "m4a", "ogg"])
        manual_query = st.text_input("Or test with text query (e.g., 'control of red rot in sugarcane'):")

    query_text = ""
    if uploaded_audio is not None:
        st.audio(uploaded_audio)
        with st.spinner(f"Transcribing audio with Whisper '{model_size}' on CPU..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_audio.read())
                tmp_path = tmp_file.name
            try:
                whisper_model = get_whisper(model_size)
                # Try direct soundfile loading first (bypasses ffmpeg for WAV/FLAC/OGG)
                audio_input = tmp_path
                try:
                    import soundfile as sf
                    data, sr = sf.read(tmp_path)
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    if sr != 16000:
                        import scipy.signal
                        samples = int(len(data) * 16000 / sr)
                        data = scipy.signal.resample(data, samples)
                    audio_input = data.astype(np.float32)
                except Exception:
                    audio_input = tmp_path

                res = whisper_model.transcribe(audio_input, language="en", fp16=False)
                query_text = res["text"].strip()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    elif manual_query:
        query_text = manual_query.strip()

    with col2:
        st.subheader("2. Assistant Advisory Response")
        if query_text:
            st.success(f"**Recognized Query:** \"{query_text}\"")
            
            # Predict Category & Confidence
            clean_q = clean_text(query_text, stop_words)
            vec = vectorizer.transform([clean_q])
            pred_category = classifier.predict(vec)[0]
            probs = classifier.predict_proba(vec)[0]
            confidence = float(np.max(probs))

            m1, m2 = st.columns(2)
            m1.metric("Predicted Domain", pred_category)
            m2.metric("Confidence", f"{confidence:.1%}")

            # Retrieve top 3 answers
            results = retriever.search(query_text, top_k=3)
            top_sim = results[0]["similarity"] if results else 0.0

            # Guardrail: Check confidence and similarity threshold
            if confidence < 0.45 or top_sim < 0.30:
                st.warning("⚠️ **Low confidence - please consult an agronomist.** The system is unsure about this query.")

            st.markdown("### Top Retrieved Historical KCC Expert Advisories")
            for idx, r in enumerate(results, 1):
                with st.expander(f"Recommendation #{idx}: {r['query']} (Similarity: {r['similarity']:.2f})", expanded=(idx==1)):
                    st.write(f"**Domain:** `{r['category']}`")
                    st.write(f"**Expert Answer:** {r['answer']}")
        else:
            st.info("Upload an audio clip or type a query above to see the assistant in action.")

if __name__ == "__main__":
    main()
