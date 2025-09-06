from batch import run_batch,report
import os
import streamlit as st
from annotated_text import annotated_text, annotation
from sentiment_llm import MovieSentimentClassifier, segments_with_highlights
import pandas as pd
from typing import Dict, Any

# ----- CACHED HELPERS -----
@st.cache_resource
def get_clf(mode: str):
    """Cache the classifier instance per mode."""
    temperature = 0.1 if mode == "Strict" else 0.3
    return MovieSentimentClassifier(temperature=temperature)

@st.cache_data(show_spinner=False, ttl=24*60*60)
def classify_cached(text: str, mode: str) -> Dict[str, Any]:
    """Cache sentiment for (text, mode)."""
    res = get_clf(mode).classify(text.strip(), mode=mode)
    return {
        "label": res.label,
        "confidence": float(res.confidence),
        "explanation": res.explanation,
        "evidence_phrases": res.evidence_phrases or [],
    }

@st.cache_data(show_spinner=True, ttl=24*60*60)
def run_batch_cached(df_in: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Streamlit-side cached batch (first column = text). Does NOT touch batch.run_batch."""
    texts = df_in.iloc[:, 0].astype(str).str.strip().tolist()
    results = [classify_cached(t, mode) for t in texts]
    df_out = df_in.copy()
    df_out["pred"] = [r["label"] for r in results]
    df_out["confidence"] = [r["confidence"] for r in results]
    df_out["evidence_phrases"] = [r["evidence_phrases"] for r in results]
    # optional: include explanations too
    df_out["explanation"] = [r["explanation"] for r in results]
    return df_out

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="wide")
st.title("🎬 Movie Review Sentiment")


tab1, tab2 = st.tabs(["Text Analysis", "CSV Analysis"])
with tab1:
    left, right = st.columns([0.40, 0.60], gap="large")
    with left:
        st.subheader("Input")
        with st.form("sentiment_form", clear_on_submit=False):
            mode = st.radio("Mode", ["Strict", "Lenient"], horizontal=True, index=0)
            review = st.text_area("Movie review", height=120, placeholder="Paste or type the review…")
            go = st.form_submit_button("Analyze", type="primary")

    with right:
        st.subheader("Analysis")
        result_box = st.container()
        with result_box:
            if not go:
                st.info("Results will appear here after you click **Analyze**.")
            else:
                if not review.strip():
                    st.warning("Please type or paste a review.")
                elif not os.getenv("GEMINI_API_KEY", "").strip():
                    st.error("Set `GEMINI_API_KEY` in your environment.")
                else:
                    res = classify_cached(review.strip(), mode)
                    st.markdown(
                        f"##### **Sentiment:** {res['label']} &nbsp; | &nbsp; **Confidence:** {res['confidence']:.2f}"
                    )
                    st.markdown("##### **Explanation:**")
                    st.write(res["explanation"])

                    st.markdown("##### **Review with highlighted phrases:**")
                    annotated_text(*segments_with_highlights(review, res["evidence_phrases"]))

with tab2:
    st.subheader("Batch CSV")

    uploaded = st.file_uploader(
        "Upload CSV (first column: review text, second column (optional): actual sentiment label)",
        type=["csv"]
    )
    mode_batch = st.radio("Batch Mode", ["Strict", "Lenient"], horizontal=True, index=0, key="batch_mode")

    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        n = len(df_in)
        rpm = 10.0
        gap = (60.0 / rpm) + 0.25
        eta = int(n * gap)
        mins, secs = divmod(eta, 60)
        st.caption(f"Rows: **{n}** · ETA at **15 Requests per minutes**: ~ **{mins}m {secs}s**")

        if st.button("Run Batch", type="primary"):
            if not os.getenv("GEMINI_API_KEY", "").strip():
                st.error("Set `GEMINI_API_KEY` in your environment.")
            else:
                df_out = run_batch_cached(df_in, mode_batch)  # cached, Streamlit-only
 # throttled here
                # download
                st.download_button(
                    "Download predictions.csv",
                    data=df_out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
                # preview
                st.markdown("##### Preview")
                st.dataframe(df_out.head(5), use_container_width=True,hide_index=True)
                if(len(df_in.columns)>1):
                    m = report(df_out)
                # show metrics (only meaningful if labels were present)
                    st.markdown("##### Metrics")
                    
                    c1, c2, c3 = st.columns(3)
                    c3.metric("Correct", f"{m['correct']}")
                    c2.metric("Total", f"{m['total']}")
                    c1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")

                    CLASSES = ["Positive", "Neutral", "Negative"]
                    st.markdown("**Confusion matrix (rows = Actual, cols = Predicted)**")

                    actual = df_out.iloc[:, 1].astype(str).str.strip().str.title()
                    pred   = df_out["pred"].astype(str).str.strip().str.title()

                    cm = pd.crosstab(actual, pred).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
                    st.table(cm)    
