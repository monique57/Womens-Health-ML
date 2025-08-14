# src/app_namcs.py  — SHAP-free prediction app for NAMCS 2019
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Women’s Health — NAMCS 2019", layout="wide")

# ---------------- Paths ----------------
MODEL_PATH  = "models/best_model_namcs2019.joblib"
LABELS_JSON = "models/class_labels.json"   # optional (maps class indices -> names)
DATA_PATH   = "data/namcs2019_clean.csv"   # optional (for sampling UI)

# -------------- Helpers ----------------
def detect_target_column(df: pd.DataFrame) -> str:
    keys = ["illness", "diagnosis", "disease", "condition", "target", "label"]
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    return df.columns[-1]  # last-column fallback

def expected_raw_columns(preprocessor, fallback_cols):
    """
    Derive raw input columns the pipeline's ColumnTransformer expects.
    Falls back to existing df columns if introspection fails.
    """
    try:
        cols = []
        for name, trans, sel in preprocessor.transformers_:
            if sel is None or sel == "remainder":
                continue
            if isinstance(sel, (list, tuple, np.ndarray)):
                cols.extend(list(sel))
            elif isinstance(sel, slice):
                cols.extend(list(fallback_cols[sel]))
            else:
                cols.append(sel)
        # Deduplicate, keep order
        seen, ordered = set(), []
        for c in cols:
            if c not in seen:
                ordered.append(c); seen.add(c)
        return ordered if ordered else list(fallback_cols)
    except Exception:
        return list(fallback_cols)

def align_raw_columns(df_row: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    """Ensure df_row has exactly required_cols (order + NaN for any missing)."""
    out = df_row.copy()
    for c in required_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[required_cols]

# -------------- Loaders ----------------
@st.cache_resource
def load_pipeline_and_labels():
    pipe = joblib.load(MODEL_PATH)
    label_map = None
    if os.path.exists(LABELS_JSON):
        try:
            with open(LABELS_JSON, "r", encoding="utf-8") as f:
                raw = json.load(f)
            label_map = {int(k): v for k, v in raw.items()}
        except Exception:
            label_map = None
    return pipe, label_map

@st.cache_resource
def load_sample_df():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, low_memory=False)
    return None

# =============== UI =====================
st.title("Women’s Health — NAMCS 2019")
st.caption("Predict using your trained NAMCS pipeline. (No SHAP in this app.)")

pipe, label_map = load_pipeline_and_labels()

# Identify final estimator & task
final_est = getattr(pipe, "named_steps", {}).get("model", pipe)
is_classifier = hasattr(final_est, "predict_proba") or hasattr(final_est, "classes_")

left, right = st.columns(2)
with left:
    st.markdown("**Model**")
    st.write(type(final_est).__name__)
    st.write("Task:", "Classification" if is_classifier else "Regression")
with right:
    st.markdown("**Artifacts**")
    st.code(os.path.abspath(MODEL_PATH))

st.divider()

st.subheader("1) Provide an input row")
mode = st.radio(
    "Choose input method",
    ["Upload single‑row CSV", "Pick a sample row", "Use a random dataset row"],
    horizontal=True
)

sample_df = load_sample_df()
input_df = None

if mode == "Upload single‑row CSV":
    up = st.file_uploader("Upload a CSV with exactly 1 row (raw columns).", type=["csv"])
    if up is not None:
        try:
            cand = pd.read_csv(up)
            if cand.shape[0] != 1:
                st.error("Please upload **exactly 1 row**.")
            else:
                input_df = cand
                st.success("Row loaded.")
                st.dataframe(input_df, use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

elif mode == "Pick a sample row":
    if sample_df is None:
        st.warning("Sample file not found at data/namcs2019_clean.csv")
    else:
        tgt = detect_target_column(sample_df)
        feats = sample_df.drop(columns=[tgt])
        idx = st.slider("Row index", 0, max(0, len(feats) - 1), 0)
        input_df = feats.iloc[[idx]].copy()
        st.caption("Selected sample row (features only)")
        st.dataframe(input_df, use_container_width=True)

else:  # random row
    if sample_df is None:
        st.warning("Sample file not found at data/namcs2019_clean.csv")
    else:
        tgt = detect_target_column(sample_df)
        feats = sample_df.drop(columns=[tgt])
        input_df = feats.sample(1, random_state=None).copy()
        st.caption("Random dataset row (features only)")
        st.dataframe(input_df, use_container_width=True)

st.divider()

st.subheader("2) Predict")

if st.button("🔮 Run prediction", type="primary"):
    if input_df is None:
        st.error("Please provide an input row first.")
        st.stop()
    try:
        # Align raw columns to preprocessor expectation
        if hasattr(pipe, "named_steps") and "preprocessor" in pipe.named_steps:
            pre = pipe.named_steps["preprocessor"]
            required = expected_raw_columns(pre, input_df.columns)
            aligned = align_raw_columns(input_df, required)
        else:
            aligned = input_df

        # Predict
        pred = pipe.predict(aligned)
        value = pred[0] if hasattr(pred, "__len__") else pred

        if is_classifier and hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(aligned)[0]
            classes = getattr(final_est, "classes_", np.arange(len(proba)))
            names = [label_map.get(int(c), str(c)) for c in classes] if label_map else [str(c) for c in classes]
            st.success(f"Predicted class: **{label_map.get(int(value), str(value)) if label_map else str(value)}**")
            st.caption("Class probabilities")
            st.dataframe(
                pd.DataFrame({"class": names, "probability": np.round(proba, 3)}).sort_values(
                    "probability", ascending=False
                ),
                use_container_width=True,
            )
        else:
            st.success(f"Predicted value: **{float(value):.4f}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("This app intentionally excludes SHAP for stability. "
           "If you need feature importance for reports, run `python src/shap_explain.py` separately.")
