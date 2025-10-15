import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import importlib

st.set_page_config(page_title="Jet Engine Fault Detection", page_icon="✈️", layout="wide")

st.title("✈️ Jet Engine Fault Detection")
st.caption("Upload engine sensor CSV to get condition predictions")

artifacts_dir = Path("artifacts")

# Compatibility shim for scikit-learn pickled artifacts across versions
# Some artifacts created with older/newer scikit-learn reference private
# helpers like `_RemainderColsList` that may not exist in the running
# version. We define a lightweight fallback to allow unpickling.
try:
    ct_module = importlib.import_module("sklearn.compose._column_transformer")
    if not hasattr(ct_module, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        setattr(ct_module, "_RemainderColsList", _RemainderColsList)
except Exception:
    # If sklearn is unavailable or module path changed, ignore;
    # joblib.load will raise a clear error below which we surface in the UI.
    pass

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_files = sorted(artifacts_dir.glob("jet_fault_model_*.pkl"))
    if not model_files:
        st.stop()
    model = joblib.load(model_files[0])
    meta = joblib.load(artifacts_dir / "feature_metadata.pkl")
    return model, meta

try:
    model, metadata = load_artifacts()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload engine data CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    required_cols = metadata.get("numeric_features", [])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        df = df[required_cols]
        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        preds = model.predict(df)
        st.subheader("Predictions")
        st.write(pd.DataFrame({"prediction": preds}))

        # If classifier exposes predict_proba, show it
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)
                proba_df = pd.DataFrame(proba, columns=[f"class_{c}" for c in metadata.get("classes_", [])])
                st.subheader("Prediction Probabilities")
                st.dataframe(proba_df.head(), use_container_width=True)
            except Exception:
                pass

st.markdown("---")
st.caption("Model pipeline loaded from artifacts. Ensure artifact files exist before running.")


