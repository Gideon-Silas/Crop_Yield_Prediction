import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Design system — earthy / agronomic, not the default AI-cream-terracotta look
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg: #14231A;
    --bg-raised: #1B2E22;
    --gold: #E3A343;
    --sage: #8AA37E;
    --rust: #C1562E;
    --text: #F3EFE3;
    --text-dim: #B9C4B4;
    --line: rgba(243, 239, 227, 0.12);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(227, 163, 67, 0.08), transparent 45%),
        radial-gradient(circle at 85% 100%, rgba(138, 163, 126, 0.10), transparent 50%),
        var(--bg);
}

#MainMenu, header, footer {visibility: hidden;}

.block-container {
    padding-top: 2.5rem;
    max-width: 720px;
}

/* ---- Header ---- */
.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--sage);
    margin-bottom: 0.6rem;
}

.hero-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.6rem;
    line-height: 1.08;
    color: var(--text);
    margin-bottom: 0.5rem;
}

.hero-title em {
    font-style: italic;
    color: var(--gold);
}

.hero-sub {
    font-size: 1rem;
    color: var(--text-dim);
    max-width: 46ch;
    margin-bottom: 2.2rem;
    line-height: 1.55;
}

hr.divider {
    border: none;
    border-top: 1px solid var(--line);
    margin: 2rem 0 1.6rem 0;
}

/* ---- Section labels ---- */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--sage);
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-label::before {
    content: "";
    width: 14px;
    height: 1px;
    background: var(--sage);
    display: inline-block;
}

/* ---- Inputs ---- */
div[data-baseweb="select"] > div, .stNumberInput input, .stSlider {
    background-color: var(--bg-raised) !important;
    border-color: var(--line) !important;
    color: var(--text) !important;
}

label, .stSlider label, .stNumberInput label, .stSelectbox label {
    color: var(--text-dim) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ---- Button ---- */
.stButton > button {
    background: var(--gold);
    color: #14231A;
    font-weight: 700;
    border: none;
    border-radius: 4px;
    padding: 0.7rem 1.8rem;
    font-size: 0.95rem;
    letter-spacing: 0.01em;
    transition: transform 0.15s ease, background 0.15s ease;
    width: 100%;
}

.stButton > button:hover {
    background: #EFB65C;
    transform: translateY(-1px);
}

/* ---- Result card ---- */
.result-card {
    background: var(--bg-raised);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1.8rem 1.8rem 1.6rem 1.8rem;
    margin-top: 1.8rem;
}

.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--sage);
    margin-bottom: 0.4rem;
}

.result-number {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 3rem;
    color: var(--gold);
    line-height: 1;
    margin-bottom: 0.2rem;
}

.result-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: var(--text-dim);
}

/* ---- Gauge (signature element) ---- */
.gauge-wrap {
    margin-top: 1.6rem;
}

.gauge-track {
    position: relative;
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #4A3820 0%, var(--sage) 45%, var(--gold) 75%, var(--rust) 100%);
    margin: 1rem 0 0.5rem 0;
}

.gauge-marker {
    position: absolute;
    top: -7px;
    width: 2px;
    height: 20px;
    background: var(--text);
}

.gauge-marker::after {
    content: "";
    position: absolute;
    top: -6px;
    left: -4px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--text);
    box-shadow: 0 0 0 3px var(--bg-raised);
}

.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-dim);
    letter-spacing: 0.02em;
}

.footnote {
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-top: 1.1rem;
    line-height: 1.5;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid var(--line);
    margin-bottom: 2rem;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-dim);
    padding: 0.7rem 0.2rem;
    background: transparent;
}

.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--gold);
}

/* ---- Findings list ---- */
.finding {
    display: flex;
    gap: 0.9rem;
    padding: 1rem 0;
    border-bottom: 1px solid var(--line);
}

.finding:last-child { border-bottom: none; }

.finding-stat {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.5rem;
    color: var(--gold);
    min-width: 4.2rem;
    line-height: 1.3;
}

.finding-text {
    font-size: 0.92rem;
    color: var(--text-dim);
    line-height: 1.55;
    padding-top: 0.15rem;
}

/* ---- Video card ---- */
.video-intro {
    font-size: 0.92rem;
    color: var(--text-dim);
    line-height: 1.6;
    margin-bottom: 1.4rem;
}

/* ---- About / bio ---- */
.bio-intro {
    font-size: 0.98rem;
    color: var(--text);
    line-height: 1.7;
    margin-bottom: 1.6rem;
}

.interest-item {
    display: flex;
    align-items: baseline;
    gap: 0.7rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--line);
}

.interest-check {
    font-family: 'JetBrains Mono', monospace;
    color: var(--gold);
    font-weight: 600;
}

.interest-text {
    font-size: 0.92rem;
    color: var(--text-dim);
}

/* ---- Project explained ---- */
.explain-block {
    margin-bottom: 1.6rem;
}

.explain-heading {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.15rem;
    color: var(--gold);
    margin-bottom: 0.4rem;
}

.explain-text {
    font-size: 0.92rem;
    color: var(--text-dim);
    line-height: 1.65;
}

/* ---- Footer ---- */
.app-footer {
    margin-top: 3rem;
    padding-top: 1.4rem;
    border-top: 1px solid var(--line);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0.03em;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Model + encoder loading
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")
ITEM_ENCODER_PATH = os.path.join(BASE_DIR, "item_encoder.pkl")   # encodes crop ("Item")
AREA_ENCODER_PATH = os.path.join(BASE_DIR, "area_encoder.pkl")   # encodes country ("Area")

@st.cache_resource
def load_pickle(path):
    return joblib.load(path)

model, item_encoder, area_encoder = None, None, None
load_errors = []

for label, path, setter in [
    ("model", MODEL_PATH, "model"),
    ("item_encoder", ITEM_ENCODER_PATH, "item_encoder"),
    ("area_encoder", AREA_ENCODER_PATH, "area_encoder"),
]:
    if os.path.exists(path):
        try:
            obj = load_pickle(path)
            if setter == "model":
                model = obj
            elif setter == "item_encoder":
                item_encoder = obj
            else:
                area_encoder = obj
        except Exception as e:
            load_errors.append(f"{label}: {e}")
    else:
        load_errors.append(f"{label}: file not found at {path}")

model_error = " | ".join(load_errors) if load_errors else None

# ----------------------------------------------------------------------------
# Reference bands for the gauge — typical low/mid/high yield (hg/ha) across
# the dataset's crop mix. Adjust GAUGE_MIN / GAUGE_MAX if your data differs.
# ----------------------------------------------------------------------------
GAUGE_MIN = 0
GAUGE_MAX = 200000  # tubers/roots run high; cereals/legumes sit much lower

# Dropdown options come straight from the fitted encoders, so the app can
# only ever offer crop/country names the model actually recognizes.
if item_encoder is not None:
    CROPS = sorted(item_encoder.classes_.tolist())
else:
    CROPS = ["Maize", "Wheat", "Rice, paddy", "Potatoes", "Cassava"]

if area_encoder is not None:
    COUNTRIES = sorted(area_encoder.classes_.tolist())
else:
    COUNTRIES = ["Nigeria", "India", "United States of America", "Brazil", "China"]

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<div class="eyebrow">3MTT · Nextgen Capstone by Gideon Monday Silas</div>', unsafe_allow_html=True)
st.title("Title: Crop Yield Prediction")
st.markdown(
    '<div class="hero-title">What will your <em>harvest</em> yield?</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub">In this project, a Random Forest model was used to train on 28,242 records across '
    '101 countries and 10 different crop types, from 1990–2013.<br>'
    'The predictive model uses historical crop and environmental data to predict yields.<br>' 
    'Now, explore the findings, watch the walkthrough, '
    'or predict a yield yourself.</div>',
    unsafe_allow_html=True,
)

tab_Findings, tab_video, tab_predict = st.tabs(
    ["Findings", "Demo Video", "Predictor"]
)

# ----------------------------------------------------------------------------
# Tab 1 — Findings / key findings
# ----------------------------------------------------------------------------
with tab_Findings:
    st.markdown("""
<style>
.finding {
    display: grid;
    grid-template-columns: 140px 1fr;
    gap: 5px;
    align-items: start;
    margin-bottom: 5px;
}

.finding-stat {
    font-size: 28px;
    font-weight: 700;
}

.finding-text {
    font-size: 16px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)
    
    findings = [
        ("61%", "Crop type is the strongest contributor to yield. Tubers and roots produce 5–10x more per hectare than cereals, legumes, and other crops."),
    
    ("R²=0.986", "The model explains that 0.986 is the Random Forest regressor's performance score in predicting yield across the full 28,242-record dataset."),
    
    ("-0.55 / -0.37", "This is temperature's correlation with yield for Maize and Wheat respectively. Climate impact is masked when all crops are analyzed together, but becomes clear once disaggregated by crop."),
    
    ("35%", "The combined influence of rainfall and pesticide application (both of which fail to reach the top-performing crop type and temperature).")
    ]

    for stat, text in findings:
        st.markdown(
            f"""
            <div class="finding">
                <div class="finding-stat">{stat}</div>
                <div class="finding-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# Tab 2 — Demo video
# ----------------------------------------------------------------------------
with tab_video:
    st.markdown('<div class="section-label">Project walkthrough</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="video-intro">A short walkthrough of the dataset, key findings, '
        'model training, and a live run of the predictor below.</div>',
        unsafe_allow_html=True,
    )

    VIDEO_PATH = os.path.join(BASE_DIR, "demo_video.mp4")
    if os.path.exists(VIDEO_PATH):
        st.video(VIDEO_PATH)
    else:
        st.info(
            "No video found yet. Drop a file named `demo_video.mp4` in this app's folder, "
            "or paste a YouTube/hosted link below."
        )
        video_url = st.text_input("Video URL (optional)", placeholder="https://youtu.be/...")
        if video_url:
            st.video(video_url)

# ----------------------------------------------------------------------------
# Tab 3 — Predictor
# ----------------------------------------------------------------------------
with tab_predict:
    st.markdown('<div class="section-label">Growing conditions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        area = st.selectbox("Country / region", COUNTRIES)
        crop = st.selectbox("Crop", CROPS)
        year = st.slider("Year", min_value=1990, max_value=2030, value=2024)

    with col2:
        rainfall = st.number_input("Average rainfall (mm/year)", min_value=0.0, max_value=3000.0, value=800.0, step=10.0)
        pesticides = st.number_input("Pesticide use (tonnes)", min_value=0.0, max_value=50000.0, value=500.0, step=10.0)
        temp = st.number_input("Average temperature (°C)", min_value=-10.0, max_value=45.0, value=22.0, step=0.5)

    st.write("")
    predict_clicked = st.button("Predict yield →")

    # ------------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------------
    if predict_clicked:
        if model is None or item_encoder is None or area_encoder is None:
            st.error(
                f"Couldn't load required files. {model_error}\n\n"
                "Make sure `random_forest_model.pkl`, `item_encoder.pkl`, and "
                "`area_encoder.pkl` all sit in the same folder as this app."
            )
        else:
            item_code = item_encoder.transform([crop])[0]
            area_code = area_encoder.transform([area])[0]

            input_df = pd.DataFrame([{
                "Year": year,
                "average_rain_fall_mm_per_year": rainfall,
                "pesticides_tonnes": pesticides,
                "avg_temp": temp,
                "Item_encoded": item_code,
                "Area_encoded": area_code,
            }])

            try:
                prediction = model.predict(input_df)[0]
            except Exception as e:
                st.error(
                    "The model didn't accept these inputs — the column names or order "
                    "still don't match what it was trained on.\n\n"
                    f"Error: {e}\n\n"
                    "In your notebook, run `X.columns.tolist()` right after the line that "
                    "builds `X` (before `train_test_split`) and send me the exact output — "
                    "I'll match `input_df` in `app.py` to it exactly."
                )
                prediction = None

            if prediction is not None:
                pct = max(0, min(1, (prediction - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN)))

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Predicted yield — {crop}, {area}</div>
                        <div class="result-number">{prediction:,.0f}</div>
                        <div class="result-unit">hg / ha</div>
                        <div class="gauge-wrap">
                            <div class="gauge-track">
                                <div class="gauge-marker" style="left: calc({pct*100}% - 1px);"></div>
                            </div>
                            <div class="gauge-labels">
                                <span>0</span>
                                <span>low-yield crops</span>
                                <span>tubers &amp; roots</span>
                                <span>{GAUGE_MAX:,}</span>
                            </div>
                        </div>
                        <div class="footnote">
                            Crop type is the strongest driver of yield in this dataset (~61% importance),
                            which is why tuber crops like cassava and potatoes sit far higher on this
                            scale than cereals like maize or wheat, regardless of climate inputs.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.markdown(
    '<div class="app-footer">Crop Yield Prediction · Random Forest Regressor · R² = 0.986 · '
    'Dataset: Kaggle Crop Yield Prediction</div>',
    unsafe_allow_html=True,
)