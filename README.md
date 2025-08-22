# Create a "ready-to-go" hackathon project with code, docs, and deploy files.
import os, json, textwrap, zipfile, pathlib

root = "/mnt/data/credtech_explainable_score"
os.makedirs(root, exist_ok=True)
os.makedirs(f"{root}/engine", exist_ok=True)

# ----------------------- app.py -----------------------
app_py = r'''# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

from engine.data_sources import get_price_data, get_news_rss, infer_company_from_ticker
from engine.features import build_feature_frame
from engine.model import train_or_load_model, score_latest, score_series_with_explanations
from engine.explain import plot_feature_importance, evidence_cards
from engine.utils import scale_score

st.set_page_config(page_title="Explainable Credit Intelligence", layout="wide")

st.title("🔍 Explainable Credit Intelligence Platform")
st.caption("Real-time, explainable issuer credit score — with evidence you can trust.")

# --- Sidebar controls ---
default_ticker = st.sidebar.text_input("Ticker (e.g., AAPL, TCS.NS, RELIANCE.NS)", value="AAPL")
period = st.sidebar.selectbox("History Window", ["3mo", "6mo", "1y", "2y"], index=2)
target_horizon = st.sidebar.slider("Label horizon (days)", 5, 30, 10, step=1)
refresh = st.sidebar.button("🔄 Refresh Data")

# --- Data ingestion ---
with st.spinner("Fetching market data..."):
    px = get_price_data(default_ticker, period=period)
if px is None or px.empty:
    st.error("Could not load price data. Please try a different ticker or check your internet connection.")
    st.stop()

company_guess = infer_company_from_ticker(default_ticker)
with st.spinner("Fetching recent news..."):
    headlines = get_news_rss(company_guess or default_ticker)

# --- Feature engineering ---
feat_df = build_feature_frame(px, headlines_df=headlines)

# --- Train / Load model ---
with st.spinner("Training adaptive scoring model..."):
    model, training_info = train_or_load_model(feat_df, horizon_days=target_horizon)

colA, colB, colC = st.columns([2.2, 1.2, 1.2], gap="large")

with colA:
    st.subheader("Credit Score Timeline")
    scored, shap_info = score_series_with_explanations(model, feat_df)
    chart = pd.DataFrame({
        "Close": px["Close"].iloc[-len(scored):],
        "CreditScore": scored["credit_score"]
    }, index=scored.index)
    st.line_chart(chart)

with colB:
    st.subheader("Latest Score")
    latest_row = scored.iloc[-1]
    st.metric(
        label=f"{default_ticker} Credit Score (300-900)",
        value=int(latest_row['credit_score']),
        delta=None
    )
    st.write("**Risk band:**", latest_row['risk_band'])
    if latest_row["counterfactual"]:
        st.caption("🔁 Counterfactual suggestion:")
        st.write(latest_row["counterfactual"])

with colC:
    st.subheader("Model Snapshot")
    st.write(f"Model: **{training_info['model_name']}**")
    st.write(f"Train window: **{training_info['train_start']} → {training_info['train_end']}**")
    st.write(f"ROC-AUC (cv est.): **{training_info['cv_auc']:.3f}**")

st.divider()

# --- Explainability ---
st.subheader("Why this score? (Feature Contributions)")
fig_imp = plot_feature_importance(shap_info['values'], shap_info['feature_names'])
st.pyplot(fig_imp, clear_figure=True)

# --- Evidence: recent events ---
st.subheader("Evidence from Unstructured Data (Latest Headlines)")
cards = evidence_cards(headlines)
for c in cards:
    with st.expander(f"{c['title']} — sentiment: {c['sentiment_label']} ({c['sentiment']:.2f})"):
        st.write(c['summary'])
        st.caption(f"Published: {c['published']} • Source: {c['source']}")

st.divider()
st.caption("This demo fuses structured (prices) and unstructured (news) signals, "
           "learns an adaptive classifier, scales probabilities to a 300–900 score, "
           "and generates counterfactual suggestions to move to a safer band.")
'''

# ----------------------- engine/data_sources.py -----------------------
data_sources_py = r'''# engine/data_sources.py
import pandas as pd
import yfinance as yf
import feedparser
import re
from datetime import datetime, timezone

def get_price_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception:
        return None

def infer_company_from_ticker(ticker: str) -> str:
    # naive cleanup; users can override via sidebar
    return re.sub(r'\..*$', '', ticker)

def get_news_rss(query: str, max_items: int = 15) -> pd.DataFrame:
    # Google News RSS (no API key). Use quoted query for specificity.
    url = f"https://news.google.com/rss/search?q={query}+finance&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    records = []
    for entry in feed.entries[:max_items]:
        title = entry.get("title", "")
        summary = re.sub("<.*?>", "", entry.get("summary", ""))[:400]
        link = entry.get("link", "")
        published = entry.get("published", "")
        source = entry.get("source", {}).get("title", "Google News")
        # Normalize published
        try:
            pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pub_dt = None
        records.append({
            "title": title, "summary": summary, "link": link,
            "published": pub_dt, "source": source
        })
    return pd.DataFrame(records)
'''

# ----------------------- engine/features.py -----------------------
features_py = r'''# engine/features.py
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def _price_features(px: pd.DataFrame) -> pd.DataFrame:
    df = px.copy()
    df["ret_1d"] = df["Close"].pct_change()
    df["vol_10"] = df["ret_1d"].rolling(10).std()
    df["ma_10"] = df["Close"].rolling(10).mean() / df["Close"] - 1.0
    df["ma_30"] = df["Close"].rolling(30).mean() / df["Close"] - 1.0
    df["drawdown"] = df["Close"] / df["Close"].cummax() - 1.0
    df["rsi_14"] = _rsi(df["Close"], window=14)
    df["vol_spike"] = (df["Volume"] / df["Volume"].rolling(10).mean()) - 1.0 if "Volume" in df.columns else 0.0
    return df

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(window).mean()
    roll_down = pd.Series(down, index=series.index).rolling(window).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _headline_sentiment_score(headlines_df: pd.DataFrame) -> pd.DataFrame:
    if headlines_df is None or headlines_df.empty:
        return pd.DataFrame({"news_sentiment": []})
    sents = []
    for _, r in headlines_df.iterrows():
        text = f"{r.get('title','')} {r.get('summary','')}"
        s = analyzer.polarity_scores(text)["compound"]
        sents.append(s)
    scores = pd.DataFrame({
        "published": headlines_df["published"],
        "sentiment": sents
    }).dropna(subset=["published"]).set_index("published").sort_index()
    # 7-day rolling average sentiment
    scores = scores.resample("1D").mean().rolling(7, min_periods=1).mean().rename(columns={"sentiment": "news_sentiment"})
    return scores

def build_feature_frame(px: pd.DataFrame, headlines_df: pd.DataFrame = None) -> pd.DataFrame:
    pf = _price_features(px)
    ns = _headline_sentiment_score(headlines_df)
    # align on index
    df = pf.join(ns, how="left")
    df["news_sentiment"] = df["news_sentiment"].fillna(method="ffill").fillna(0.0)
    # target label for training: future 10-day drawdown risk (to be configured by horizon)
    return df
'''

# ----------------------- engine/model.py -----------------------
model_py = r'''# engine/model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime

from .utils import scale_score, risk_band_from_score
from .explain import shap_values_for_model, minimal_counterfactual

FEATURES = ["ret_1d","vol_10","ma_10","ma_30","drawdown","rsi_14","vol_spike","news_sentiment"]

def _make_labels(df: pd.DataFrame, horizon_days: int = 10, dd_thresh: float = -0.05) -> pd.Series:
    # Label 1 if "good" (no large drawdown ahead), 0 if "bad"
    future_min = df["Close"].shift(-1).rolling(horizon_days).min()
    curr = df["Close"]
    fut_dd = (future_min / curr) - 1.0
    y = (fut_dd > dd_thresh).astype(int)
    return y

def train_or_load_model(df: pd.DataFrame, horizon_days: int = 10):
    data = df.dropna(subset=FEATURES + ["Close"]).copy()
    y = _make_labels(data, horizon_days=horizon_days).dropna()
    X = data.loc[y.index, FEATURES]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1
        ))
    ])
    # time series aware CV
    tscv = TimeSeriesSplit(n_splits=5)
    auc = cross_val_score(pipe, X, y, cv=tscv, scoring="roc_auc").mean()

    pipe.fit(X, y)

    info = {
        "model_name": "RandomForestClassifier",
        "train_start": str(X.index.min().date()),
        "train_end": str(X.index.max().date()),
        "cv_auc": float(auc)
    }
    return pipe, info

def score_series_with_explanations(model, df: pd.DataFrame):
    X = df[FEATURES].dropna()
    proba = model.predict_proba(X)[:,1]  # probability of "good"
    scores = np.array([scale_score(p) for p in proba])
    scored = pd.DataFrame({
        "proba_good": proba,
        "credit_score": scores
    }, index=X.index)
    scored["risk_band"] = [risk_band_from_score(s) for s in scored["credit_score"]]

    # SHAP values
    shap_vals, feat_names = shap_values_for_model(model, X)

    # counterfactual on the latest row
    cf_text = minimal_counterfactual(model, X.iloc[[-1]], target_band="A-")
    scored.loc[X.index[-1], "counterfactual"] = cf_text

    shap_info = {"values": shap_vals, "feature_names": feat_names}
    return scored, shap_info

def score_latest(model, df: pd.DataFrame):
    s, _ = score_series_with_explanations(model, df)
    return s.iloc[-1]
'''

# ----------------------- engine/explain.py -----------------------
explain_py = r'''# engine/explain.py
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from .utils import score_band_thresholds, scale_score

def shap_values_for_model(model, X: pd.DataFrame):
    # Use TreeExplainer if available (RandomForest is supported)
    explainer = shap.TreeExplainer(model.named_steps["rf"])
    shap_vals = explainer.shap_values(model.named_steps["scaler"].transform(X))[1]
    feat_names = list(X.columns)
    return shap_vals, feat_names

def plot_feature_importance(shap_values, feature_names):
    # mean absolute SHAP
    vals = np.abs(shap_values).mean(axis=0)
    order = np.argsort(vals)[::-1]
    fig = plt.figure(figsize=(6,4))
    plt.bar([feature_names[i] for i in order], vals[order])
    plt.title("Feature Contributions (mean |SHAP|)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def minimal_counterfactual(model, x_row: pd.DataFrame, target_band: str = "A-") -> str:
    # Simple one-step counterfactual: nudge top-2 influential features in a helpful direction
    rf = model.named_steps["rf"]
    scaler = model.named_steps["scaler"]
    Xs = scaler.transform(x_row)
    proba = rf.predict_proba(Xs)[0,1]
    current_score = scale_score(proba)
    bands = score_band_thresholds()
    target_min = bands.get(target_band, 740)

    if current_score >= target_min:
        return "Already at or above target band."

    # SHAP on single row
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(Xs)[1][0]
    names = list(x_row.columns)
    # rank features by negative impact
    order = np.argsort(sv)  # ascending; most negative first
    suggestions = []
    # heuristic directions for features
    directions = {
        "vol_10": "↓ volatility over 10d",
        "drawdown": "↓ recent drawdown",
        "ret_1d": "↑ daily returns stability",
        "ma_10": "↑ 10d momentum",
        "ma_30": "↑ 30d momentum",
        "rsi_14": "keep RSI in 45–60 range",
        "vol_spike": "↓ abnormal volume spikes",
        "news_sentiment": "↑ improve news sentiment"
    }
    for idx in order[:2]:
        fname = names[idx]
        suggestions.append(directions.get(fname, f"improve {fname}"))
    return "To reach a safer band, focus on: " + "; ".join(suggestions)

def evidence_cards(headlines_df: pd.DataFrame):
    cards = []
    if headlines_df is None or headlines_df.empty:
        return cards
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    for _, r in headlines_df.iterrows():
        text = f"{r.get('title','')} {r.get('summary','')}"
        s = analyzer.polarity_scores(text)["compound"]
        cards.append({
            "title": r.get("title","(no title)"),
            "summary": r.get("summary","(no summary)")[:400],
            "sentiment": float(s),
            "sentiment_label": "positive" if s > 0.2 else ("negative" if s < -0.2 else "neutral"),
            "published": str(r.get("published","")),
            "source": r.get("source",""),
            "link": r.get("link","")
        })
    # sort by magnitude of sentiment (strongest first)
    cards.sort(key=lambda c: abs(c["sentiment"]), reverse=True)
    return cards
'''

# ----------------------- engine/utils.py -----------------------
utils_py = r'''# engine/utils.py
def scale_score(p_good: float) -> int:
    # Map probability [0,1] -> score [300,900]
    return int(300 + (900 - 300) * float(p_good))

def risk_band_from_score(score: int) -> str:
    if score >= 800: return "AA"
    if score >= 740: return "A-"
    if score >= 680: return "BBB"
    if score >= 620: return "BB"
    return "B"

def score_band_thresholds():
    return {"AA": 800, "A-": 740, "BBB": 680, "BB": 620, "B": 300}
'''

# ----------------------- requirements.txt -----------------------
requirements = r'''streamlit==1.37.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
yfinance>=0.2.40
feedparser>=6.0.10
vaderSentiment>=3.3.2
shap>=0.45.0
matplotlib>=3.7.0
plotly>=5.22.0
'''

# ----------------------- Dockerfile -----------------------
dockerfile = r'''# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
'''

# ----------------------- README.md -----------------------
readme = r'''# CredTech — Explainable Credit Intelligence (Ready-to-Run)

Out-of-the-box, explainable credit scoring demo that fuses **structured price data** with **unstructured news sentiment**, produces a **300–900 credit score**, shows **SHAP feature contributions**, and generates **counterfactual suggestions** ("what to improve to reach A-").

## 🚀 Quickstart (no Docker)

```bash
# 1) Create a virtualenv (recommended)
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# 2) Install deps
pip install -r requirements.txt

# 3) Run app
streamlit run app.py
