import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # required on Streamlit Cloud (headless)
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import shap
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb

# ------------------------------------------------------------------
# 0. PATHS, CONSTANTS, GLOBAL STYLE
# ------------------------------------------------------------------
ROOT = Path(__file__).parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
WINDOW_DAYS = 60

st.set_page_config(page_title="Transfer Value Lab", page_icon="⚽",
                   layout="wide", initial_sidebar_state="expanded")

pio.templates.default = "plotly_dark"

plt.rcParams.update({
    "figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117",
    "savefig.facecolor": "#0e1117", "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3", "xtick.color": "#9aa4b2",
    "ytick.color": "#9aa4b2", "axes.edgecolor": "#39414d",
})

BG = "rgba(0,0,0,0)"          # transparent plot backgrounds
GRID = "rgba(148,163,184,0.10)"
POS, NEG = "#34d399", "#f87171"
HYPE, PRAISE, CRIT = "#f5b942", "#34d399", "#f87171"
VIOLET = "#8b5cf6"

st.markdown("""
<style>
/* ---------- hero ---------- */
.hero{padding:1.3rem 1.6rem;border-radius:16px;color:#fff;margin-bottom:1.2rem;
background:linear-gradient(115deg,#0b3d2e 0%,#0f2027 45%,#203a43 75%,#2c5364 100%);
border:1px solid rgba(255,255,255,.08);}
.hero h1{margin:0;font-size:1.9rem;letter-spacing:.01em}
.hero p{margin:.3rem 0 .6rem;opacity:.85;font-size:.95rem}
.chips span{display:inline-block;background:rgba(255,255,255,.09);
border:1px solid rgba(255,255,255,.16);border-radius:999px;padding:3px 12px;
font-size:.75rem;margin:0 6px 6px 0;color:#dbe7f3}

/* ---------- player header card ---------- */
.player-card{display:flex;align-items:center;gap:18px;background:#161b22;
border:1px solid #232a35;border-radius:16px;padding:16px 20px;margin-bottom:14px}
.player-img{width:92px;height:92px;object-fit:cover;border-radius:12px;
border:1px solid #2c3542;background:#0e1117}
.player-avatar{width:92px;height:92px;border-radius:12px;background:#1f2733;
display:flex;align-items:center;justify-content:center;font-size:2.2rem}
.player-name{font-size:1.35rem;font-weight:700;color:#f1f5f9}
.player-meta{font-size:.85rem;color:#8b94a3;margin-top:2px}
.badge{display:inline-block;background:#1f2733;border:1px solid #2c3542;
border-radius:999px;padding:2px 10px;font-size:.72rem;color:#aeb8c6;margin:6px 8px 0 0}
.badge.future{background:rgba(139,92,246,.14);border-color:rgba(139,92,246,.5);color:#c4b5fd}
.tm-btn{margin-left:auto;text-decoration:none;background:#1f2733;color:#dbe7f3;
border:1px solid #2c3542;border-radius:10px;padding:8px 14px;font-size:.8rem;white-space:nowrap}
.tm-btn:hover{background:#27313f;color:#fff}

/* ---------- KPI cards ---------- */
.kpi-grid{display:flex;gap:14px;margin-bottom:6px}
.kpi-card{flex:1;background:#161b22;border:1px solid #232a35;border-radius:14px;
padding:14px 18px;border-top:3px solid var(--accent,#2ecc71)}
.kpi-title{font-size:.72rem;color:#8b94a3;text-transform:uppercase;letter-spacing:.07em}
.kpi-value{font-size:1.65rem;font-weight:700;color:#f1f5f9;margin-top:3px}
.kpi-sub{font-size:.78rem;color:#8b94a3;margin-top:2px}

/* ---------- section polish ---------- */
.section-h{font-size:1.15rem;font-weight:650;color:#e6edf3;margin:1.1rem 0 .4rem}
.muted{color:#8b94a3;font-size:.82rem}
hr{border-color:#232a35}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"]{gap:6px}
.stTabs [data-baseweb="tab"]{background:#161b22;border:1px solid #232a35;
border-radius:10px 10px 0 0;padding:8px 18px}
.stTabs [aria-selected="true"]{background:#1f2733;border-bottom:2px solid #2ecc71}

/* ---------- footer ---------- */
.footer{margin-top:2.5rem;padding-top:1rem;border-top:1px solid #232a35;
color:#6b7484;font-size:.78rem;text-align:center}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 1. FEATURE METADATA
# ------------------------------------------------------------------
ALL_FEATURES = json.load(open(MODELS / f"feature_list_{WINDOW_DAYS}d.json"))
POSITION_COLS = [c for c in ALL_FEATURES if c.startswith("position_")]
NLP_FEATURES = [c for c in ALL_FEATURES if c in {
    "log_reddit_volume", "recent_volume_14d", "avg_hype_prob", "avg_mkt_neg_prob",
    "avg_perf_pos_prob", "avg_perf_neg_prob", "std_hype_prob", "std_perf_neg_prob",
    "wavg_hype_prob", "wavg_mkt_neg_prob", "wavg_perf_pos_prob", "wavg_perf_neg_prob",
    "momentum_hype", "momentum_perf_pos", "momentum_perf_neg", "momentum_mkt_neg"}]

FEATURE_LABELS = {
    "age_at_transfer": "Age at transfer", "age_sq": "Age²",
    "market_value_lagged_adj": "Market value (≥180d prior, 2024 €)",
    "has_lagged_mv": "Has prior valuation",
    "goals_per_90": "Goals / 90", "assists_per_90": "Assists / 90",
    "sb_xg_per_90": "xG / 90 (StatsBomb)", "sb_xa_per_90": "xA / 90 (StatsBomb)",
    "total_minutes_window": "Minutes played (window)",
    "has_statsbomb": "StatsBomb coverage",
    "contract_years_remaining": "Contract years left (capped 5)",
    "is_january_window": "January window",
    "from_league_tier": "Selling-league tier", "to_league_tier": "Buying-league tier",
    "league_step": "League step up/down",
    "buying_club_avg_spend": "Buying club avg spend",
    "log_reddit_volume": "Reddit volume (log)", "recent_volume_14d": "Comments (final 14d)",
    "avg_hype_prob": "Market hype", "avg_mkt_neg_prob": "Market negativity",
    "avg_perf_pos_prob": "Performance praise", "avg_perf_neg_prob": "Performance criticism",
    "std_hype_prob": "Hype volatility", "std_perf_neg_prob": "Criticism volatility",
    "wavg_hype_prob": "Hype (upvote-weighted)", "wavg_mkt_neg_prob": "Mkt negativity (weighted)",
    "wavg_perf_pos_prob": "Praise (weighted)", "wavg_perf_neg_prob": "Criticism (weighted)",
    "momentum_hype": "Hype momentum (14d)", "momentum_perf_pos": "Praise momentum (14d)",
    "momentum_perf_neg": "Criticism momentum (14d)", "momentum_mkt_neg": "Mkt-neg momentum (14d)",
    "position_Defender": "Position: Defender", "position_Goalkeeper": "Position: Goalkeeper",
    "position_Midfield": "Position: Midfield", "position_Unknown": "Position: Unknown",
}
DISPLAY_NAMES = [FEATURE_LABELS.get(f, f) for f in ALL_FEATURES]

GROUPS = {
    "💰 Financial": ["market_value_lagged_adj", "has_lagged_mv", "contract_years_remaining",
                     "is_january_window", "buying_club_avg_spend"],
    "⚽ Performance": ["goals_per_90", "assists_per_90", "sb_xg_per_90", "sb_xa_per_90",
                       "total_minutes_window", "has_statsbomb"],
    "🧬 Profile & League": ["age_at_transfer", "age_sq", "from_league_tier",
                            "to_league_tier", "league_step"] + POSITION_COLS,
    "📱 Reddit NLP": NLP_FEATURES,
}

def eur(x):
    return f"€{x/1e6:,.1f}M" if abs(x) >= 1e6 else f"€{x/1e3:,.0f}k"

def style_dark(fig, height=None, legend_top=True):
    """Apply consistent dark styling to a Plotly figure."""
    lay = dict(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#e6edf3"),
               margin=dict(l=10, r=10, t=36 if legend_top else 10, b=10),
               hovermode="x unified")
    if height:
        lay["height"] = height
    if legend_top:
        lay["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                             bgcolor="rgba(0,0,0,0)", font=dict(size=11))
    fig.update_layout(**lay)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)", zerolinecolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="rgba(0,0,0,0)")
    return fig

def restyle_mpl_dark(fig):
    """Recolor a matplotlib (SHAP) figure for the dark theme."""
    fig.patch.set_facecolor("#0e1117")
    for ax in fig.axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#9aa4b2")
        for s in ax.spines.values():
            s.set_color("#39414d")
        for item in (ax.title, ax.xaxis.label, ax.yaxis.label):
            item.set_color("#e6edf3")
        for t in ax.texts:
            try:
                r, g, b = mcolors.to_rgb(t.get_color())
                if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.25:  # near-black → light
                    t.set_color("#e6edf3")
            except Exception:
                pass
    return fig

# ------------------------------------------------------------------
# 2. CACHED LOADERS
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    m = xgb.XGBRegressor(); m.load_model(str(MODELS / f"final_xgb_transfer_model_{WINDOW_DAYS}d.json"))
    q10 = xgb.XGBRegressor(); q10.load_model(str(MODELS / f"final_xgb_q10_{WINDOW_DAYS}d.json"))
    q90 = xgb.XGBRegressor(); q90.load_model(str(MODELS / f"final_xgb_q90_{WINDOW_DAYS}d.json"))
    return m, q10, q90

@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(load_models()[0])

@st.cache_data
def load_players():
    df = pd.read_csv(DATA / f"demo_players_dataset_{WINDOW_DAYS}d.csv")
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[ALL_FEATURES] = df[ALL_FEATURES].astype(float)
    df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    return df.reset_index(drop=True)

@st.cache_data
def load_weekly():
    p = DATA / "player_weekly_sentiment.parquet"
    if not p.exists():
        return None
    wk = pd.read_parquet(p)
    wk["week"] = pd.to_datetime(wk["week"])
    return wk

@st.cache_data
def batch_predict(df):
    return np.expm1(load_models()[0].predict(df[ALL_FEATURES].astype(float)))

model, q10, q90 = load_models()
explainer = load_explainer()
players = load_players()
players["predicted_fee"] = batch_predict(players)
players["delta"] = players["predicted_fee"] - players["transfer_fee_adj"]
weekly = load_weekly()

def recompute_derived(d):
    d = d.copy()
    if "age_sq" in d.columns:
        d["age_sq"] = d["age_at_transfer"] ** 2
    if "has_lagged_mv" in d.columns:
        d["has_lagged_mv"] = (d["market_value_lagged_adj"] > 0).astype(float)
    if {"from_league_tier", "to_league_tier", "league_step"} <= set(d.columns):
        d["league_step"] = d["to_league_tier"] - d["from_league_tier"]
    return d

# headline metrics for hero chips (best-effort)
chip_r2 = chip_mae = ""
try:
    _m = pd.read_csv(MODELS / f"metrics_{WINDOW_DAYS}d.csv")
    _xg = _m[_m["Model"].str.contains("XGBoost")].iloc[0]
    chip_r2, chip_mae = f"R² {_xg['R2']:.3f}", f"MAE {eur(_xg['MAE'])}"
except Exception:
    pass

# ------------------------------------------------------------------
# 3. HERO
# ------------------------------------------------------------------
st.markdown(f"""
<div class="hero"><h1>⚽ Transfer Value Lab</h1>
<p>What is a footballer worth — and how much of that is hype? XGBoost fee predictions
explained with SHAP, powered by a custom RoBERTa sentiment pipeline over 61.6M Reddit comments.</p>
<div class="chips"><span>61.6M comments</span><span>Custom RoBERTa NLP</span>
<span>StatsBomb xG</span><span>SHAP explainability</span>
{f"<span>{chip_r2}</span>" if chip_r2 else ""}{f"<span>{chip_mae}</span>" if chip_mae else ""}</div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Player Lab", "🏆 Bargains & Overpays", "📈 Results & Windows", "🏗️ Architecture"])

# ==================================================================
# TAB 1 — PLAYER LAB
# ==================================================================
with tab1:
    def make_label(r):
        yr = r["transfer_date"].year if pd.notna(r["transfer_date"]) else "?"
        tag = " 🔮" if isinstance(yr, int) and yr >= 2025 else ""
        return f"{r['player_name']} ({yr}){tag} · #{r.name}"
    players["_label"] = players.apply(make_label, axis=1)
    sel = st.selectbox("🔎 Search a player (🔮 = future transfer, never seen in training)",
                       players["_label"].tolist())
    row = players.loc[players["_label"] == sel].iloc[0]

    # ---------------- What-if sidebar ----------------
    st.sidebar.header("🎛️ What-if lab")
    st.sidebar.caption("Override features to run counterfactuals — derived fields "
                       "(age², league step, coverage flags) recompute automatically.")
    edit = st.sidebar.toggle("Override features", value=False)
    ov = {}
    if edit:
        with st.sidebar.expander("📊 Bio & Financial", expanded=True):
            ov["age_at_transfer"] = st.slider("Age", 16.0, 40.0,
                                              float(row["age_at_transfer"]), 0.1)
            ov["market_value_lagged_adj"] = st.number_input(
                "Market value (€, pre-window)", 0, 250_000_000,
                int(row["market_value_lagged_adj"]), 250_000)
            ov["contract_years_remaining"] = st.slider(
                "Contract yrs left", 0.0, 6.0, float(row["contract_years_remaining"]), 0.25)
            ov["is_january_window"] = float(st.selectbox(
                "Window", ["Summer", "January"],
                index=int(row["is_january_window"])) == "January")
            pos_opts = ["Attack (baseline)"] + [c.replace("position_", "") for c in POSITION_COLS]
            cur_pos = next((c.replace("position_", "") for c in POSITION_COLS
                            if row.get(c, 0) == 1), "Attack (baseline)")
            new_pos = st.selectbox("Position", pos_opts, index=pos_opts.index(cur_pos))
            for c in POSITION_COLS:
                ov[c] = float(c == f"position_{new_pos}")
        with st.sidebar.expander("⚽ Performance (per 90)"):
            for c, lab in [("goals_per_90", "Goals/90"), ("assists_per_90", "Assists/90"),
                           ("sb_xg_per_90", "xG/90"), ("sb_xa_per_90", "xA/90")]:
                if c in ALL_FEATURES:
                    ov[c] = st.slider(lab, 0.0, 2.0, float(row[c]), 0.05)
        with st.sidebar.expander("📱 Reddit sentiment"):
            raw_vol = int(np.expm1(row["log_reddit_volume"]))
            new_vol = st.number_input("Comments in window", 0, 1_000_000, raw_vol, 100)
            ov["log_reddit_volume"] = float(np.log1p(new_vol))
            for c in ["avg_hype_prob", "avg_mkt_neg_prob", "avg_perf_pos_prob", "avg_perf_neg_prob"]:
                if c in ALL_FEATURES:
                    ov[c] = st.slider(FEATURE_LABELS[c], 0.0, 1.0, float(row[c]), 0.01)
            if "momentum_hype" in ALL_FEATURES:
                ov["momentum_hype"] = st.slider("Hype momentum", -0.5, 0.5,
                                                float(row["momentum_hype"]), 0.01)

    input_df = pd.DataFrame([{**{f: float(row[f]) for f in ALL_FEATURES}, **ov}],
                            columns=ALL_FEATURES)
    input_df = recompute_derived(input_df)

    # ---------------- Predictions ----------------
    fee_pred = float(np.expm1(model.predict(input_df)[0]))
    pi_lo = float(np.expm1(q10.predict(input_df)[0]))
    pi_hi = float(np.expm1(q90.predict(input_df)[0]))
    fee_actual = float(row["transfer_fee_adj"])
    delta = fee_pred - fee_actual

    no_nlp = input_df.copy()
    no_nlp[[f for f in NLP_FEATURES if f in no_nlp.columns]] = 0.0
    reddit_fx = fee_pred - float(np.expm1(model.predict(no_nlp)[0]))

    # ---------------- Player header card ----------------
    img_url = row.get("image_url") if isinstance(row.get("image_url"), str) else ""
    tm_url = row.get("url") if isinstance(row.get("url"), str) else ""
    route = ""
    if pd.notna(row.get("from_club_name")) and pd.notna(row.get("to_club_name")):
        route = f"{row['from_club_name']} → {row['to_club_name']} · "
    date_str = row["transfer_date"].strftime("%d %b %Y") if pd.notna(row["transfer_date"]) else "—"
    yr = row["transfer_date"].year if pd.notna(row["transfer_date"]) else None
    future_badge = ('<span class="badge future">🔮 out-of-time — never in training</span>'
                    if isinstance(yr, int) and yr >= 2025 else "")
    img_html =