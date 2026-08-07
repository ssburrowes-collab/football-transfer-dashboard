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
import xgboost as xgb

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
WINDOW_DAYS = 60

st.set_page_config(
    page_title="Football Transfer Fee Prediction Model",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

pio.templates.default = "plotly_dark"

plt.rcParams.update({
    "figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117",
    "savefig.facecolor": "#0e1117", "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3", "xtick.color": "#9aa4b2",
    "ytick.color": "#9aa4b2", "axes.edgecolor": "#39414d",
})

# Color Constants
BG = "rgba(0,0,0,0)"
GRID = "rgba(148,163,184,0.10)"
POS, NEG = "#34d399", "#f87171"
HYPE, PRAISE, CRIT = "#f5b942", "#34d399", "#f87171"
VIOLET = "#8b5cf6"
ACCENT = "#2ecc71"

st.markdown("""
<style>
.hero{padding:1.35rem 1.6rem;border-radius:14px;color:#e6edf3;margin-bottom:1.2rem;
background:#161b22;border:1px solid #232a35;border-left:4px solid #2ecc71;}
.hero .overline{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;
color:#2ecc71;font-weight:600}
.hero h1{margin:.25rem 0;font-size:1.75rem;font-weight:700;color:#f1f5f9}
.hero p{margin:.15rem 0 .7rem;color:#8b94a3;font-size:.92rem;max-width:60rem}
.chips span{display:inline-block;background:#1f2733;border:1px solid #2c3542;
border-radius:999px;padding:3px 12px;font-size:.74rem;margin:0 6px 6px 0;color:#aeb8c6}

.player-card{display:flex;flex-wrap:wrap;align-items:center;gap:18px;background:#161b22;
border:1px solid #232a35;border-radius:14px;padding:16px 20px;margin-bottom:14px}
.player-img{width:92px;height:92px;object-fit:cover;border-radius:12px;
border:1px solid #2c3542;background:#0e1117}
.player-avatar{width:92px;height:92px;border-radius:12px;background:#1f2733;
border:1px solid #2c3542;display:flex;align-items:center;justify-content:center;
font-size:1.7rem;font-weight:650;color:#8b94a3;letter-spacing:.04em}
.player-name{font-size:1.3rem;font-weight:700;color:#f1f5f9}
.player-meta{font-size:.85rem;color:#8b94a3;margin-top:2px}
.badge{display:inline-block;background:rgba(139,92,246,.12);
border:1px solid rgba(139,92,246,.45);border-radius:999px;padding:2px 10px;
font-size:.72rem;color:#c4b5fd;margin:6px 8px 0 0}
.tm-btn{margin-left:auto;text-decoration:none;background:#1f2733;color:#dbe7f3;
border:1px solid #2c3542;border-radius:10px;padding:8px 14px;font-size:.8rem;
white-space:nowrap;transition:all 0.2s ease;}
.tm-btn:hover{background:#27313f;color:#fff;border-color:#3b4758}

.kpi-grid{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:10px}
.kpi-card{flex:1;min-width:220px;background:#161b22;border:1px solid #232a35;border-radius:12px;
padding:14px 18px;border-top:3px solid var(--accent,#2ecc71)}
.kpi-title{font-size:.72rem;color:#8b94a3;text-transform:uppercase;letter-spacing:.07em}
.kpi-value{font-size:1.6rem;font-weight:700;color:#f1f5f9;margin-top:3px}
.kpi-sub{font-size:.78rem;color:#8b94a3;margin-top:2px}

.interval-note{background:rgba(245,185,66,.07);border:1px solid rgba(245,185,66,.25);
border-radius:10px;padding:10px 16px;font-size:.82rem;color:#c9b98a;margin:10px 0 14px}

.spotlight-box{background:rgba(5,150,105,0.08);border:1px solid rgba(5,150,105,0.3);
border-radius:10px;padding:10px 14px;margin-bottom:12px;font-size:0.82rem;color:#a7f3d0}

.section-h{font-size:1.1rem;font-weight:650;color:#e6edf3;margin:1.2rem 0 .5rem}
hr{border-color:#232a35;margin:1.5rem 0}

.stTabs [data-baseweb="tab-list"]{gap:6px}
.stTabs [data-baseweb="tab"]{background:#161b22;border:1px solid #232a35;
border-radius:10px 10px 0 0;padding:8px 18px}
.stTabs [aria-selected="true"]{background:#1f2733;border-bottom:2px solid #2ecc71}

.footer{margin-top:2.5rem;padding-top:1rem;border-top:1px solid #232a35;
color:#6b7484;font-size:.78rem;text-align:center}
</style>
""", unsafe_allow_html=True)

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
    "Financial": ["market_value_lagged_adj", "has_lagged_mv", "contract_years_remaining",
                  "is_january_window", "buying_club_avg_spend"],
    "Performance": ["goals_per_90", "assists_per_90", "sb_xg_per_90", "sb_xa_per_90",
                    "total_minutes_window", "has_statsbomb"],
    "Profile & League": ["age_at_transfer", "age_sq", "from_league_tier",
                         "to_league_tier", "league_step"] + POSITION_COLS,
    "Reddit NLP": NLP_FEATURES,
}

def eur(x):
    return f"€{x/1e6:,.1f}M" if abs(x) >= 1e6 else f"€{x/1e3:,.0f}k"

def initials(name):
    parts = [p for p in str(name).replace("-", " ").split() if p]
    return "".join(p[0] for p in parts[:2]).upper() or "?"

def style_dark(fig, height=None, legend_top=True):
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
                if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.25:
                    t.set_color("#e6edf3")
            except Exception:
                pass
    return fig

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
    path = DATA / f"demo_players_dataset_{WINDOW_DAYS}d.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
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
def load_interval_meta():
    p = MODELS / f"interval_meta_{WINDOW_DAYS}d.json"
    if p.exists():
        return json.load(open(p))
    return {"nominal": 0.80, "empirical_coverage": 0.668, "median_half_width_eur": None}

@st.cache_data
def batch_predict(df):
    return np.expm1(load_models()[0].predict(df[ALL_FEATURES].astype(float)))

model, q10, q90 = load_models()
explainer = load_explainer()
players = load_players()
weekly = load_weekly()
INTERVAL_META = load_interval_meta()

_missing = [c for c in ("from_club_name", "to_club_name") if c not in players.columns]
if _missing:
    _path = DATA / f"demo_players_dataset_{WINDOW_DAYS}d.csv"
    st.error(f"`{_path.name}` is missing columns: {_missing}. Please verify the file integrity.")
    st.stop()

players["predicted_fee"] = batch_predict(players)
players["delta"] = players["predicted_fee"] - players["transfer_fee_adj"]

NOMINAL_PCT = int(round(INTERVAL_META.get("nominal", 0.80) * 100))
EMPIRICAL = INTERVAL_META.get("empirical_coverage")
EMPIRICAL_TXT = (f"{EMPIRICAL:.1%}" if isinstance(EMPIRICAL, (int, float)) else "not measured")

def recompute_derived(d):
    d = d.copy()
    if "age_sq" in d.columns:
        d["age_sq"] = d["age_at_transfer"] ** 2
    if "has_lagged_mv" in d.columns:
        d["has_lagged_mv"] = (d["market_value_lagged_adj"] > 0).astype(float)
    if {"from_league_tier", "to_league_tier", "league_step"} <= set(d.columns):
        d["league_step"] = d["to_league_tier"] - d["from_league_tier"]
    return d

def pick(df, *cands):
    return next((c for c in cands if c in df.columns), None)

chip_r2 = chip_mae = ""
try:
    _m = pd.read_csv(MODELS / f"metrics_{WINDOW_DAYS}d.csv")
    _xg = _m[_m["Model"].str.contains("XGBoost")].iloc[0]
    chip_r2, chip_mae = f"R² {_xg['R2']:.3f}", f"MAE {eur(_xg['MAE'])}"
except Exception:
    pass

st.markdown(f"""
<div class="hero">
  <div class="overline">Football transfer valuation & narrative pricing</div>
  <h1>Football Transfer Fee Prediction Model</h1>
  <p>What is a footballer worth and how much of that is hype? 
     XGBoost fee predictions explained with SHAP, powered by a custom RoBERTa sentiment pipeline over 61.6M
     Reddit comments.</p>
  <div class="chips">
    <span>61.6M comments</span><span>Custom RoBERTa NLP</span>
    <span>StatsBomb xG</span><span>SHAP explainability</span>
    {f"<span>{chip_r2}</span>" if chip_r2 else ""}{f"<span>{chip_mae}</span>" if chip_mae else ""}
  </div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Player Lab", "Bargains & Overpays", "Results & Windows", "Architecture"])

# ==================================================================
# TAB 1 — PLAYER LAB
# ==================================================================
with tab1:
    def make_label(r):
        yr = r["transfer_date"].year if pd.notna(r["transfer_date"]) else "?"
        tag = " · out-of-sample" if isinstance(yr, int) and yr >= 2025 else ""
        return f"{r['player_name']} ({yr}){tag} · #{r.name}"
    
    players["_label"] = players.apply(make_label, axis=1)

    sel = st.selectbox(
        "Search a player — transfers from 2025 onward are out-of-sample and were never seen in training",
        players["_label"].tolist(),
        key="player_selectbox"
    )
    row = players.loc[players["_label"] == sel].iloc[0]

    # ---------------- What-if sidebar ----------------
    st.sidebar.header("What-if lab")
    st.sidebar.caption("Override features to run counterfactuals. Derived fields "
                       "(age², league step, coverage flags) recompute automatically.")
    edit = st.sidebar.toggle("Override features", value=False)
    ov = {}
    if edit:
        with st.sidebar.expander("Bio & financials", expanded=True):
            ov["age_at_transfer"] = st.slider(
                "Age", 16.0, 40.0, float(np.clip(row["age_at_transfer"], 16, 40)), 0.1)
            ov["market_value_lagged_adj"] = st.number_input(
                "Market value (€, pre-window)", 0, 250_000_000,
                int(np.clip(row["market_value_lagged_adj"], 0, 250_000_000)), 250_000)
            ov["contract_years_remaining"] = st.slider(
                "Contract yrs left", 0.0, 6.0,
                float(np.clip(row["contract_years_remaining"], 0, 6)), 0.25)
            ov["is_january_window"] = float(st.selectbox(
                "Window", ["Summer", "January"],
                index=int(row["is_january_window"])) == "January")
            pos_opts = ["Attack (baseline)"] + [c.replace("position_", "") for c in POSITION_COLS]
            cur_pos = next((c.replace("position_", "") for c in POSITION_COLS
                            if row.get(c, 0) == 1), "Attack (baseline)")
            new_pos = st.selectbox("Position", pos_opts, index=pos_opts.index(cur_pos))
            for c in POSITION_COLS:
                ov[c] = float(c == f"position_{new_pos}")
        with st.sidebar.expander("Performance (per 90)"):
            for c, lab in [("goals_per_90", "Goals/90"), ("assists_per_90", "Assists/90"),
                           ("sb_xg_per_90", "xG/90"), ("sb_xa_per_90", "xA/90")]:
                if c in ALL_FEATURES:
                    ov[c] = st.slider(lab, 0.0, 2.0, float(np.clip(row[c], 0, 2)), 0.05)
        with st.sidebar.expander("Reddit sentiment"):
            raw_vol = int(np.expm1(row["log_reddit_volume"]))
            new_vol = st.number_input("Comments in window", 0, 1_000_000, raw_vol, 100)
            ov["log_reddit_volume"] = float(np.log1p(new_vol))
            for c in ["avg_hype_prob", "avg_mkt_neg_prob", "avg_perf_pos_prob", "avg_perf_neg_prob"]:
                if c in ALL_FEATURES:
                    ov[c] = st.slider(FEATURE_LABELS[c], 0.0, 1.0,
                                      float(np.clip(row[c], 0, 1)), 0.01)
            if "momentum_hype" in ALL_FEATURES:
                ov["momentum_hype"] = st.slider(
                    "Hype momentum", -0.5, 0.5,
                    float(np.clip(row["momentum_hype"], -0.5, 0.5)), 0.01)

    input_df = pd.DataFrame([{**{f: float(row[f]) for f in ALL_FEATURES}, **ov}],
                            columns=ALL_FEATURES)
    input_df = recompute_derived(input_df)

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
    future_badge = ('<span class="badge">Out-of-sample — never in training</span>'
                    if isinstance(yr, int) and yr >= 2025 else "")
    
    img_html = f'<div class="player-avatar">{initials(row["player_name"])}</div>'
    if img_url and img_url.startswith("http"):
        img_html = f'<img class="player-img" src="{img_url}" alt="{row["player_name"]}">'
        
    tm_html = (f'<a class="tm-btn" href="{tm_url}" target="_blank">Transfermarkt profile ↗</a>'
               if tm_url else "")
    st.markdown(f"""
    <div class="player-card">{img_html}
      <div>
        <div class="player-name">{row['player_name']}</div>
        <div class="player-meta">{route}{date_str}</div>
        <div>{future_badge}</div>
      </div>{tm_html}
    </div>""", unsafe_allow_html=True)

    delta_color = POS if delta > 0 else NEG
    delta_word = ("model above market · potential bargain" if delta > 0
                  else "market above model · premium paid")
    reddit_color = POS if reddit_fx > 0 else NEG
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card" style="--accent:#2ecc71">
        <div class="kpi-title">Predicted fee</div>
        <div class="kpi-value">{eur(fee_pred)}</div>
        <div class="kpi-sub">{eur(pi_lo)} – {eur(pi_hi)}</div>
      </div>
      <div class="kpi-card" style="--accent:#60a5fa">
        <div class="kpi-title">Actual fee</div>
        <div class="kpi-value">{eur(fee_actual)}</div>
        <div class="kpi-sub">inflation-adjusted 2024 €</div>
      </div>
      <div class="kpi-card" style="--accent:{delta_color}">
        <div class="kpi-title">Model vs market</div>
        <div class="kpi-value" style="color:{delta_color}">{'+' if delta>0 else '−'}€{abs(delta)/1e6:.1f}M</div>
        <div class="kpi-sub">{delta_word}</div>
      </div>
      <div class="kpi-card" style="--accent:{VIOLET}">
        <div class="kpi-title">Reddit sentiment impact</div>
        <div class="kpi-value" style="color:{reddit_color}">{'+' if reddit_fx>0 else '−'}€{abs(reddit_fx)/1e6:.1f}M</div>
        <div class="kpi-sub">change if all Reddit sentiment is removed</div>
      </div>
    </div>""", unsafe_allow_html=True)
    reddit_dir = ("raises" if reddit_fx > 0 else "lowers")
    st.caption(
        f"Reddit sentiment impact: re-running the model with all 16 Reddit features removed "
        f"{reddit_dir} this prediction by €{abs(reddit_fx)/1e6:.1f}M "
        f"({eur(fee_pred)} with sentiment vs {eur(fee_pred - reddit_fx)} without)."
    )

    st.markdown(f"""
    <div class="interval-note"><b>About the interval:</b> the range under
    <i>Predicted fee</i> is a nominal {NOMINAL_PCT}% prediction interval from quantile XGBoost models. 
    Measured empirical coverage on the 2022+ test set: <b>{EMPIRICAL_TXT}</b>.</div>
    """, unsafe_allow_html=True)

    if edit:
        base_row = pd.DataFrame([{f: float(row[f]) for f in ALL_FEATURES}], columns=ALL_FEATURES)
        base_pred = float(np.expm1(model.predict(base_row)[0]))
        st.caption(f"What-if active — your overrides moved the prediction "
                   f"{'+' if fee_pred-base_pred>=0 else '−'}€{abs(fee_pred-base_pred)/1e6:.1f}M "
                   f"vs this player's real feature set.")

    sv = explainer.shap_values(input_df)[0]
    base = float(explainer.expected_value)
    total = base + float(np.sum(sv))

    st.markdown('<div class="section-h">Why the model prices this transfer</div>',
                unsafe_allow_html=True)
    
    # Graceful selector for segmented control / radio
    if hasattr(st, "segmented_control"):
        view = st.segmented_control("SHAP view", ["Grouped impact", "Waterfall", "Feature impacts"], default="Grouped impact")
    else:
        view = st.radio("SHAP view", ["Grouped impact", "Waterfall", "Feature impacts"], horizontal=True)

    if view == "Grouped impact":
        gnames, gfees = [], []
        for g, feats in GROUPS.items():
            idx = [ALL_FEATURES.index(f) for f in feats if f in ALL_FEATURES]
            if idx:
                contrib = float(np.sum(sv[idx]))
                gnames.append(g)
                gfees.append(float(np.expm1(total) - np.expm1(total - contrib)))
        order = np.argsort(np.abs(gfees))
        fig = go.Figure(go.Bar(
            y=[gnames[i] for i in order], x=[gfees[i] for i in order], orientation="h",
            marker_color=[POS if gfees[i] >= 0 else NEG for i in order],
            text=[f"{'+' if gfees[i]>=0 else '−'}€{abs(gfees[i])/1e6:.1f}M "
                  f"({gfees[i]/fee_pred*100:+.0f}%)" for i in order],
            textposition="outside", hovertemplate="%{y}: %{text}<extra></extra>"))
        style_dark(fig, height=260)
        fig.update_layout(margin=dict(l=10, r=100, t=10, b=10))
        fig.update_xaxes(title="Effect on predicted fee (€)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Feature families converted from log space to euros. "
                   "Green raises the predicted fee, red lowers it.")

    elif view == "Waterfall":
        try:
            exp = shap.Explanation(values=sv, base_values=base,
                                   data=input_df.iloc[0].values, feature_names=DISPLAY_NAMES)
            plt.figure(figsize=(13, 7))
            shap.plots.waterfall(exp, max_display=14, show=False)
            fig = restyle_mpl_dark(plt.gcf())
            st.pyplot(fig, use_container_width=True)
        finally:
            plt.close('all')

    else:  # Feature impacts
        n = min(14, len(sv))
        order = np.argsort(np.abs(sv))[-n:]
        vals = sv[order]
        names = [DISPLAY_NAMES[i] for i in order]
        eur_fx = [float(np.expm1(total) - np.expm1(total - v)) for v in vals]
        fig = go.Figure(go.Bar(
            y=names, x=vals, orientation="h",
            marker_color=[POS if v >= 0 else NEG for v in vals],
            text=[f"{'+' if e>=0 else '−'}€{abs(e)/1e6:.1f}M" for e in eur_fx],
            textposition="outside",
            hovertemplate="%{y}<br>SHAP %{x:+.3f} ≈ %{text}<extra></extra>"))
        style_dark(fig, height=480)
        fig.update_layout(margin=dict(l=10, r=80, t=10, b=10))
        fig.update_xaxes(title="Contribution to log-fee (SHAP)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Top features by impact. Labels show each feature's approximate euro effect.")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="section-h">Reddit sentiment timeline</div>',
                unsafe_allow_html=True)
    pid = int(row["player_id"])
    tdate = row["transfer_date"]

    if weekly is None:
        st.info("Weekly sentiment dataset not found in `data/`.")
    else:
        pw = weekly[weekly["player_id"] == pid].sort_values("week")
        if pw.empty:
            st.info("No Reddit timeline for this player (outside demo set or zero comments).")
        else:
            fig = go.Figure()
            vol_col = pick(pw, "comment_count", "n_comments", "volume")
            if vol_col:
                fig.add_trace(go.Bar(
                    x=pw["week"], y=pw[vol_col], name="Comments",
                    marker_color="rgba(255,80,80,0.45)", yaxis="y2",
                    hovertemplate="%{x|%b %Y}: %{y} comments<extra></extra>"))
            for cands, name, color in [
                    (("avg_hype", "hype", "avg_hype_prob"), "Hype", HYPE),
                    (("avg_praise", "praise", "avg_perf_pos_prob"), "Praise", PRAISE),
                    (("avg_criticism", "criticism", "avg_perf_neg_prob"), "Criticism", CRIT)]:
                c = pick(pw, *cands)
                if c:
                    fig.add_trace(go.Scatter(
                        x=pw["week"], y=pw[c], name=name, mode="lines",
                        line=dict(color=color, width=2.2),
                        hovertemplate="%{x|%b %Y}: %{y:.2f}<extra>" + name + "</extra>"))

            if pd.notna(tdate):
                w0 = tdate - pd.Timedelta(days=WINDOW_DAYS)
                fig.add_vrect(x0=w0, x1=tdate,
                              fillcolor="rgba(139,92,246,0.22)",
                              line=dict(color=VIOLET, width=1.5, dash="dash"),
                              layer="below")
                fig.add_vline(x=tdate, line=dict(color="#e6edf3", width=1, dash="dot"))
                fig.add_annotation(x=w0 + (tdate - w0) / 2, y=1.05, yref="paper",
                                   text=f"{WINDOW_DAYS}-day pre-transfer window",
                                   showarrow=False, font=dict(size=11, color="#c4b5fd"))
                fig.add_annotation(x=tdate, y=0.97, yref="paper", text="transfer day",
                                   showarrow=False, xanchor="left", xshift=6,
                                   font=dict(size=10, color="#e6edf3"))
                pad = pd.Timedelta(days=120)
                fig.update_xaxes(range=[min(pw["week"].min(), w0) - pad,
                                        max(pw["week"].max(), tdate) + pad])

            fig.update_layout(
                barmode="overlay",
                yaxis=dict(title="Avg probability", range=[0, 1.05]),
                yaxis2=dict(title="Comments", overlaying="y", side="right",
                            showgrid=False, rangemode="tozero"))
            style_dark(fig, height=440)
            st.plotly_chart(fig, use_container_width=True)
            n_comments = int(np.expm1(row["log_reddit_volume"]))
            st.caption(f"Shaded band = the {WINDOW_DAYS}-day window the model sees · "
                       f"{n_comments:,} comments in this player's window · "
                       "weekly RoBERTa class probabilities.")

    # ---------------- Export CSV ----------------
    st.markdown("<hr>", unsafe_allow_html=True)
    out = pd.DataFrame({"feature": DISPLAY_NAMES,
                        "value": input_df.iloc[0].values, "shap": sv})
    out["abs"] = out["shap"].abs()
    out = out.sort_values("abs", ascending=False).drop(columns="abs")
    st.download_button("Download current player analysis (CSV)",
                       out.to_csv(index=False).encode(),
                       file_name=f"{row['player_name'].replace(' ', '_')}_shap.csv",
                       mime="text/csv")

# ==================================================================
# TAB 2 — BARGAINS & OVERPAYS
# ==================================================================
with tab2:
    st.markdown('<div class="section-h">Where the model disagrees with the market</div>',
                unsafe_allow_html=True)
    st.caption("Gap = model prediction − actual fee. Positive = model rates the player "
               "above what was paid (potential bargain); negative = premium paid.")

    lb = players[["player_name", "from_club_name", "to_club_name", "transfer_date",
                  "transfer_fee_adj", "predicted_fee", "delta"]].copy()
    lb["year"] = lb["transfer_date"].dt.year
    lb = lb.drop(columns="transfer_date").rename(columns={
        "player_name": "Player", "from_club_name": "From", "to_club_name": "To",
        "transfer_fee_adj": "Actual fee", "predicted_fee": "Model value", "delta": "Gap"})
    fmt = {"Actual fee": lambda v: eur(v), "Model value": lambda v: eur(v),
           "Gap": lambda v: f"{'+' if v >= 0 else '−'}€{abs(v)/1e6:.1f}M"}

    st.markdown("**Biggest bargains**")
    st.dataframe(lb.nlargest(12, "Gap").style.format(fmt), use_container_width=True, hide_index=True)
    st.markdown("**Biggest overpays**")
    st.dataframe(lb.nsmallest(12, "Gap").style.format(fmt), use_container_width=True, hide_index=True)

# ==================================================================
# TAB 3 — RESULTS & WINDOWS
# ==================================================================
with tab3:
    st.markdown('<div class="section-h">Model comparison across temporal windows</div>',
                unsafe_allow_html=True)
    mets = []
    for p in sorted(MODELS.glob("metrics_*d.csv")):
        tmp = pd.read_csv(p)
        tmp["window"] = p.stem.split("_")[1]
        mets.append(tmp)
    if mets:
        allm = pd.concat(mets)
        pick_m = st.radio("Metric", ["R2", "RMSLE", "MAE", "RMSE"], horizontal=True)
        piv = allm.pivot_table(index="Model", columns="window", values=pick_m)
        st.dataframe(piv.style.format("{:.3f}" if pick_m in ("R2", "RMSLE") else "{:,.0f}"),
                     use_container_width=True)
    else:
        st.info("No metrics_*.csv files found in `models/`.")

    ab = [json.load(open(p)) for p in sorted(MODELS.glob("ablation_*d.json"))]
    if ab:
        st.markdown('<div class="section-h">Does Reddit sentiment help? (ablation)</div>',
                    unsafe_allow_html=True)
        abdf = pd.DataFrame(ab).sort_values("window_days")
        d = abdf["r2_nlp"] - abdf["r2_stats"]
        fig = go.Figure(go.Bar(
            x=abdf["window_days"].astype(str) + "d", y=d,
            marker_color=[POS if v >= 0 else NEG for v in d],
            text=[f"{v:+.3f}" for v in d], textposition="outside"))
        style_dark(fig, height=330)
        fig.update_xaxes(title="Pre-transfer window")
        fig.update_yaxes(title="Δ R² (Stats + NLP − Stats only)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Identical XGBoost configuration, same temporal split — the only difference "
                   "is the 16 Reddit features. Positive bars = sentiment improves out-of-sample "
                   "fit. The 365-day window is the one exception: sentiment that old is noise, "
                   "which is how the 60-day production window was chosen.")

    img = MODELS / f"shap_summary_{WINDOW_DAYS}d.png"
    if img.exists():
        st.markdown('<div class="section-h">Global feature importance (SHAP)</div>',
                    unsafe_allow_html=True)
        st.markdown('<div style="background:#fff;border-radius:12px;padding:10px">',
                    unsafe_allow_html=True)
        st.image(str(img), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# TAB 4 — ARCHITECTURE
# ==================================================================
with tab4:
    st.markdown('<div class="section-h">How the system works</div>', unsafe_allow_html=True)
    st.markdown("""
1. **Data lake (PySpark)** — ~1–2 TB of raw Reddit archives filtered to **61.6M player-linked
   English comments**; Transfermarkt transfers, appearances & valuations; StatsBomb event data.
2. **Custom NLP** — Word2Vec trained from scratch into a LLM-annotated 4-class sentiment taxonomy
   into fine-tuned **RoBERTa** into probabilities for all 61.6M comments.
3. **Feature engineering** — 60-day pre-transfer windows: comment volume, upvote-weighted
   sentiment, volatility, 14-day momentum plus structured performance & financial features.
   Only pre-transfer information is used (valuations lagged ≥180 days, contracts capped at the
   5-year regulatory maximum).
4. **Modelling** — XGBoost with monotonicity constraints, strictly temporal train/test split,
   quantile models for prediction intervals (reported with measured coverage), SHAP for
   explanations.
5. **This app** — serves the trained model artifacts directly. The demo CSV carries features
   and metadata only; every prediction, interval, counterfactual and SHAP value you see is
   computed live in the browser session.
    """)

st.markdown('<div class="footer">Predictions inflation-adjusted to 2024 € · Sentiment: custom '
            'RoBERTa over 61.6M Reddit comments · Built with Streamlit, SHAP and Plotly · '
            'COMP6830 Capstone — Shemar Burrowes</div>', unsafe_allow_html=True)