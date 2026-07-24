import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # required on Streamlit Cloud (headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb

# ------------------------------------------------------------------
# 0. PATHS & CONSTANTS
# ------------------------------------------------------------------
ROOT = Path(__file__).parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
WINDOW_DAYS = 60

st.set_page_config(page_title="Transfer Value Lab", page_icon="⚽",
                   layout="wide", initial_sidebar_state="expanded")

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

# ------------------------------------------------------------------
# 1. CACHED LOADERS
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
    model = load_models()[0]
    return np.expm1(model.predict(df[ALL_FEATURES].astype(float)))

model, q10, q90 = load_models()
explainer = load_explainer()
players = load_players()
players["predicted_fee"] = batch_predict(players)
players["delta"] = players["predicted_fee"] - players["transfer_fee_adj"]
weekly = load_weekly()

def recompute_derived(d):
    """Keep engineered features consistent after what-if edits."""
    d = d.copy()
    if "age_sq" in d.columns:
        d["age_sq"] = d["age_at_transfer"] ** 2
    if "has_lagged_mv" in d.columns:
        d["has_lagged_mv"] = (d["market_value_lagged_adj"] > 0).astype(float)
    if {"from_league_tier", "to_league_tier", "league_step"} <= set(d.columns):
        d["league_step"] = d["to_league_tier"] - d["from_league_tier"]
    return d

# ------------------------------------------------------------------
# 2. HERO
# ------------------------------------------------------------------
st.markdown("""
<style>
.hero{padding:1.1rem 1.4rem;border-radius:14px;color:#fff;margin-bottom:1rem;
background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);}
.hero h1{margin:0;font-size:1.8rem}.hero p{margin:.2rem 0 0;opacity:.85}
</style>
<div class="hero"><h1>⚽ Transfer Value Lab</h1>
<p>XGBoost · RoBERTa sentiment over 65M Reddit comments · StatsBomb xG · SHAP —
COMP6830 Capstone | Shemar Burrowes</p></div>""", unsafe_allow_html=True)

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
    st.sidebar.header("🎛️ What-if mode")
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

    no_nlp = input_df.copy()
    no_nlp[[f for f in NLP_FEATURES if f in no_nlp.columns]] = 0.0
    reddit_fx = fee_pred - float(np.expm1(model.predict(no_nlp)[0]))

    head_l, head_r = st.columns([1, 3])
    with head_l:
        if isinstance(row.get("image_url"), str) and row["image_url"].startswith("http"):
            st.image(row["image_url"], width=130)
        if isinstance(row.get("url"), str) and row["url"].startswith("http"):
            st.link_button("Transfermarkt profile ↗", row["url"])
    with head_r:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted fee", eur(fee_pred))
        c1.caption(f"80% interval: {eur(pi_lo)} – {eur(pi_hi)}")
        c2.metric("Actual fee", eur(fee_actual))
        delta = fee_pred - fee_actual
        c3.metric("Model vs market", eur(abs(delta)),
                  delta=("bargain for buyer" if delta > 0 else "premium paid"))
        c4.metric("📱 Reddit effect",
                  f"{'+' if reddit_fx >= 0 else '−'}{eur(abs(reddit_fx))}",
                  help="Predicted fee with Reddit sentiment vs. all sentiment features zeroed.")
    st.caption(f"{WINDOW_DAYS}-day feature window · fees in 2024 € · "
               "values are log-space calibrated; typical error on the temporal test set is in Tab 3.")

    st.markdown("---")
    shap_l, shap_r = st.columns([3, 2])

    exp = explainer(input_df)
    exp.feature_names = DISPLAY_NAMES
    base_val = float(np.atleast_1d(explainer.expected_value)[0])

    with shap_l:
        st.subheader(f"Why {row['player_name']} costs what he costs")
        view = (st.segmented_control("SHAP view",
                                     ["Waterfall", "Force (interactive)", "Grouped impact"],
                                     default="Waterfall")
                if hasattr(st, "segmented_control")
                else st.radio("SHAP view",
                              ["Waterfall", "Force (interactive)", "Grouped impact"],
                              horizontal=True))
        if view == "Waterfall":
            shap.plots.waterfall(exp[0], max_display=12, show=False)
            fig = plt.gcf(); fig.set_size_inches(9.5, 5)
            st.pyplot(fig); plt.close("all")
            st.caption("SHAP values are in log-fee space: red pushes the fee up, blue down.")
        elif view == "Force (interactive)":
            fp = shap.force_plot(base_val, exp.values[0],
                                 pd.Series(exp.data[0], index=DISPLAY_NAMES),
                                 matplotlib=False)
            components.html(f"<head>{shap.getjs()}</head><body>{fp.html()}</body>",
                            height=220, scrolling=True)
        else:
            sv = exp.values[0]
            contrib = {g: float(sv[[ALL_FEATURES.index(f) for f in fs
                                    if f in ALL_FEATURES]].sum())
                       for g, fs in GROUPS.items()}
            fig = go.Figure(go.Bar(
                x=list(contrib.values()), y=list(contrib.keys()), orientation="h",
                marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in contrib.values()]))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="Contribution to log-fee")
            st.plotly_chart(fig, use_container_width=True)

    with shap_r:
        st.subheader("📱 Reddit sentiment timeline")
        if weekly is not None:
            wk = weekly[weekly["player_id"] == int(row["player_id"])].sort_values("week")
        else:
            wk = pd.DataFrame()
        if len(wk) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=wk["week"], y=wk["volume"], name="Comments",
                                 marker_color="rgba(150,150,150,.3)", yaxis="y2"))
            fig.add_trace(go.Scatter(x=wk["week"], y=wk["hype"], name="Hype",
                                     line=dict(color="#f1c40f")))
            fig.add_trace(go.Scatter(x=wk["week"], y=wk["perf_pos"], name="Praise",
                                     line=dict(color="#2ecc71")))
            fig.add_trace(go.Scatter(x=wk["week"], y=wk["perf_neg"], name="Criticism",
                                     line=dict(color="#e74c3c")))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h"),
                              yaxis=dict(title="Avg probability", range=[0, 1]),
                              yaxis2=dict(overlaying="y", side="right",
                                          showgrid=False, title="Volume"))
            if pd.notna(row["transfer_date"]):
                t = row["transfer_date"]
                fig.add_vrect(x0=t - pd.Timedelta(days=WINDOW_DAYS), x1=t,
                              fillcolor="blue", opacity=0.08, line_width=0)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Shaded band = {WINDOW_DAYS}-day window before the transfer.")
        else:
            st.info("No Reddit activity found for this player.")

# ==================================================================
# TAB 2 — LEADERBOARD
# ==================================================================
with tab2:
    st.subheader("🏆 Where the model disagrees with the market most")
    st.caption("Predicted − Actual. Positive = model thinks the buying club got a bargain.")
    show = ["player_name", "transfer_date", "transfer_fee_adj", "predicted_fee", "delta"]
    fmt = {"transfer_fee_adj": "€{:,.0f}", "predicted_fee": "€{:,.0f}",
           "delta": "€{:+,.0f}", "transfer_date": lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) else "—"}
    c1, c2 = st.columns(2)
    c1.markdown("**💎 Biggest bargains (model > market)**")
    c1.dataframe(players.nlargest(15, "delta")[show].style.format(fmt),
                 use_container_width=True, hide_index=True)
    c2.markdown("**💸 Biggest overpays (market > model)**")
    c2.dataframe(players.nsmallest(15, "delta")[show].style.format(fmt),
                 use_container_width=True, hide_index=True)

# ==================================================================
# TAB 3 — RESULTS & WINDOWS
# ==================================================================
with tab3:
    st.subheader("Model comparison & temporal window sensitivity")

    mfiles = sorted(MODELS.glob("metrics_*.csv"))
    afiles = sorted(MODELS.glob("ablation_*.json"))
    if mfiles:
        metrics_all = pd.concat([pd.read_csv(f) for f in mfiles])
        st.markdown("**All models × windows (R², log space)**")
        st.dataframe(metrics_all.pivot_table(index="Model", columns="window_days",
                                             values="R2").round(4),
                     use_container_width=True)
        xg_rows = metrics_all[metrics_all["Model"].str.contains("XGBoost")] \
                      .sort_values("window_days")
        try:
            mae60 = metrics_all[(metrics_all["Model"].str.contains("XGBoost")) &
                                (metrics_all["window_days"] == WINDOW_DAYS)]["MAE"].iloc[0]
            st.caption(f"XGBoost (60d) typical error: ±{eur(mae60)} MAE on the temporal test set.")
        except Exception:
            pass
    else:
        xg_rows = None
        st.warning("No metrics_*.csv found in models/.")

    if afiles:
        abl = pd.DataFrame([json.load(open(f)) for f in afiles]).sort_values("window_days")
        abl["Δ R²"] = abl["r2_nlp"] - abl["r2_stats"]
        abl["Δ RMSE (€M)"] = (abl["rmse_nlp"] - abl["rmse_stats"]) / 1e6
        fig = go.Figure()
        if xg_rows is not None:
            fig.add_trace(go.Scatter(x=xg_rows["window_days"], y=xg_rows["R2"],
                                     name="R² (Stats+NLP)", mode="lines+markers"))
        fig.add_trace(go.Bar(x=abl["window_days"], y=abl["Δ R²"], name="NLP ΔR²",
                             marker_color="#2ecc71", opacity=0.55, yaxis="y2"))
        fig.update_layout(xaxis=dict(title="Window (days)", type="log"),
                          yaxis=dict(title="R²"),
                          yaxis2=dict(overlaying="y", side="right", showgrid=False),
                          height=340, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(abl[["window_days", "r2_stats", "r2_nlp", "Δ R²",
                          "Δ RMSE (€M)"]].round(4),
                     use_container_width=True, hide_index=True)
        st.info("Windows where NLP improves both R² and RMSE are the defensible ones; "
                "Δ values of this size are reported with bootstrap confidence intervals "
                "in the notebook.")

    st.markdown("**Global SHAP summary**")
    shap_imgs = sorted(MODELS.glob("shap_summary_*d.png"))
    if shap_imgs:
        choice = st.selectbox("Window", [p.stem.replace("shap_summary_", "")
                                         for p in shap_imgs],
                              index=[p.stem for p in shap_imgs].index(
                                  f"shap_summary_{WINDOW_DAYS}d")
                              if f"shap_summary_{WINDOW_DAYS}d" in
                              [p.stem for p in shap_imgs] else 0)
        st.image(str(MODELS / f"shap_summary_{choice}.png"),
                 caption=f"Global feature importance ({choice} window)")
    else:
        st.warning("No shap_summary_*.png found in models/.")

# ==================================================================
# TAB 4 — ARCHITECTURE
# ==================================================================
with tab4:
    st.markdown("""
    1. **ETL (PySpark):** ~1–2TB of Reddit `.zst` archives → 61.6M-row English,
       player-linked football corpus; 12.9GB StatsBomb JSON flattened (12.1M events).
    2. **Entity resolution:** `rapidfuzz` fuzzy matching (≥90% after audit) + custom
       dictionary NER (3,092 name variants incl. nicknames) linking comments to players.
    3. **NLP:** Word2Vec/Doc2Vec trained from scratch; LLM (GLM) synthetic annotation
       (3,539 clean rows); fine-tuned RoBERTa with class-weighted loss (Macro F1 = 0.51),
       temperature-calibrated before aggregation.
    4. **Massive inference:** FP16 GPU inference over 65M rows; softmax probabilities
       aggregated per player per transfer window (volume, means, upvote-weighted means,
       volatility, 14-day momentum).
    5. **Regression:** XGBoost (early-stopped, monotonicity-constrained) predicting
       inflation-adjusted fees (2024 €) with a temporal train/test split, lagged market
       values, and quantile models for 80% prediction intervals. SHAP for global +
       per-player explainability.
    """)

st.markdown("---")
st.download_button(
    "⬇️ Download current player analysis",
    input_df.assign(player_name=row["player_name"], predicted_fee=fee_pred,
                    pi_lo=pi_lo, pi_hi=pi_hi).to_csv(index=False),
    file_name=f"analysis_{row['player_name'].replace(' ', '_')}.csv", mime="text/csv")
st.caption("Predictions inflation-adjusted to 2024 € · Sentiment: custom RoBERTa over "
           "65M Reddit comments · Built with Streamlit + SHAP + Plotly")