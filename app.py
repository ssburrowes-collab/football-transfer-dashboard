import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION & MODEL LOADING
# ==========================================
st.set_page_config(page_title="Football Transfer Predictor", layout="wide")

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("final_xgb_transfer_model_60d.json")
    return model

@st.cache_data
def load_demo_data():
    df = pd.read_csv("demo_players_dataset_60d.csv")
    # Ensure position columns exist even if missing in top 50
    for col in ['position_Defender', 'position_Goalkeeper', 'position_Midfield', 'position_Unknown']:
        if col not in df.columns:
            df[col] = 0
    return df

model = load_model()
demo_df = load_demo_data()

# Define the exact feature list used in training (60-day window)
base_features = [
    'age_at_transfer', 'market_value_adj', 'goals_per_90', 'assists_per_90', 
    'sb_xg_per_90', 'sb_xa_per_90', 'contract_years_remaining', 'is_january_window',
    'position_Defender', 'position_Goalkeeper', 'position_Midfield'
]
nlp_features = [
    'log_reddit_volume', 'avg_hype_prob', 'avg_mkt_neg_prob', 
    'avg_perf_pos_prob', 'avg_perf_neg_prob', 'std_hype_prob', 'std_perf_neg_prob'
]
all_features = base_features + nlp_features

# ==========================================
# 2. DASHBOARD UI (MULTI-PAGE)
# ==========================================
st.title("⚽ Football Transfer Fee Predictor & Explainer")
st.markdown("##### COMP6830 Capstone Project | Shemar Burrowes")

tab1, tab2, tab3 = st.tabs(["🎯 Prediction Dashboard", "📊 Model Results", "🏗️ Project Architecture"])

# ------------------------------------------
# TAB 1: PREDICTION DASHBOARD
# ------------------------------------------
with tab1:
    st.markdown("This dashboard utilises an XGBoost Regressor trained on StatsBomb tactical data, Transfermarkt financials, and a 65M-row custom RoBERTa NLP sentiment pipeline (60-day window) to predict inflation-adjusted transfer fees.")

    st.sidebar.title("🔧 Player Selection & Inputs")

    # --- Player Dropdown ---
    player_names = demo_df['player_name'].tolist()
    selected_name = st.sidebar.selectbox("Select an actual player to load:", player_names)

    # Extract the selected player's real data
    player_row = demo_df[demo_df['player_name'] == selected_name].iloc[0]
    actual_fee = player_row['transfer_fee_adj']

    def get_val(key, default=0.0):
        return float(player_row.get(key, default))

    # --- Sidebar Inputs (Pre-populated with real player data) ---
    with st.sidebar.expander("📊 Biographical & Financial", expanded=True):
        age = st.slider("Age at Transfer", 16.0, 40.0, get_val('age_at_transfer', 22.0), 0.1)
        market_val = st.number_input("Market Value (€, Adjusted)", min_value=0, max_value=200000000, value=int(get_val('market_value_adj', 40000000)), step=1000000)
        contract_years = st.slider("Contract Years Remaining", 0.5, 6.0, get_val('contract_years_remaining', 3.0), 0.5)
        is_jan = st.selectbox("Transfer Window", ["Summer (0)", "January (1)"], index=int(get_val('is_january_window', 0)))
        is_jan = int(is_jan.startswith("January"))
        
        pos_str = "Attack (Baseline)"
        if get_val('position_Defender') == 1: pos_str = "Defender"
        elif get_val('position_Goalkeeper') == 1: pos_str = "Goalkeeper"
        elif get_val('position_Midfield') == 1: pos_str = "Midfield"
        elif get_val('position_Unknown') == 1: pos_str = "Unknown"
        
        position = st.selectbox("Position", ["Attack (Baseline)", "Defender", "Goalkeeper", "Midfield", "Unknown"], index=["Attack (Baseline)", "Defender", "Goalkeeper", "Midfield", "Unknown"].index(pos_str))
        pos_def = 1 if position == "Defender" else 0
        pos_gk = 1 if position == "Goalkeeper" else 0
        pos_mid = 1 if position == "Midfield" else 0
        pos_unk = 1 if position == "Unknown" else 0

    with st.sidebar.expander("⚽ Tactical Stats (Per 90)", expanded=True):
        goals_90 = st.slider("Goals per 90", 0.0, 2.0, get_val('goals_per_90', 0.4), 0.05)
        assists_90 = st.slider("Assists per 90", 0.0, 2.0, get_val('assists_per_90', 0.2), 0.05)
        xg_90 = st.slider("StatsBomb xG per 90", 0.0, 1.5, get_val('sb_xg_per_90', 0.3), 0.05)
        xa_90 = st.slider("StatsBomb xA per 90", 0.0, 1.5, get_val('sb_xa_per_90', 0.2), 0.05)

    with st.sidebar.expander("📱 NLP Sentiment (Reddit 60d Window)", expanded=True):
        raw_vol = np.expm1(get_val('log_reddit_volume', 10.0))
        reddit_vol = st.number_input("Reddit Comment Volume (60 days)", min_value=0, max_value=100000, value=int(raw_vol), step=500)
        log_vol = np.log1p(reddit_vol)
        
        hype_prob = st.slider("Avg Market Hype Prob", 0.0, 1.0, get_val('avg_hype_prob', 0.2), 0.01)
        mkt_neg = st.slider("Avg Market Negative Prob", 0.0, 1.0, get_val('avg_mkt_neg_prob', 0.1), 0.01)
        perf_pos = st.slider("Avg Performance Positive Prob", 0.0, 1.0, get_val('avg_perf_pos_prob', 0.4), 0.01)
        perf_neg = st.slider("Avg Performance Negative Prob", 0.0, 1.0, get_val('avg_perf_neg_prob', 0.3), 0.01)
        std_hype = st.slider("Std Dev (Volatility) of Hype", 0.0, 0.5, get_val('std_hype_prob', 0.15), 0.01)

    st.markdown("---")
    st.header(f"🎯 Analysis: {selected_name}")

    # Construct the input array
    input_data = pd.DataFrame([[
        age, market_val, goals_90, assists_90, xg_90, xa_90, contract_years, is_jan,
        pos_def, pos_gk, pos_mid, pos_unk,
        log_vol, hype_prob, mkt_neg, perf_pos, perf_neg, std_hype, get_val('std_perf_neg_prob', 0.1)
    ]], columns=all_features)

    for c in input_data.columns:
        if input_data[c].dtype == bool:
            input_data[c] = input_data[c].astype(int)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Fee Comparison")
        log_pred = model.predict(input_data)[0]
        eur_pred = np.expm1(log_pred)
        
        delta = eur_pred - actual_fee
        delta_str = f"€{delta:,.0f} Overvalued" if delta > 0 else f"€{abs(delta):,.0f} Undervalued"
        
        st.metric(label="Predicted Fee (2024 €)", value=f"€{eur_pred:,.0f}", delta=delta_str)
        st.metric(label="Actual Fee (2024 €)", value=f"€{actual_fee:,.0f}")
        
        st.markdown("""
        * **Overvalued:** The model thinks the player's stats & hype should have cost *more* than they actually did.
        * **Undervalued:** The model thinks the player was a bargain relative to their stats & hype.
        """)

    with col2:
        st.subheader(f"SHAP Explainer: Why {selected_name} costs what he costs")
        st.write("This player-specific waterfall plot shows exactly how each feature pushed the prediction higher (red) or lower (blue) compared to the average player in the dataset.")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], input_data.iloc[0], max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(fig, bbox_inches='tight')

# ------------------------------------------
# TAB 2: MODEL RESULTS
# ------------------------------------------
with tab2:
    st.header("Model Performance & Ablation Study (60-Day Window)")
    st.markdown("""
    Below are the results of the Feature Ablation Study.
    
    **Feature Ablation Conclusion:**
    The baseline model using *only* structured stats yielded a weak R² of 0.0838. 
    When the 65M-row NLP sentiment features were introduced, the model's R² more than doubled to **0.1999**, 
    proving that social media sentiment provides statistically significant predictive value.
    """)
    
    st.subheader("Global SHAP Summary Plot (60-Day Window)")
    st.markdown("This plot shows the global feature importance across the entire test set. Features are ranked by their overall impact on the model's output.")
    
    try:
        st.image("shap_summary_60d.png", caption="Global SHAP Feature Importance (60-Day Window)")
    except:
        st.warning("Upload `shap_summary_60d.png` to the GitHub repo to display this chart.")

# ------------------------------------------
# TAB 3: PROJECT ARCHITECTURE
# ------------------------------------------
with tab3:
    st.header("End-to-End Architecture")
    st.markdown("""
    This project follows a rigorous Big Data ETL and Machine Learning pipeline:
    
    1. **Data Ingestion (PySpark):** Streamed and decompressed ~1-2TB of raw Reddit `.zst` archives, filtering 240M raw records down to a 65M-row English football corpus. Flattened 12.9GB of StatsBomb JSON event data.
    2. **Entity Resolution:** Bridged Transfermarkt and StatsBomb databases using `rapidfuzz` fuzzy matching (≥90% confidence) and custom Named Entity Recognition (NER) to link 65M Reddit comments to specific players.
    3. **Custom NLP & LLM Annotation:** Trained Word2Vec/Doc2Vec from scratch. Utilised a Large Language Model (GLM-4) for zero-shot synthetic data annotation to create a 3,500-row labelled dataset.
    4. **Deep Learning (RoBERTa):** Fine-tuned a RoBERTa Transformer (Macro F1: 0.51) with a custom weighted loss function to handle class imbalance.
    5. **Massive Inference & Regression:** Deployed the RoBERTa model across all 65M rows using PyTorch and Hugging Face Datasets, extracting softmax probabilities. Aggregated these to the player level and fed them into an XGBoost Regressor to predict the final transfer fee.
    """)

st.markdown("---")
st.caption("Note: This model utilises an NLP pipeline trained on 65M Reddit comments via a custom RoBERTa sentiment classifier. Predictions are inflation-adjusted to 2024 Euros.")