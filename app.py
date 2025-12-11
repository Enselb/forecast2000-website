import streamlit as st
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
import os
import plotly.graph_objects as go
import google.auth

key_path = os.path.join("secrets", "forecast2000-ec89eb4db84e.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
credentials, project_id = google.auth.default()



# ---------------------------
# 1. CONFIGURATION & DESIGN NEON
# ---------------------------
st.set_page_config(page_title="Forecast 2000", page_icon="✨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #E8DBDA;
    color: #212529;
}

h1 {
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    color: #FFFFFF;
    text-shadow: 0 0 5px #00D2FF, 0 0 10px #00D2FF, 0 0 20px #FF00FF, 0 0 30px #FF00FF;
    letter-spacing: 1px;
}

.glass {
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(0, 210, 255, 0.3);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
}

.stButton>button {
    background: linear-gradient(45deg, #00D2FF, #FF00FF);
    color: white;
    border-radius: 8px;
    padding: 10px 24px;
    border: none;
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    text-transform: uppercase;
}

section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid rgba(255, 0, 255, 0.2); }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label { color: #E0E0E0; }
.stSelectbox > div > div, .stDateInput > div > div, .stSlider > div > div { background-color: #1A1A1A; color: white; border-color: #333333; }
.muted { color: #666666; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

API_URL = "https://forecast-49942818362.europe-west1.run.app/predict"
SPLIT_DATE = pd.to_datetime('2016-04-24')

# fonctions de chargement des données
@st.cache_data
def load_historical_data():
    gcs_path = 'gs://forecast_2000_raw_data/df_from_2016.parquet'
    df = pd.read_parquet(gcs_path)
    #df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_forecast_data():
    response = requests.get(API_URL)
    df = response.json().get('predictions', [])
    return df

@st.cache_data
def load_X_test_data():
    gcs_path = 'gs://forecast_2000_raw_data/X_test_20251210.csv'
    df = pd.read_csv(gcs_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

# chargement des données
df_history = load_historical_data()
df_forecast = load_forecast_data()
X_test = load_X_test_data()
# df_forecast = df_forecast.rename(columns={0: 'pred_sales'})

df_forecast = pd.DataFrame(df_forecast)
df_pred = pd.concat([X_test, df_forecast], axis=1)
df_pred = df_pred.rename(columns={0: 'pred_sales'})
df_pred.set_index('date', inplace=True)


# ---------------------------
# INTERFACE
# ---------------------------
col_titre, col_logo = st.columns([4, 1])
with col_titre:
    st.markdown("<h1>FORECAST 2000 — Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted' style='color: #00D2FF;'>Simulation prix & prévisions futuristes</div>", unsafe_allow_html=True)
with col_logo:
    if os.path.exists("assets/logo_neon.png"): st.image("assets/logo_neon.png", use_container_width=True)

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# SIDEBAR
# st.sidebar.header("Filtres")
# state_id = st.sidebar.selectbox("État", ["CA", "TX", "WI"])
# #selected_state = st.sidebar.selectbox("État", ["Californiaexas", "Wisconsin"])
# # Simulation liste magasins
# stores = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
# #selected_state = st.sidebar.selectbox("État", ["Californie_2", "Texas_1", "Wisconsin_1"]
# store_id = st.sidebar.selectbox("Magasin", [s for s in stores if state_id in s] or stores)
# cat_id = st.sidebar.selectbox("Catégorie", ["HOBBIES","FOODS","HOUSEHOLD"])
# item_id = st.sidebar.selectbox("Item ID", [f"{cat_id}_1_{str(i).zfill(3)}" for i in range(1,10)])


st.sidebar.header("Filtres")

# --- 1. Filtre État ---
# Récupérer les états disponibles
states_available = df_history['state_id'].unique().tolist() if not df_history.empty and 'state_id' in df_history.columns else ["CA", "TX", "WI"]
state_options = ["Tous"] + sorted(states_available)
state_id_selected = st.sidebar.selectbox("État", state_options)
state_filter = state_id_selected if state_id_selected != "Tous" else None

# --- 2. Filtre Magasin ---
stores_available = df_history['store_id'].unique().tolist() if not df_history.empty and 'store_id' in df_history.columns else ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
store_options = ["Tous"]
# Filtrer les magasins disponibles en fonction de l'état sélectionné, sauf si "Tous" est choisi pour l'état
if state_filter is not None:
    store_options += sorted([s for s in stores_available if state_filter in s])
else:
    store_options += sorted(stores_available)

store_id_selected = st.sidebar.selectbox("Magasin", store_options)
store_filter = store_id_selected if store_id_selected != "Tous" else None

# --- 3. Filtre Catégorie ---
cats_available = df_history['cat_id'].unique().tolist() if not df_history.empty and 'cat_id' in df_history.columns else ["HOBBIES","FOODS","HOUSEHOLD"]
cat_options = ["Tous"] + sorted(cats_available)
cat_id_selected = st.sidebar.selectbox("Catégorie", cat_options)
cat_filter = cat_id_selected if cat_id_selected != "Tous" else None

# --- 4. Filtre Article ---
items_available = []
if cat_filter is not None and not df_history.empty and 'cat_id' in df_history.columns:
    items_available = df_history[df_history['cat_id'] == cat_filter]['item_id'].unique().tolist()
elif not df_history.empty and 'item_id' in df_history.columns:
     items_available = df_history['item_id'].unique().tolist()
else:
    # Fallback pour le débogage/simulation
    items_available = [f"{cat_filter or 'FOODS'}_1_{str(i).zfill(3)}" for i in range(1,10)]

item_options = ["Tous"] + sorted(items_available)
item_id_selected = st.sidebar.selectbox("Item ID", item_options)
item_filter = item_id_selected if item_id_selected != "Tous" else None


# Dates cachées (Logique interne)
split_date = pd.to_datetime("2016-04-24")
end_date = split_date + timedelta(days=28)

st.sidebar.markdown("---")

# Dates cachées (Logique interne)
split_date = pd.to_datetime("2016-04-24")
end_date = split_date + timedelta(days=28)

st.sidebar.markdown("---")
# # INPUT PRIX
# price_change_slider = st.sidebar.slider("Variation prix (%)", -50, 50, 0, step=5)
# price_change_pct = price_change_slider / 100.0

# # DATES SIMULATION
# sim_col1, sim_col2 = st.sidebar.columns(2)
# sim_start = sim_col1.date_input("Sim start", value=split_date, min_value=split_date, max_value=end_date)
# sim_end = sim_col2.date_input("Sim end", value=end_date, min_value=split_date, max_value=end_date)

# st.sidebar.markdown("---")
# run_sim = st.sidebar.button("Lancer la simulation ✅")

# Fonction de visualisation

def plot_aggregated_sales(df_h, df_p, state_id=None, store_id=None, item_id=None, cat_id=None):
    # Fonction locale pour appliquer le filtre et agrégerfilter=None):
    # Fonction locale pour appliquer le filtre et agréger
    def apply_filter_logic(df, target_col):
        mask = pd.Series(True, index=df.index)
        filters_desc = []

        if item_id is not None:
            mask = mask & (df['item_id'] == item_id)
            filters_desc.append(f"Item: {item_id}")

        if store_id is not None:
            mask = mask & (df['store_id'] == store_id)
            filters_desc.append(f"Magasin: {store_id}")

        if cat_id is not None:
            mask = mask & (df['cat_id'] == cat_id)
            filters_desc.append(f"Catégorie: {cat_id}")

        if state_id is not None and 'state_id' in df.columns:
            mask = mask & (df['state_id'] == state_id)
            filters_desc.append(f"État: {state_id}")

        if not filters_desc:
            filters_desc.append("Vision Globale")

        # Agrégation
        return df[mask].groupby(level=0)[target_col].sum(), " / ".join(filters_desc)

    # Filtre sur Historique et Prévision
    ts_hist, title_suffix = apply_filter_logic(df_h, 'sales')
    ts_pred, _ = apply_filter_logic(df_p, 'pred_sales')

    # 4. Tracé
    fig = go.Figure()

    # Historique
    fig.add_trace(go.Scatter(
        x=ts_hist.index, y=ts_hist.values,
        mode='lines', name='Historique',
        line=dict(color='#1f77b4')
    ))

    # Prévision
    fig.add_trace(go.Scatter(
        x=ts_pred.index, y=ts_pred.values,
        mode='lines', name='Prévision',
        line=dict(color='red', dash='dash', width=2)
    ))

    fig.update_layout(
        title=f"Ventes & Prévisions - {title_suffix}",
        xaxis_title="Date", yaxis_title="Volume",
        template="plotly_dark", hovermode="x unified"
    )

    return fig, ts_hist, ts_pred

if not df_history.empty and not df_pred.empty:
    fig, ts_hist, ts_pred = plot_aggregated_sales(df_history, df_pred, state_filter, store_filter, item_filter, cat_filter)

    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Le warning a déjà été affiché dans la fonction plot_aggregated_sales si les données étaient vides
        pass
elif not df_history.empty and df_pred.empty:
    st.warning("Les données de prédiction sont manquantes ou n'ont pas pu être chargées/traitées.")
else:
    st.error("Les données historiques n'ont pas pu être chargées.")
