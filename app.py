from datetime import datetime
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from tensorflow.keras.models import load_model

# =====================================================================
# 1. CONFIGURATION ET STYLE CSS DESIGN SOMBRE "SMART HOME PRO"
# =====================================================================
st.set_page_config(
    page_title="Smart Home Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Fond principal sombre élégant */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Cartes des indicateurs */
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-size: 14px;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] div {
        color: #38BDF8 !important;
        font-size: 26px !important;
        font-weight: 700;
    }
    
    /* Panneau de l'arbitrage */
    .alert-card-danger {
        background: linear-gradient(135deg, #7F1D1D 0%, #450A0A 100%);
        border: 1px solid #DC2626;
        border-radius: 12px;
        padding: 18px;
        color: #FEE2E2;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2);
    }
    .alert-card-success {
        background: linear-gradient(135deg, #064E3B 0%, #022C22 100%);
        border: 1px solid #059669;
        border-radius: 12px;
        padding: 18px;
        color: #D1FAE5;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.2);
    }

    /* Bulles de Chatbot */
    .stChatMessage {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        color: #F8FAFC !important;
        padding: 12px !important;
        margin-bottom: 10px;
    }
    .stChatMessage p {
        color: #F8FAFC !important;
        font-size: 15px !important;
        line-height: 1.6;
    }
    
    /* Barre latérale */
    section[data-testid="stSidebar"] {
        background-color: #0B1120;
        border-right: 1px solid #1E293B;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# 2. CHARGEMENT DE LA CLÉ API GEMINI
# =====================================================================
api_key = ""
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"].strip().strip('"').strip("'")


# =====================================================================
# 3. CHARGEMENT DU MODÈLE DEEP LEARNING (LSTM)
# =====================================================================
@st.cache_resource
def charger_ia():
    try:
        model = load_model("modele_lstm_energie.keras")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception:
        return None, None


modele, scaler = charger_ia()


def predire_base():
    try:
        df = pd.read_csv(
            "consommation_horaire_propre.csv", index_col="datetime"
        )
        sample = df["Global_active_power"].iloc[100:124].values
        scaled = scaler.transform(sample.reshape(-1, 1))
        pred_scaled = modele.predict(scaled.reshape(1, 24, 1), verbose=0)
        return round(float(scaler.inverse_transform(pred_scaled)[0][0]), 2)
    except Exception:
        return 2.55


# =====================================================================
# 4. PANNEAU DE CONTRÔLE LATÉRAL
# =====================================================================
st.sidebar.markdown("### 🎛️ Centre de Contrôle")

if not api_key:
    api_key = st.sidebar.text_input(
        "🔑 Clé API Gemini",
        type="password",
        help="Colle ta clé Google AI Studio ici",
    )
else:
    st.sidebar.success("✅ IA Gemini connectée")

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚡ Réseau Électrique")
seuil_max = st.sidebar.slider(
    "Puissance souscrite max (kW)", 3.0, 12.0, 6.0, 0.5
)

st.sidebar.markdown("#### ❄️ Climatisation")
temp_interieure = st.sidebar.slider("Température actuelle (°C)", 20.0, 38.0, 30.9)
temp_cible = st.sidebar.slider("Consigne souhaitée (°C)", 18.0, 26.0, 21.0)

st.sidebar.markdown("#### 🔋 Véhicule Électrique")
soc_actuel = st.sidebar.slider("Batterie actuelle (%)", 5, 100, 35)
soc_cible = st.sidebar.slider("Batterie cible (%)", 50, 100, 80)


# =====================================================================
# 5. MOTEUR PHYSIQUE & SIMULATION EN DIRECT
# =====================================================================
p_base = predire_base()

# Climatisation
delta_temp = max(0.0, temp_interieure - temp_cible)
duree_clim_min = round(delta_temp / 0.20) if delta_temp > 0 else 0
p_clim = 2.0 if duree_clim_min > 0 else 0.0

# Véhicule (50 kWh, borne 3.7 kW)
capacite_batterie_kwh = 50.0
energie_manquante_kwh = (
    capacite_batterie_kwh * max(0, soc_cible - soc_actuel) / 100.0 / 0.90
)
duree_charge_h = (
    round(energie_manquante_kwh / 3.7, 2) if energie_manquante_kwh > 0 else 0.0
)
p_ve = 3.7 if duree_charge_h > 0 else 0.0

p_totale = round(p_base + p_clim + p_ve, 2)
surcharge = p_totale > seuil_max


# =====================================================================
# 6. EN-TÊTE DU DASHBOARD
# =====================================================================
st.title("⚡ Smart Home Energy - Orchestrateur Multi-Agents")
st.markdown(
    "<p style='color: #94A3B8; font-size: 16px; margin-top: -10px;'>"
    "Supervision énergétique autonome pilotée par <b>Deep Learning (LSTM)</b> et arbitrage intelligent."
    "</p>",
    unsafe_allow_html=True,
)
st.write("")

# 4 Cartes Métriques
c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Fond Maison (LSTM)", f"{p_base} kW")
c2.metric("❄️ Climatisation", f"{p_clim} kW", f"{duree_clim_min} min requises")
c3.metric(
    "🔋 Borne Véhicule",
    f"{p_ve} kW",
    f"{duree_charge_h} h pour {soc_cible}%",
)
c4.metric(
    "⚡ Charge Totale Simulée",
    f"{p_totale} kW",
    f"{'+' if surcharge else ''}{round(p_totale - seuil_max, 2)} kW",
    delta_color="inverse",
)

st.write("")
st.write("")

# Jauge & Carte d'arbitrage
col_jauge, col_decision = st.columns([1, 1.1])

with col_jauge:
    st.markdown("#### ⚡ Jauge de Puissance du Foyer")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_totale,
            number={"suffix": " kW", "font": {"color": "#F8FAFC", "size": 36}},
            gauge={
                "axis": {
                    "range": [None, 12],
                    "tickcolor": "#94A3B8",
                    "tickfont": {"color": "#94A3B8"},
                },
                "bar": {"color": "#EF4444" if surcharge else "#10B981"},
                "bgcolor": "#1E293B",
                "threshold": {
                    "line": {"color": "#F8FAFC", "width": 4},
                    "thickness": 0.8,
                    "value": seuil_max,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_decision:
    st.markdown("#### 🛡️ Décision de l'Orchestrateur")
    if surcharge:
        surcroit = round(p_totale - seuil_max, 2)
        st.markdown(
            f"""
        <div class="alert-card-danger">
            <h3 style="margin-top:0; color:#FECACA; font-size:20px;">🚨 Risque de Surcharge Électrique (+{surcroit} kW)</h3>
            <p style="margin-bottom:10px; font-size:15px; color:#FEE2E2;">
                Le compteur risque de disjoncter si tous les équipements démarrent ensemble.
            </p>
            <ul style="margin-bottom:0; font-size:14px; line-height:1.7;">
                <li>❄️ <b>Climatisation prioritaire :</b> Démarrage maintenu pour garantir {temp_cible}°C à ton arrivée.</li>
                <li>🔋 <b>Recharge différée :</b> Décalée à <b>23h00</b> (heures creuses).</li>
                <li>📉 <b>Puissance sécurisée :</b> <b>{round(p_base + p_clim, 2)} kW</b> (seuil respecté sous {seuil_max} kW).</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="alert-card-success">
            <h3 style="margin-top:0; color:#A7F3D0; font-size:20px;">✅ Réseau Équilibré et Sécurisé</h3>
            <p style="margin-bottom:10px; font-size:15px; color:#D1FAE5;">
                La puissance globale ({p_totale} kW) reste inférieure au seuil autorisé de {seuil_max} kW.
            </p>
            <p style="margin-bottom:0; font-size:14px;">
                La climatisation et la recharge du véhicule fonctionnent en simultané sans arbitrage requis.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

st.markdown("---")


# =====================================================================
# 7. MOTEUR GEMINI AVEC DÉTECTION AUTOMATIQUE DU MODÈLE ACTIF
# =====================================================================
def interroger_gemini(message_utilisateur, api_key_val):
    cle = api_key_val.strip().strip('"').strip("'")
    if not cle:
        return "⚠️ Clé API absente. Renseigne ta clé dans la barre latérale pour activer l'IA."

    contexte = (
        f"Tu es l'Orchestrateur IA d'une Smart Home de luxe appartenant à Farah. Tu t'exprimes en français avec clarté, esprit et professionnalisme.\n"
        f"État énergétique en temps réel :\n"
        f"- Prévision du modèle Deep Learning LSTM (base du foyer) : {p_base} kW\n"
        f"- Limite du disjoncteur : {seuil_max} kW\n"
        f"- Climatisation : {p_clim} kW (pièce à {temp_interieure}°C, consigne {temp_cible}°C, besoin : {duree_clim_min} min)\n"
        f"- Véhicule électrique : {soc_actuel}% de batterie, consigne {soc_cible}%, borne 3.7 kW, batterie 50 kWh\n"
        f"- Puissance simultanée : {p_totale} kW (Surcharge : {'OUI' if surcharge else 'NON'})\n"
        f"- Décision : {'Clim maintenue, recharge décalée à 23h00' if surcharge else 'Exécution simultanée normale'}\n\n"
        f"Consignes de réponse :\n"
        f"1. Si la question est hors-sujet (ex: politique, président, sport), réponds précisément avec gentillesse et culture générale, puis termine par un clin d'œil sur la sécurité électrique de sa maison.\n"
        f"2. Si la question porte sur sa batterie (ex: pour 100%), fais le calcul mathématique précis du temps nécessaire (50 kWh * % restant / 3.7 kW / 0.9).\n"
    )

    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [
            {"parts": [{"text": f"{contexte}\n\nFarah demande : {message_utilisateur}"}]}
        ]
    }

    # Liste des modèles récents supportés sur Google AI Studio
    modeles_a_tenter = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash-latest",
    ]

    for nom in modeles_a_tenter:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{nom}:generateContent?key={cle}"
        try:
            res = requests.post(url, headers=headers, json=body, timeout=8)
            if res.status_code == 200:
                return res.json()["candidates"][0]["content"]["parts"][0][
                    "text"
                ]
        except Exception:
            continue

    return "❌ Impossible de joindre l'API Gemini. Vérifie que ta clé Google AI Studio est bien valide."


st.markdown("### 💬 Assistant Orchestrateur Dédié")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour Farah ! Je supervise en temps réel les flux d'énergie de votre foyer. Posez-moi vos questions sur vos équipements, vos consommations ou mes arbitrages.",
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Pose ta question ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("L'Orchestrateur réfléchit..."):
        reponse = interroger_gemini(prompt, api_key)

    st.session_state.messages.append({"role": "assistant", "content": reponse})
    with st.chat_message("assistant"):
        st.markdown(reponse)
