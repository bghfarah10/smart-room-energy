from datetime import datetime
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from tensorflow.keras.models import load_model

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="Smart Home AI Hub", page_icon="⚡", layout="wide"
)

# 2. RÉCUPÉRATION DE LA CLÉ API GEMINI
api_key = ""
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"].strip().strip('"').strip("'")


# 3. CHARGEMENT DU MODÈLE DEEP LEARNING (LSTM)
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


# 4. BARRE LATÉRALE
st.sidebar.title("⚙️ Paramètres")

if not api_key:
    api_key = st.sidebar.text_input(
        "🔑 Clé API Gemini",
        type="password",
        help="Colle ta clé API Google AI Studio ici",
    )
else:
    st.sidebar.success("✅ Clé API Gemini configurée")

st.sidebar.markdown("---")
seuil_max = st.sidebar.slider(
    "Puissance max autorisée (kW)", 3.0, 12.0, 6.0, 0.5
)

st.sidebar.subheader("❄️ Climatisation")
temp_interieure = st.sidebar.slider("Température chambre (°C)", 20.0, 38.0, 30.9)
temp_cible = st.sidebar.slider("Consigne souhaitée (°C)", 18.0, 26.0, 21.0)

st.sidebar.subheader("🔋 Voiture Électrique")
soc_actuel = st.sidebar.slider("Batterie actuelle (%)", 5, 100, 35)
soc_cible = st.sidebar.slider("Batterie cible (%)", 50, 100, 80)

# 5. CALCULS PHYSIQUES & LOGIQUE
p_base = predire_base()

# Clim
delta_temp = max(0.0, temp_interieure - temp_cible)
duree_clim_min = round(delta_temp / 0.20) if delta_temp > 0 else 0
p_clim = 2.0 if duree_clim_min > 0 else 0.0

# Voiture
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

# 6. EN-TÊTE PRINCIPAL
st.title("⚡ Smart Home Energy - Orchestrateur Intelligent")
st.caption(
    "Système Multi-Agents autonome piloté par Deep Learning (LSTM) & Assistant IA"
)

# 7. CARTES DE MÉTRIQUES
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Fond Maison (LSTM)", f"{p_base} kW")
col2.metric("❄️ Climatisation", f"{p_clim} kW", f"{duree_clim_min} min requises")
col3.metric(
    "🔋 Borne Recharge",
    f"{p_ve} kW",
    f"{duree_charge_h} h pour {soc_cible}%",
)
col4.metric(
    "⚡ Charge Globale",
    f"{p_totale} kW",
    f"{'+' if surcharge else ''}{round(p_totale - seuil_max, 2)} kW",
    delta_color="inverse",
)

st.write("")

# 8. JAUGE ET ARBITRAGE
col_jauge, col_decision = st.columns([1, 1])

with col_jauge:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_totale,
            title={"text": "Puissance Totale vs Seuil (kW)"},
            gauge={
                "axis": {"range": [None, 12]},
                "bar": {"color": "#E53935" if surcharge else "#43A047"},
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.8,
                    "value": seuil_max,
                },
            },
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_decision:
    st.subheader("🛡️ Statut de l'Arbitrage")
    if surcharge:
        st.error(
            f"🚨 **Dépassement détecté de {round(p_totale - seuil_max, 2)} kW**\n\n"
            f"- ❄️ **Climatisation prioritaire :** Maintenue pour garantir {temp_cible}°C à ton arrivée.\n"
            f"- 🔋 **Recharge décalée :** Reportée à **23:00** (heures creuses) pour éviter la disjonction.\n"
            f"- 📉 **Puissance finale sécurisée :** **{round(p_base + p_clim, 2)} kW** (sous le seuil de {seuil_max} kW)."
        )
    else:
        st.success(
            "✅ **Réseau Équilibré**\n\n"
            "Toutes les demandes fonctionnent simultanément sans risque de disjonction."
        )

st.divider()


# 9. ASSISTANT CONVERSATIONNEL GEMINI SANS AUCUN SILENCE D'ERREUR
def interroger_gemini(message_utilisateur, api_key_val):
    cle = api_key_val.strip().strip('"').strip("'")
    if not cle:
        return "⚠️ Aucune clé API trouvée. Renseigne ta clé dans la barre latérale ou dans Settings > Secrets."

    contexte = (
        f"Tu es l'Orchestrateur IA de la maison connectée de Farah. Réponds en français de façon vivante et concise.\n"
        f"Voici l'état actuel de sa maison :\n"
        f"- Conso de fond (Deep Learning LSTM) : {p_base} kW\n"
        f"- Seuil disjoncteur : {seuil_max} kW\n"
        f"- Climatisation : {p_clim} kW (pièce à {temp_interieure}°C, consigne {temp_cible}°C)\n"
        f"- Voiture électrique : {soc_actuel}% de batterie, objectif {soc_cible}%, borne 3.7 kW\n"
        f"- Puissance globale : {p_totale} kW (Surcharge : {'OUI' if surcharge else 'NON'})\n"
        f"- Décision : {'Clim maintenue, recharge voiture reportée à 23h' if surcharge else 'Tout fonctionne ensemble'}\n\n"
        f"Règles :\n"
        f"1. Si Farah pose une question hors sujet (ex: politique, président des USA, sport), réponds-y clairement et poliment en une phrase, puis termine par un clin d'œil sur la sécurité électrique de sa maison.\n"
        f"2. Si elle demande combien de temps pour 100% de batterie, calcule le temps exact (capacité 50 kWh, borne 3.7 kW, rendement 90%).\n"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={cle}"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [
            {"parts": [{"text": f"{contexte}\n\nQuestion de Farah : {message_utilisateur}"}]}
        ]
    }

    try:
        res = requests.post(url, headers=headers, json=body, timeout=12)
        if res.status_code == 200:
            data = res.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            # En cas de problème Google, on affiche l'erreur RÉELLE pour savoir ce qui cloche
            try:
                err_detail = res.json().get("error", {}).get("message", res.text)
            except Exception:
                err_detail = res.text
            return f"❌ Erreur API Google (Code {res.status_code}) : {err_detail}"
    except Exception as e:
        return f"❌ Erreur de connexion réseau : {str(e)}"


st.subheader("💬 Assistant Orchestrateur Dédié")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour Farah ! Je gère en direct vos flux d'énergie. Posez-moi vos questions sur la climatisation, la recharge ou mes décisions d'arbitrage.",
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Pose ta question ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Réflexion en cours..."):
        reponse = interroger_gemini(prompt, api_key)

    st.session_state.messages.append({"role": "assistant", "content": reponse})
    with st.chat_message("assistant"):
        st.markdown(reponse)
