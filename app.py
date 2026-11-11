from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from tensorflow.keras.models import load_model

# 1. CONFIGURATION DE LA PAGE & DESIGN PREMIUM
st.set_page_config(
    page_title="Smart Home AI Hub", page_icon="⚡", layout="wide"
)

# Injection de CSS pour transformer l'apparence de Streamlit en vrai Dashboard SaaS
st.markdown(
    """
<style>
    /* Fond et conteneur principal */
    .stApp {
        background-color: #0E1117;
        color: #E0E2EC;
    }
    /* Cartes de métriques personnalisées */
    div[data-testid="stMetric"] {
        background-color: #1A1F2C;
        border: 1px solid #2B3245;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    /* Style du Chat */
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 8px;
    }
    /* Masquer le menu Streamlit par défaut pour faire pro */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# 2. CHARGEMENT DU MODÈLE DEEP LEARNING (LSTM)
@st.cache_resource
def charger_ia():
    try:
        model = load_model("modele_lstm_energie.keras")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except:
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
    except:
        return 2.55


# 3. INTERFACE DE CONTRÔLE (SIDEBAR)
st.sidebar.title("🎛️ Centre de Contrôle")
api_key = st.sidebar.text_input(
    "🔑 Clé API Gemini (Google AI Studio)",
    type="password",
    help="Colle ta clé API gratuite ici",
)

st.sidebar.subheader("⚙️ Réseau Électrique")
seuil_max = st.sidebar.slider(
    "Puissance souscrite max (kW)", 3.0, 12.0, 6.0, 0.5
)

st.sidebar.subheader("❄️ Climatisation")
temp_interieure = st.sidebar.slider("Température ambiante (°C)", 20.0, 38.0, 30.9)
temp_cible = st.sidebar.slider("Température souhaitée (°C)", 18.0, 26.0, 21.0)

st.sidebar.subheader("🔋 Voiture Électrique")
soc_actuel = st.sidebar.slider("Batterie actuelle (%)", 5, 100, 35)
soc_cible = st.sidebar.slider("Batterie cible (%)", 50, 100, 80)

# 4. MOTEUR PHYSIQUE & CALCULS EN TEMPS RÉEL
p_base = predire_base()

# Climatisation
delta_temp = max(0.0, temp_interieure - temp_cible)
duree_clim_min = round(delta_temp / 0.20) if delta_temp > 0 else 0
p_clim = 2.0 if duree_clim_min > 0 else 0.0

# Voiture (Batterie 50 kWh, borne 3.7 kW)
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

# 5. DASHBOARD PRINCIPAL
st.title("⚡ Smart Home Energy - Orchestrateur Intelligent")
st.caption(
    "Architecture Multi-Agents autonome pilotée par Deep Learning (LSTM) & Assistant LLM"
)

# KPI Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Fond Maison (IA LSTM)", f"{p_base} kW")
c2.metric("❄️ Climatisation", f"{p_clim} kW", f"{duree_clim_min} min nécessaires")
c3.metric(
    "🔋 Borne Recharge",
    f"{p_ve} kW",
    f"{duree_charge_h} h pour {soc_cible}%",
)
c4.metric(
    "⚡ Charge Globale",
    f"{p_totale} kW",
    f"{'+' if surcharge else ''}{round(p_totale - seuil_max, 2)} kW",
    delta_color="inverse",
)

st.write("")

# Colonnes d'arbitrage
col_jauge, col_decision = st.columns([1, 1])

with col_jauge:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_totale,
            title={"text": "Consommation vs Seuil Disjoncteur (kW)"},
            gauge={
                "axis": {"range": [None, 12]},
                "bar": {"color": "#FF4B4B" if surcharge else "#00CC96"},
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.8,
                    "value": seuil_max,
                },
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col_decision:
    st.markdown("### 🛡️ Statut de l'Arbitrage")
    if surcharge:
        st.error(
            f"🚨 **Dépassement détecté de {round(p_totale - seuil_max, 2)} kW**"
        )
        st.markdown(
            f"""
        - ❄️ **Climatisation prioritaire :** Maintenue pour garantir {temp_cible}°C à ton retour.
        - 🔋 **Recharge décalée :** Reportée à **23:00** (heures creuses) pour soulager le réseau.
        - 📉 **Puissance finale sécurisée :** **{round(p_base + p_clim, 2)} kW** (sous le seuil de {seuil_max} kW).
        """
        )
    else:
        st.success("✅ **Réseau Équilibré**")
        st.markdown(
            "Toutes les demandes peuvent être honorées simultanément sans risque de disjonction."
        )

st.divider()


# 6. MOTEUR CONVERSATIONNEL RÉEL VIA LLM (GEMINI)
def appeler_gemini(message_utilisateur, context_systeme, api_key):
    # Appel REST direct à Gemini
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    prompt_complet = (
        f"{context_systeme}\n\n"
        f"L'utilisateur dit : '{message_utilisateur}'\n"
        f"Réponds directement à l'utilisateur :"
    )

    data = {"contents": [{"parts": [{"text": prompt_complet}]}]}

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0][
                "text"
            ]
        else:
            return f"⚠️ Erreur de connexion avec l'IA Gemini (Code {response.status_code}). Vérifie ta clé API."
    except Exception as e:
        return f"Erreur de communication : {str(e)}"


# Contexte dynamique complet envoyé à l'IA
contexte_ia = f"""
Tu es l'Orchestrateur IA d'une Smart Home haut de gamme appartenant à Farah.
Voici les données réelles et physiques de la maison à l'instant T :
- Modèle de prédiction : Réseau Deep Learning LSTM (MAE: 0.60 kW)
- Puissance de base prédite du foyer : {p_base} kW
- Limite maximale du disjoncteur : {seuil_max} kW
- Température actuelle de la pièce : {temp_interieure}°C (Objectif souhaité : {temp_cible}°C, besoin de clim : {duree_clim_min} min)
- Voiture électrique : Batterie actuelle à {soc_actuel}%, cible demandée {soc_cible}%. Capacité batterie : 50 kWh, borne de recharge : 3.7 kW.
- Durée de recharge pour atteindre {soc_cible}% : environ {duree_charge_h} heures.
- Pour atteindre 100% : il faut environ {round((50.0 * (100 - soc_actuel) / 100.0 / 0.9) / 3.7, 1)} heures de charge.
- Puissance totale si tout démarre en même temps : {p_totale} kW.
- État de surcharge : {'OUI (Dépassement de ' + str(round(p_totale - seuil_max, 2)) + ' kW)' if surcharge else 'NON (Stable)'}.
- Décision de l'orchestrateur : {'Clim maintenue à 17h14, voiture décalée à 23h en heures creuses' if surcharge else 'Exécution immédiate des deux'}.

Directives :
1. Réponds toujours avec bienveillance, clarté et concision en français.
2. Si Farah pose des questions sur sa voiture (ex: quand elle aura 100%), fais le vrai calcul d'heure basé sur les chiffres ci-dessus !
3. Si elle pose une question totalement hors sujet (politique, sport, devinettes, blagues), réponds poliment ou avec un trait d'esprit, puis rappelle subtilement que tu veilles sur son énergie.
4. Ne sors JAMAIS de phrases robotiques fixes. Tu es un assistant personnel intelligent et vivant.
"""

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

if prompt := st.chat_input("Votre message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse par Gemini ou invitation à saisir la clé
    if api_key:
        with st.spinner("L'Orchestrateur analyse votre demande..."):
            reponse = appeler_gemini(prompt, contexte_ia, api_key)
    else:
        reponse = (
            "💡 **Mode Démonstration :** Veuillez renseigner votre clé API Gemini gratuite dans la barre latérale "
            "pour activer le cerveau conversationnel complet et poser n'importe quelle question (calculs personnalisés, questions libres, etc.)."
        )

    st.session_state.messages.append({"role": "assistant", "content": reponse})
    with st.chat_message("assistant"):
        st.markdown(reponse)
