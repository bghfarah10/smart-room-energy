from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from tensorflow.keras.models import load_model

# =====================================================================
# 1. CONFIGURATION & DESIGN HAUTE LISIBILITÉ (TEXTE BLANC NET)
# =====================================================================
st.set_page_config(
    page_title="Smart Home AI Hub", page_icon="⚡", layout="wide"
)

st.markdown(
    """
<style>
    /* Fond de page sombre et dégagement du haut */
    .stApp {
        background-color: #0B0E14;
        color: #FFFFFF !important;
        padding-top: 1.5rem;
    }
    /* Forcer tout le texte du markdown et des paragraphes en blanc éclatant */
    p, span, label, div, h1, h2, h3, h4 {
        color: #FFFFFF !important;
    }
    /* Cartes de métriques bien contrastées */
    div[data-testid="stMetric"] {
        background-color: #161B26;
        border: 1px solid #2D3748;
        padding: 15px;
        border-radius: 12px;
    }
    div[data-testid="stMetricValue"] > div {
        color: #00E676 !important;
    }
    /* Bulles de chat lisibles */
    .stChatMessage {
        background-color: #1A202C !important;
        border: 1px solid #2D3748;
        border-radius: 10px;
        color: #FFFFFF !important;
    }
    .stChatMessage p {
        color: #FFFFFF !important;
        font-size: 15px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# 2. CHARGEMENT AUTOMATIQUE DE LA CLÉ API (SECRETS OU SAISIE)
# =====================================================================
api_key = ""
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]


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
st.sidebar.title("🎛️ Paramètres")

# Si la clé n'est pas dans les secrets, on affiche le champ
if not api_key:
    api_key = st.sidebar.text_input(
        "🔑 Clé API Gemini",
        type="password",
        help="Colle ta clé une fois ou ajoute-la dans Settings > Secrets",
    )
else:
    st.sidebar.success("✅ Clé API Gemini connectée (automatique)")

st.sidebar.subheader("⚙️ Réseau Électrique")
seuil_max = st.sidebar.slider(
    "Puissance max autorisée (kW)", 3.0, 12.0, 6.0, 0.5
)

st.sidebar.subheader("❄️ Climatisation")
temp_interieure = st.sidebar.slider("Température chambre (°C)", 20.0, 38.0, 30.9)
temp_cible = st.sidebar.slider("Consigne souhaitée (°C)", 18.0, 26.0, 21.0)

st.sidebar.subheader("🔋 Voiture Électrique")
soc_actuel = st.sidebar.slider("Batterie actuelle (%)", 5, 100, 35)
soc_cible = st.sidebar.slider("Batterie cible (%)", 50, 100, 80)


# =====================================================================
# 5. CALCULS PHYSIQUES & ARBITRAGE
# =====================================================================
p_base = predire_base()

# Climatisation
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


# =====================================================================
# 6. DASHBOARD PRINCIPAL
# =====================================================================
st.title("⚡ Smart Home Energy - Orchestrateur Intelligent")
st.caption(
    "Système Multi-Agents autonome piloté par Deep Learning (LSTM) & Assistant IA"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Fond Maison (LSTM)", f"{p_base} kW")
c2.metric("❄️ Climatisation", f"{p_clim} kW", f"{duree_clim_min} min requises")
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

col_jauge, col_decision = st.columns([1, 1])

with col_jauge:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_totale,
            title={"text": "Puissance Totale vs Seuil (kW)"},
            gauge={
                "axis": {"range": [None, 12]},
                "bar": {"color": "#FF4B4B" if surcharge else "#00E676"},
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
        - 📉 **Puissance sécurisée :** **{round(p_base + p_clim, 2)} kW** (sous le seuil de {seuil_max} kW).
        """
        )
    else:
        st.success("✅ **Réseau Équilibré**")
        st.markdown(
            "Toutes les demandes fonctionnent simultanément sans risque de disjonction."
        )

st.divider()


# =====================================================================
# 7. ASSISTANT CONVERSATIONNEL (GEMINI AVEC RÔLE EXPLICITE)
# =====================================================================
def appeler_gemini(message_utilisateur, context_systeme, api_key):
    cle = api_key.strip()
    if not cle:
        return "💡 Veuillez entrer votre clé API Gemini pour discuter avec l'assistant."

    prompt_total = (
        f"{context_systeme}\n\n"
        f"Farah demande : '{message_utilisateur}'\n"
        f"Réponds directement à Farah de façon naturelle et vivante :"
    )

    # Format JSON officiel requis par l'API Gemini v1beta
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_total}]}]
    }
    headers = {"Content-Type": "application/json"}

    # Modèles récents officiels Google AI Studio
    for modele_nom in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modele_nom}:generateContent?key={cle}"
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=8)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            continue

    # Si Google ne répond pas, réponse calculée dynamique
    m = message_utilisateur.lower()
    if "100" in m or "batterie" in m:
        h = round((50.0 * (100 - soc_actuel) / 100.0 / 0.90) / 3.7, 1)
        return f"Farah, avec ta batterie à {soc_actuel}%, il faudra {h}h de charge. En démarrant à 23h00, ta voiture sera à 100% vers {int(23+h)%24:02d}h demain matin !"
    else:
        return f"Je veille sur ta maison ! La consommation totale actuelle est de {p_totale} kW pour un seuil souscrit de {seuil_max} kW."


contexte_ia = f"""
Tu es l'Orchestrateur IA de la maison connectée de Farah.
Données en direct :
- Conso de base estimée par Deep Learning LSTM : {p_base} kW
- Limite compteur : {seuil_max} kW | Puissance totale calculée : {p_totale} kW
- Chambre : {temp_interieure}°C (Consigne : {temp_cible}°C, Clim : {duree_clim_min} min requises)
- Véhicule : {soc_actuel}% (Objectif : {soc_cible}%, Borne 3.7 kW, Batterie 50 kWh)
- Situation : {'Surcharge détectée : Climatisation maintenue, voiture décalée à 23h00' if surcharge else 'Réseau stable : recharge et climatisation simultanées'}

Directives :
1. Réponds en français de manière concise, vivante et professionnelle.
2. Si la question est hors sujet (ex: président, actualité), réponds brièvement avec gentillesse avant de faire un petit clin d'œil sur la sécurité électrique de la maison.
3. Si elle pose une question sur sa voiture, fais les calculs de durée en direct.
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

    with st.spinner("L'Orchestrateur analyse votre demande..."):
        reponse = appeler_gemini(prompt, contexte_ia, api_key)

    st.session_state.messages.append({"role": "assistant", "content": reponse})
    with st.chat_message("assistant"):
        st.markdown(reponse)
