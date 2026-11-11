from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from tensorflow.keras.models import load_model


# =====================================================================
# 1. CONFIGURATION ET STYLE
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
        .stApp {
            background-color: #0F172A;
            color: #F8FAFC;
        }
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
# 2. CLÉ API GEMINI
# =====================================================================
api_key = ""

try:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    # Aucun fichier secrets.toml n'est obligatoire :
    # la clé peut être saisie dans la barre latérale.
    pass

api_key = str(api_key).strip().strip('"').strip("'")


# =====================================================================
# 3. CHARGEMENT DU MODÈLE LSTM
# =====================================================================
@st.cache_resource
def charger_ia():
    try:
        model = load_model("modele_lstm_energie.keras")
        scaler_obj = joblib.load("scaler.pkl")
        return model, scaler_obj
    except Exception as erreur:
        return None, None


modele, scaler = charger_ia()


def predire_base():
    """Prévoit la consommation à partir des 24 dernières valeurs disponibles."""
    if modele is None or scaler is None:
        return 2.55, False

    try:
        df = pd.read_csv("consommation_horaire_propre.csv")

        if "Global_active_power" not in df.columns:
            raise ValueError("Colonne Global_active_power absente du CSV.")

        valeurs = df["Global_active_power"].dropna().iloc[100:124].to_numpy()

        if len(valeurs) != 24:
            raise ValueError("Le modèle LSTM nécessite exactement 24 valeurs.")

        valeurs = valeurs.reshape(-1, 1)
        valeurs_scaled = scaler.transform(valeurs)
        entree = valeurs_scaled.reshape(1, 24, 1)

        prediction_scaled = modele.predict(entree, verbose=0)
        prediction = scaler.inverse_transform(
            np.asarray(prediction_scaled).reshape(-1, 1)
        )[0, 0]

        return round(max(0.0, float(prediction)), 2), True

    except Exception as erreur:
        st.warning(
            f"Prévision LSTM indisponible : {erreur}. "
            "Une valeur de démonstration de 2.55 kW est utilisée."
        )
        return 2.55, False


# =====================================================================
# 4. PANNEAU DE CONTRÔLE LATÉRAL
# =====================================================================
st.sidebar.markdown("### 🎛️ Centre de Contrôle")

if not api_key:
    api_key = st.sidebar.text_input(
        "🔑 Clé API Gemini",
        type="password",
        help="Colle ici ta clé Google AI Studio.",
    ).strip()
else:
    st.sidebar.success("✅ Clé API Gemini configurée")

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚡ Réseau électrique")
seuil_max = st.sidebar.slider(
    "Puissance souscrite max (kW)", 3.0, 12.0, 6.0, 0.5
)

st.sidebar.markdown("#### ❄️ Climatisation")
temp_interieure = st.sidebar.slider(
    "Température actuelle (°C)", 20.0, 38.0, 30.9
)
temp_cible = st.sidebar.slider(
    "Consigne souhaitée (°C)", 18.0, 26.0, 21.0
)

st.sidebar.markdown("#### 🔋 Véhicule électrique")
soc_actuel = st.sidebar.slider("Batterie actuelle (%)", 5, 100, 35)
soc_cible = st.sidebar.slider("Batterie cible (%)", 50, 100, 80)


# =====================================================================
# 5. MOTEUR PHYSIQUE ET SIMULATION
# =====================================================================
p_base, prediction_reelle = predire_base()

# Modèle simplifié de climatisation
delta_temp = max(0.0, temp_interieure - temp_cible)
duree_clim_min = round(delta_temp / 0.20) if delta_temp > 0 else 0
p_clim = 2.0 if duree_clim_min > 0 else 0.0

# Véhicule électrique : batterie de 50 kWh et borne de 3.7 kW
capacite_batterie_kwh = 50.0
puissance_borne_kw = 3.7
rendement_charge = 0.90

energie_manquante_kwh = (
    capacite_batterie_kwh
    * max(0, soc_cible - soc_actuel)
    / 100.0
    / rendement_charge
)

duree_charge_h = (
    round(energie_manquante_kwh / puissance_borne_kw, 2)
    if energie_manquante_kwh > 0
    else 0.0
)

p_ve = puissance_borne_kw if duree_charge_h > 0 else 0.0

# Puissance théorique si tous les équipements fonctionnent ensemble
p_totale = round(p_base + p_clim + p_ve, 2)
surcharge = p_totale > seuil_max

# Arbitrage : priorité à la climatisation, recharge du VE différée.
p_securisee = round(p_base + p_clim, 2) if surcharge else p_totale
recharge_differee = surcharge and p_ve > 0
surcharge_residuelle = p_securisee > seuil_max


# =====================================================================
# 6. EN-TÊTE DU DASHBOARD
# =====================================================================
st.title("⚡ Smart Home Energy - Orchestrateur Multi-Agents")
st.markdown(
    """
    <p style='color: #94A3B8; font-size: 16px; margin-top: -10px;'>
    Supervision énergétique pilotée par <b>Deep Learning (LSTM)</b>
    et arbitrage intelligent multi-agents.
    </p>
    """,
    unsafe_allow_html=True,
)

if not prediction_reelle:
    st.caption(
        "Mode démonstration : la prévision affichée est une valeur de secours."
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Fond maison (LSTM)", f"{p_base:.2f} kW")
c2.metric("❄️ Climatisation", f"{p_clim:.2f} kW",
          f"{duree_clim_min} min estimées")
c3.metric("🔋 Borne véhicule", f"{p_ve:.2f} kW",
          f"{duree_charge_h} h pour atteindre {soc_cible}%")
c4.metric(
    "⚡ Puissance totale théorique",
    f"{p_totale:.2f} kW",
    f"{p_totale - seuil_max:+.2f} kW par rapport au seuil",
    delta_color="inverse",
)

st.write("")

# Jauge et décision
col_jauge, col_decision = st.columns([1, 1.1])

with col_jauge:
    st.markdown("#### ⚡ Jauge de puissance du foyer")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_totale,
            number={
                "suffix": " kW",
                "font": {"color": "#F8FAFC", "size": 36},
            },
            gauge={
                "axis": {
                    "range": [0, max(12, seuil_max, p_totale + 1)],
                    "tickcolor": "#94A3B8",
                    "tickfont": {"color": "#94A3B8"},
                },
                "bar": {
                    "color": "#EF4444" if surcharge else "#10B981"
                },
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
        font={"color": "#F8FAFC"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col_decision:
    st.markdown("#### 🛡️ Décision de l'orchestrateur")

    if surcharge and not surcharge_residuelle:
        st.markdown(
            f"""
            <div class="alert-card-danger">
                <h3 style="margin-top:0; color:#FECACA; font-size:20px;">
                    🚨 Risque de surcharge électrique
                </h3>
                <p style="font-size:15px; color:#FEE2E2;">
                    La puissance théorique atteint {p_totale:.2f} kW,
                    au-dessus du seuil de {seuil_max:.2f} kW.
                </p>
                <ul style="font-size:14px; line-height:1.7;">
                    <li>❄️ Climatisation prioritaire : fonctionnement maintenu.</li>
                    <li>🔋 Recharge du véhicule différée à 23h00.</li>
                    <li>📉 Puissance après arbitrage : {p_securisee:.2f} kW,
                    sous le seuil autorisé.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif surcharge_residuelle:
        st.markdown(
            f"""
            <div class="alert-card-danger">
                <h3 style="margin-top:0; color:#FECACA; font-size:20px;">
                    ⚠️ Surcharge persistante
                </h3>
                <p style="font-size:15px; color:#FEE2E2;">
                    Même après le report de la recharge du véhicule,
                    la puissance prioritaire atteint {p_securisee:.2f} kW,
                    au-dessus du seuil de {seuil_max:.2f} kW.
                </p>
                <p style="font-size:14px;">
                    Une réduction supplémentaire de la consommation
                    est nécessaire. La sécurité ne peut pas être garantie
                    par le seul report de la recharge.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            f"""
            <div class="alert-card-success">
                <h3 style="margin-top:0; color:#A7F3D0; font-size:20px;">
                    ✅ Réseau équilibré et sécurisé
                </h3>
                <p style="font-size:15px; color:#D1FAE5;">
                    La puissance totale ({p_totale:.2f} kW) reste
                    inférieure au seuil autorisé ({seuil_max:.2f} kW).
                </p>
                <p style="font-size:14px;">
                    La climatisation et la recharge du véhicule peuvent
                    fonctionner simultanément selon cette simulation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")


# =====================================================================
# 7. CONNEXION GEMINI : DÉTECTION DES MODÈLES DISPONIBLES
# =====================================================================
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def obtenir_modele_gemini(cle):
    """
    Interroge l'API Google pour récupérer les modèles réellement
    disponibles pour cette clé et compatibles avec generateContent.
    """
    url = f"{GEMINI_API_BASE}/models"
    try:
        response = requests.get(
            url,
            headers={"x-goog-api-key": cle},
            timeout=15,
        )

        if response.status_code != 200:
            try:
                details = response.json().get("error", {}).get(
                    "message", response.text
                )
            except ValueError:
                details = response.text
            return None, f"Erreur {response.status_code} lors de la liste des modèles : {details}"

        donnees = response.json()
        modeles = donnees.get("models", [])

        disponibles = []
        for modele_info in modeles:
            methodes = modele_info.get("supportedGenerationMethods", [])
            nom = modele_info.get("name", "")

            if "generateContent" in methodes and nom.startswith("models/"):
                nom_court = nom.replace("models/", "", 1)
                disponibles.append(nom_court)

        if not disponibles:
            return None, (
                "Aucun modèle compatible avec generateContent n'est disponible "
                "pour cette clé API."
            )

        # Priorité aux modèles Flash lorsqu'ils sont proposés par Google.
        priorites = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite",
        ]

        for nom in priorites:
            if nom in disponibles:
                return nom, None

        # Sinon, prendre le premier modèle compatible retourné par l'API.
        return disponibles[0], None

    except requests.RequestException as erreur:
        return None, f"Impossible de contacter Google AI Studio : {erreur}"
    except (ValueError, KeyError) as erreur:
        return None, f"Réponse inattendue de l'API Google : {erreur}"


def interroger_gemini(message_utilisateur, api_key_val):
    cle = str(api_key_val).strip().strip('"').strip("'")

    if not cle:
        return "⚠️ Clé API absente. Renseigne ta clé dans la barre latérale."

    nom_modele, erreur_modele = obtenir_modele_gemini(cle)

    if not nom_modele:
        return f"❌ {erreur_modele}\n\nVérifie ta clé API dans Google AI Studio."

    contexte = f"""
Tu es l'assistant orchestrateur d'une maison intelligente.
Tu réponds en français, de manière claire, naturelle et professionnelle.

État énergétique actuel :
- Prévision LSTM de la consommation de base : {p_base} kW
- Seuil de puissance souscrite : {seuil_max} kW
- Température intérieure : {temp_interieure} °C
- Température cible : {temp_cible} °C
- Puissance de climatisation estimée : {p_clim} kW
- Batterie du véhicule : {soc_actuel} %
- Objectif de batterie : {soc_cible} %
- Capacité de batterie : 50 kWh
- Puissance de recharge : 3.7 kW
- Puissance totale théorique : {p_totale} kW
- Surcharge théorique : {"OUI" if surcharge else "NON"}
- Puissance après arbitrage : {p_securisee} kW
- Recharge différée : {"OUI, à 23h00" if recharge_differee else "NON"}

Règles :
1. Réponds directement à la question posée.
2. Pour une question sur la maison, utilise les valeurs ci-dessus.
3. Pour une question sur le temps de recharge, utilise :
   temps (heures) = énergie nécessaire (kWh) / (3.7 kW * 0.90).
   L'énergie nécessaire est calculée à partir d'une batterie de 50 kWh.
4. Ne prétends pas commander réellement les appareils : il s'agit d'une simulation.
5. Pour une question générale hors sujet, réponds normalement avec une
   information factuelle et concise. Ne force pas le sujet énergétique.
6. Si une donnée manque, indique clairement qu'il s'agit d'une estimation.
"""

    url = (
        f"{GEMINI_API_BASE}/models/{nom_modele}:generateContent"
    )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": cle,
    }

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"{contexte}\n\n"
                            f"Question de l'utilisateur : {message_utilisateur}"
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 800,
        },
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=30,
        )

        if response.status_code != 200:
            try:
                details = response.json().get("error", {}).get(
                    "message", response.text
                )
            except ValueError:
                details = response.text

            return (
                f"❌ Erreur Gemini ({response.status_code}) avec "
                f"{nom_modele} : {details}"
            )

        donnees = response.json()
        candidats = donnees.get("candidates", [])

        if not candidats:
            return (
                "⚠️ Gemini n'a pas retourné de réponse. "
                "La requête a peut-être été bloquée ou limitée."
            )

        parties = candidats[0].get("content", {}).get("parts", [])
        texte = "\n".join(
            partie.get("text", "")
            for partie in parties
            if partie.get("text")
        ).strip()

        if not texte:
            return "⚠️ La réponse Gemini est vide."

        return texte

    except requests.Timeout:
        return "❌ Gemini met trop de temps à répondre. Réessaie dans quelques instants."
    except requests.RequestException as erreur:
        return f"❌ Erreur de connexion à Gemini : {erreur}"
    except (ValueError, KeyError, IndexError) as erreur:
        return f"❌ Impossible de lire la réponse Gemini : {erreur}"


# =====================================================================
# 8. ASSISTANT ORCHESTRATEUR
# =====================================================================
st.markdown("### 💬 Assistant Orchestrateur Dédié")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour ! Je supervise les flux énergétiques de votre "
                "maison en mode simulation. Posez-moi vos questions sur "
                "la climatisation, la recharge du véhicule ou la consommation."
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Pose ta question ici...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("L'orchestrateur réfléchit..."):
            reponse = interroger_gemini(prompt, api_key)
        st.markdown(reponse)

    st.session_state.messages.append(
        {"role": "assistant", "content": reponse}
    )
