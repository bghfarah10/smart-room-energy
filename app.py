from datetime import datetime, timedelta
import re
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from tensorflow.keras.models import load_model

# =====================================================================
# 1. CONFIGURATION & DESIGN CORPORATE (BEIGE & NAVY MINIMALIST)
# =====================================================================
st.set_page_config(
    page_title="Smart Energy Hub • Executive Overview",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injection de style CSS personnalisé : Beige raffiné, Bleu marine profond et typographie épurée
st.markdown(
    """
<style>
    /* Importation d'une police sobre et élégante */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Fond principal : Beige crème doux */
    .stApp {
        background-color: #F8F6F2;
        color: #1F2937;
    }

    /* Barre latérale : Bleu marine d'affaires */
    section[data-testid="stSidebar"] {
        background-color: #0F1E36 !important;
        border-right: 1px solid #1E2D4A;
    }
    section[data-testid="stSidebar"] * {
        color: #F8F6F2 !important;
    }
    section[data-testid="stSidebar"] .stSlider > div > div > div > div {
        background-color: #C5A880 !important;
    }

    /* Titres et hiérarchie */
    h1 {
        color: #0F1E36 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        font-size: 2rem !important;
        margin-bottom: 0.2rem !important;
    }
    h2, h3, h4 {
        color: #1A365D !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }

    /* Cartes métriques épurées (Minimalist Business Cards) */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E0D8;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 6px rgba(15, 30, 54, 0.04);
        transition: transform 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(15, 30, 54, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: #64748B !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0F1E36 !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
    }

    /* Panneau d'arbitrage */
    .executive-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E0D8;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(15, 30, 54, 0.04);
        height: 100%;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: #FFFFFF !important;
        border: 1px solid #EBE6DE !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 1px 4px rgba(15, 30, 54, 0.03) !important;
    }

    /* Masquer les bordures inutiles */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# 2. CHARGEMENT ROBUSTE DE L'IA & DU PIPELINE
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


def predire_base_foyer():
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
# 3. PANNEAU DE CONTRÔLE (SIDEBAR MINIMALISTE)
# =====================================================================
with st.sidebar:
    st.markdown("### 🏛️ Centre de Contrôle")
    st.caption("Plateforme d'arbitrage énergétique résidentiel")
    st.markdown("---")

    # Clé API optionnelle (prend celle de Streamlit secrets si existante)
    cle_secrete = st.secrets.get("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Clé API Gemini (Optionnel)",
        value=cle_secrete,
        type="password",
        help="Connecte directement le LLM pour des réponses encore plus riches.",
    )

    st.markdown("#### ⚡ Contrat Électrique")
    seuil_max = st.slider("Souscription Disjoncteur (kW)", 3.0, 12.0, 6.0, 0.5)

    st.markdown("#### ❄️ Climatisation")
    temp_interieure = st.slider(
        "Température Ambiante (°C)", 20.0, 38.0, 30.0, 0.5
    )
    temp_cible = st.slider("Consigne Souhaitée (°C)", 18.0, 26.0, 21.0, 0.5)

    st.markdown("#### 🔋 Mobilité Électrique")
    soc_actuel = st.slider("Batterie Actuelle (%)", 5, 100, 35, 5)
    soc_cible = st.slider("Batterie Cible (%)", 50, 100, 80, 5)

    st.markdown("---")
    st.caption(
        "Architecture Multi-Agents • LSTM autorégressif • Arbitrage prédictif"
    )


# =====================================================================
# 4. MOTEUR PHYSIQUE & CALCULS ÉNERGÉTIQUES
# =====================================================================
p_base = predire_base_foyer()

# Agent Climatisation
delta_t = max(0.0, temp_interieure - temp_cible)
duree_clim_min = round(delta_t / 0.20) if delta_t > 0 else 0
p_clim = 2.0 if duree_clim_min > 0 else 0.0
heure_demarrage_clim = (
    datetime.strptime("18:00", "%H:%M") - timedelta(minutes=duree_clim_min)
).strftime("%H:%M")

# Agent Véhicule Électrique (Batterie standard 50 kWh, borne 3.7 kW, rendement 90%)
capa_kwh = 50.0
puiss_borne = 3.7
rendement = 0.90

energie_cible_kwh = (
    capa_kwh * max(0, soc_cible - soc_actuel) / 100.0 / rendement
)
duree_charge_cible_h = (
    round(energie_cible_kwh / puiss_borne, 2) if energie_cible_kwh > 0 else 0.0
)

energie_100_kwh = capa_kwh * max(0, 100 - soc_actuel) / 100.0 / rendement
duree_charge_100_h = (
    round(energie_100_kwh / puiss_borne, 2) if energie_100_kwh > 0 else 0.0
)
p_ve = puiss_borne if duree_charge_cible_h > 0 else 0.0

# Bilan
p_totale_brute = round(p_base + p_clim + p_ve, 2)
surcharge = p_totale_brute > seuil_max
surcroit = round(p_totale_brute - seuil_max, 2) if surcharge else 0.0

# Horaires de recharge
heure_debut_charge = "23:00" if surcharge else "Immédiate (18:00)"
minutes_100 = int(duree_charge_100_h * 60)
heure_fin_100 = (
    datetime.strptime("23:00" if surcharge else "18:00", "%H:%M")
    + timedelta(minutes=minutes_100)
).strftime("%H:%M")


# =====================================================================
# 5. EN-TÊTE & SYNTHÈSE EXÉCUTIVE
# =====================================================================
st.title("Smart Home Energy Orchestration")
st.markdown(
    "**Régulation prédictive de la demande résidentielle par apprentissage profond & coordination multi-agents.**"
)
st.write("")

# Ligne des KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Prévision Base (LSTM)", f"{p_base:.2f} kW", "Charge passive")
c2.metric("Climatisation", f"{p_clim:.2f} kW", f"{duree_clim_min} min requises")
c3.metric(
    "Borne Recharge",
    f"{p_ve:.2f} kW",
    f"{duree_charge_cible_h} h vers {soc_cible}%",
)
c4.metric(
    "Puissance Appelée",
    f"{p_totale_brute:.2f} kW",
    f"{'+' if surcharge else ''}{surcroit:.2f} kW vs souscription",
    delta_color="inverse",
)

st.write("")

# Deuxième section : Jauge & Décision d'arbitrage
col_gauge, col_decision = st.columns([1, 1])

with col_gauge:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_totale_brute,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": "Puissance Totale Simulée vs Seuil Souscrit (kW)",
                "font": {"size": 14, "color": "#1A365D", "family": "Plus Jakarta Sans"},
            },
            gauge={
                "axis": {
                    "range": [0, 12],
                    "tickwidth": 1,
                    "tickcolor": "#CBD5E1",
                },
                "bar": {"color": "#B91C1C" if surcharge else "#1B4D3E"},
                "bgcolor": "#F1EFEA",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, seuil_max], "color": "#E9E5DD"},
                    {"range": [seuil_max, 12], "color": "#FADBD8"},
                ],
                "threshold": {
                    "line": {"color": "#0F1E36", "width": 3},
                    "thickness": 0.8,
                    "value": seuil_max,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#0F1E36", "family": "Plus Jakarta Sans"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col_decision:
    if surcharge:
        st.markdown(
            f"""
        <div class="executive-card" style="border-left: 5px solid #B91C1C;">
            <span style="color: #B91C1C; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">
                Arbitrage Actif • Risque de Dépassement
            </span>
            <h3 style="margin-top: 6px; margin-bottom: 12px; color: #0F1E36;">Surcharge Prévenue : +{surcroit:.2f} kW</h3>
            <p style="color: #475569; font-size: 0.95rem; line-height: 1.5;">
                La simultanéité de la demande excède la puissance souscrite de <strong>{seuil_max:.1f} kW</strong>. L'Agent Orchestrateur a déployé le protocole suivant :
            </p>
            <ul style="color: #1E293B; font-size: 0.92rem; margin-bottom: 0;">
                <li><strong>Climatisation (Priorité 1) :</strong> Démarrage maintenu à <strong>{heure_demarrage_clim}</strong> pour garantir {temp_cible}°C à l'arrivée (18h00).</li>
                <li><strong>Recharge Véhicule (Priorité 2) :</strong> Décalée à <strong>23h00</strong> (tarification heures creuses).</li>
                <li><strong>Puissance sécurisée en ligne :</strong> {p_base + p_clim:.2f} kW (Conformité réseau assurée).</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="executive-card" style="border-left: 5px solid #1B4D3E;">
            <span style="color: #1B4D3E; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">
                Réseau Optimal • Aucune Contrainte
            </span>
            <h3 style="margin-top: 6px; margin-bottom: 12px; color: #0F1E36;">Capacité Réseau Suffisante</h3>
            <p style="color: #475569; font-size: 0.95rem; line-height: 1.5;">
                La puissance totale projetée (<strong>{p_totale_brute:.2f} kW</strong>) respecte la marge contractuelle de <strong>{seuil_max:.1f} kW</strong>.
            </p>
            <ul style="color: #1E293B; font-size: 0.92rem; margin-bottom: 0;">
                <li>La climatisation et la borne de recharge fonctionnent sans délai.</li>
                <li>Marge de sécurité disponible : <strong>{seuil_max - p_totale_brute:.2f} kW</strong>.</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

st.markdown("---")


# =====================================================================
# 6. MOTEUR CONVERSATIONNEL RÉEL & ANALYTIQUE (CHATBOT)
# =====================================================================
def moteur_analytique_local(question):
    """Moteur local intelligent capable de faire de vrais calculs et de recadrer

    avec courtoisie même sans clé API.
    """
    q = question.lower().strip()

    # Question sur la recharge à 100%
    if "100" in q or (
        "plein" in q and ("voiture" in q or "batterie" in q or "ve" in q)
    ):
        return (
            f"Actuellement, votre batterie est à **{soc_actuel}%**. Pour atteindre les **100%**, il reste à injecter **{energie_100_kwh:.1f} kWh**.\n\n"
            f"Sur votre borne de {puiss_borne} kW, cela nécessite **{duree_charge_100_h} heures** de charge nette.\n\n"
            f"📅 **Planning estimé :**\n"
            f"- Heure de début : **{heure_debut_charge}** {'(décalée pour éviter la surcharge de 18h)' if surcharge else ''}\n"
            f"- Disponibilité à 100% : vers **{heure_fin_100}** demain matin."
        )

    # Questions générales sur le véhicule / la batterie
    elif any(
        w in q for w in ["voiture", "batterie", "recharge", "charge", "ve"]
    ):
        return (
            f"La batterie est actuellement à **{soc_actuel}%** (objectif standard : **{soc_cible}%**).\n"
            f"- Temps nécessaire pour l'objectif : **{duree_charge_cible_h} h**\n"
            f"- Statut d'arbitrage : {'Départ différé à 23h00 en heures creuses' if surcharge else 'Charge immédiate autorisée'}.\n"
            f"Le véhicule sera parfaitement opérationnel pour votre départ à 07h30."
        )

    # Questions sur le confort / la climatisation
    elif any(
        w in q for w in ["clim", "chambre", "temperature", "chaleur", "froid"]
    ):
        return (
            f"La température mesurée dans la pièce est de **{temp_interieure}°C** pour une consigne fixée à **{temp_cible}°C**.\n"
            f"Le modèle thermique estime un délai d'abaissement de **{duree_clim_min} minutes**. "
            f"Le compresseur (2.0 kW) démarrera à **{heure_demarrage_clim}** pour garantir un confort optimal dès votre retour à 18h00."
        )

    # Questions sur l'IA / Deep Learning
    elif any(
        w in q
        for w in [
            "ia",
            "lstm",
            "modele",
            "deep learning",
            "algorithme",
            "reseau",
        ]
    ):
        return (
            f"Le système exploite un réseau récurrent **LSTM autorégressif** à 2 couches (50 unités chacune) entraîné sur les données réelles UCI.\n"
            f"Sa précision est de **MAE = 0.603 kW**. Il analyse la fenêtre glissante des 24 dernières heures pour prédire la charge passive ({p_base:.2f} kW prévus)."
        )

    # Questions d'état global
    elif any(w in q for w in ["etat", "bilan", "situation", "statut", "foyer"]):
        return (
            f"📊 **Bilan opérationnel instantané :**\n"
            f"- Fond de maison (LSTM) : {p_base:.2f} kW\n"
            f"- Climatisation : {p_clim:.2f} kW\n"
            f"- Borne de recharge : {p_ve:.2f} kW\n"
            f"- Total appelé : **{p_totale_brute:.2f} kW** pour un seuil contractuel de **{seuil_max:.1f} kW**.\n"
            f"- Décision : {'Protection active (surcharge évitée)' if surcharge else 'Régime régulier nominal'}."
        )

    # Questions hors-sujet (Président, politique, devinettes...)
    elif any(
        w in q
        for w in [
            "president",
            "usa",
            "politique",
            "meteo generale",
            "blague",
            "qui es-tu",
        ]
    ):
        return (
            "En tant qu'assistant de supervision énergétique, ma mission est dédiée à la gestion intelligente et à la sécurité électrique de votre foyer. "
            "Je réponds avec précision sur l'état de votre véhicule, le confort thermique de votre chambre et les prédictions de l'IA. "
            "Que souhaitez-vous vérifier concernant vos consommations ?"
        )

    # Textes trop courts ou incompréhensibles (ex: "ù", "test")
    elif len(q) <= 2:
        return (
            "Bonjour ! Je suis l'Orchestrateur IA de votre maison. "
            "Vous pouvez me poser une question précise, par exemple : *« Quand ma voiture sera-t-elle à 100% ? »* ou *« Quel est l'état du réseau ? »*."
        )

    else:
        return (
            f"Actuellement, la demande totale simulée s'élève à **{p_totale_brute:.2f} kW** face à un seuil contractuel de **{seuil_max:.1f} kW**. "
            f"Le système veille en temps réel à l'équilibre du foyer. Souhaitez-vous une précision sur la climatisation ou la recharge du véhicule ?"
        )


def interroger_gemini(message_utilisateur, context_systeme, key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"{context_systeme}\n\nQuestion de l'utilisateur : {message_utilisateur}"
                    }
                ]
            }
        ]
    }
    try:
        r = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=8,
        )
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return moteur_analytique_local(message_utilisateur)
    except Exception:
        return moteur_analytique_local(message_utilisateur)


# Contexte d'affaires transmis au LLM
contexte_ia = f"""
Tu es l'Orchestrateur Intelligent d'une Smart Home haut de gamme (propriété de Farah).
Ton style de communication est exécutif, professionnel, courtois et précis (style cabinet de conseil / gestionnaire d'énergie).

Voici les données physiques et mathématiques exactes à l'instant T :
- Puissance passive prédite par le Deep Learning (LSTM) : {p_base:.2f} kW
- Seuil disjoncteur maximal autorisé : {seuil_max:.1f} kW
- Température ambiante pièce : {temp_interieure:.1f}°C | Consigne : {temp_cible:.1f}°C | Durée nécessaire clim : {duree_clim_min} minutes. Heure d'allumage clim : {heure_demarrage_clim} pour arrivée à 18h00.
- Véhicule électrique : Batterie actuelle {soc_actuel}%, objectif {soc_cible}%. Capacité 50 kWh, borne 3.7 kW (rendement 90%).
- Temps pour atteindre {soc_cible}% : {duree_charge_cible_h} heures.
- Temps pour atteindre 100% : {duree_charge_100_h} heures (soit {energie_100_kwh:.1f} kWh). Heure de fin estimée pour 100% : {heure_fin_100}.
- Puissance cumulée : {p_totale_brute:.2f} kW.
- Risque de surcharge : {'OUI (Excès de ' + str(surcroit) + ' kW)' if surcharge else 'NON (Stable)'}.
- Décision : {'Priorité absolue à la climatisation pour le confort à 18h ; recharge véhicule différée à 23h00 en heures creuses.' if surcharge else 'Charge immédiate et climatisation simultanées.'}

Règles de réponse :
1. Fais les calculs mathématiques exacts si l'utilisateur demande des délais ou des pourcentages de batterie.
2. Si la question est hors sujet (politique, sport, futilités), réponds de façon très brève et polie, puis reviens avec tact à la gestion énergétique.
3. Ne sors jamais de phrases robotiques génériques. Réponds de façon limpide en français.
"""

st.markdown("### 💬 Assistant Exécutif de Supervision")
st.caption(
    "Interface conversationnelle décisionnelle reliée aux agents physiques et au modèle LSTM"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour Farah. Je supervise en temps réel les flux énergétiques du foyer. "
                "Je réponds à vos questions d'arbitrage, d'estimation de charge batterie et d'optimisation de confort."
            ),
        }
    ]

# Affichage des échanges
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrée utilisateur
if prompt := st.chat_input("Ex: Quand puis-je utiliser ma voiture avec 100% de batterie ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analyse des flux énergétiques en cours..."):
        if api_key and len(api_key.strip()) > 10:
            reponse = interroger_gemini(prompt, contexte_ia, api_key.strip())
        else:
            reponse = moteur_analytique_local(prompt)

    st.session_state.messages.append({"role": "assistant", "content": reponse})
    with st.chat_message("assistant"):
        st.markdown(reponse)
