
from datetime import datetime, timedelta
import re

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Maison & Énergie",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --background: #F6F5F1;
    --white: #FFFFFF;
    --text: #263548;
    --muted: #788493;
    --border: #E8E5DE;
    --green: #557C65;
    --red: #B75C52;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

/* Mise en page compacte sur ordinateur */
@media (min-width: 900px) {
    .main .block-container {
        max-width: 100% !important;
        padding: 0.7rem 1.3rem 0.3rem 1.3rem !important;
    }

    [data-testid="stAppViewContainer"] {
        overflow: hidden;
    }
}

.block-container {
    padding-top: 0.7rem !important;
    padding-bottom: 0.4rem !important;
}

/* Barre latérale */
section[data-testid="stSidebar"] {
    background: #29394C;
}

section[data-testid="stSidebar"] * {
    color: #F8F7F4 !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #D9E0E7 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #526176;
}

/* Titres */
h1, h2, h3 {
    color: var(--text) !important;
    letter-spacing: -0.025em;
}

/* Cartes */
div[data-testid="stMetric"] {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.55rem 0.8rem;
    min-height: 85px;
}

div[data-testid="stMetric"] label {
    color: var(--muted) !important;
    font-size: 0.73rem !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--white);
    border: 1px solid var(--border) !important;
    border-radius: 13px !important;
    padding: 0.7rem 0.8rem !important;
}

/* En-tête */
.page-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0;
}

.page-description {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 0.1rem;
    margin-bottom: 0.55rem;
}

/* Textes des cartes */
.section-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--text);
}

.section-description {
    font-size: 0.75rem;
    color: var(--muted);
    margin-bottom: 0.4rem;
}

.decision {
    background: #F7F6F2;
    border-radius: 9px;
    padding: 0.65rem;
    font-size: 0.82rem;
    line-height: 1.45;
    color: #536174;
}

.good {
    color: var(--green);
    font-weight: 700;
    font-size: 0.78rem;
}

.warning {
    color: var(--red);
    font-weight: 700;
    font-size: 0.78rem;
}

/* Chat */
.stChatMessage {
    background: #F8F7F4 !important;
    border: 1px solid #ECE9E3 !important;
    border-radius: 10px !important;
    padding: 0.4rem 0.65rem !important;
    margin-bottom: 0.3rem !important;
}

.stChatMessage p {
    font-size: 0.82rem !important;
    line-height: 1.4 !important;
}

[data-testid="stChatInput"] {
    margin-top: 0.35rem;
}

[data-testid="stChatInput"] textarea {
    background: white !important;
    border-radius: 10px !important;
}

.small-note {
    font-size: 0.68rem;
    color: #8A929C;
    margin-top: 0.35rem;
}

/* Réduction des espaces */
div[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}

@media (max-width: 899px) {
    .block-container {
        padding: 0.8rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. CHARGEMENT DU MODÈLE LSTM
# ============================================================

@st.cache_resource
def charger_ia():
    try:
        model = load_model("modele_lstm_energie.keras")
        scaler_obj = joblib.load("scaler.pkl")
        return model, scaler_obj
    except Exception:
        return None, None


modele, scaler = charger_ia()


def predire_base_foyer():
    try:
        if modele is None or scaler is None:
            return 2.55

        df = pd.read_csv(
            "consommation_horaire_propre.csv",
            index_col="datetime"
        )

        sample = df["Global_active_power"].iloc[100:124].values

        if len(sample) != 24:
            return 2.55

        scaled = scaler.transform(sample.reshape(-1, 1))

        prediction = modele.predict(
            scaled.reshape(1, 24, 1),
            verbose=0
        )

        return round(
            float(scaler.inverse_transform(prediction)[0][0]),
            2
        )

    except Exception:
        return 2.55


# ============================================================
# 3. PARAMÈTRES DU FOYER
# ============================================================

with st.sidebar:
    st.markdown("## Paramètres du foyer")
    st.caption("Ajustez les valeurs pour actualiser la simulation.")
    st.divider()

    seuil_max = st.slider(
        "Puissance souscrite (kW)",
        3.0, 12.0, 6.0, 0.5
    )

    temp_interieure = st.slider(
        "Température actuelle (°C)",
        20.0, 38.0, 30.0, 0.5
    )

    temp_cible = st.slider(
        "Température souhaitée (°C)",
        18.0, 26.0, 21.0, 0.5
    )

    soc_actuel = st.slider(
        "Batterie actuelle (%)",
        5, 100, 35, 5
    )

    soc_cible = st.slider(
        "Objectif de charge (%)",
        50, 100, 80, 5
    )

    st.divider()

    st.caption(
        "Hypothèses : batterie 50 kWh, "
        "rendement 90 %, borne 3,7 kW."
    )


# ============================================================
# 4. CALCULS
# ============================================================

p_base = predire_base_foyer()

# Climatisation
delta_t = max(0, temp_interieure - temp_cible)

duree_clim_min = (
    round(delta_t / 0.20)
    if delta_t > 0 else 0
)

p_clim = 2.0 if duree_clim_min > 0 else 0.0

heure_demarrage_clim = (
    datetime.strptime("18:00", "%H:%M")
    - timedelta(minutes=duree_clim_min)
).strftime("%H:%M")


# Recharge du véhicule
capa_kwh = 50.0
puiss_borne = 3.7
rendement = 0.90

energie_cible_kwh = (
    capa_kwh
    * max(0, soc_cible - soc_actuel)
    / 100
    / rendement
)

duree_charge_cible_h = (
    energie_cible_kwh / puiss_borne
    if energie_cible_kwh > 0 else 0
)

energie_100_kwh = (
    capa_kwh
    * max(0, 100 - soc_actuel)
    / 100
    / rendement
)

duree_charge_100_h = (
    energie_100_kwh / puiss_borne
    if energie_100_kwh > 0 else 0
)

p_ve = (
    puiss_borne
    if duree_charge_cible_h > 0 else 0
)


# Puissance totale
p_totale_brute = round(
    p_base + p_clim + p_ve,
    2
)

surcharge = p_totale_brute > seuil_max

surcroit = (
    round(p_totale_brute - seuil_max, 2)
    if surcharge else 0
)

heure_debut_charge = "23:00" if surcharge else "18:00"

minutes_100 = round(duree_charge_100_h * 60)

heure_fin_100_dt = (
    datetime.strptime(heure_debut_charge, "%H:%M")
    + timedelta(minutes=minutes_100)
)

heure_fin_100 = heure_fin_100_dt.strftime("%H:%M")

if heure_fin_100_dt.day > 1:
    heure_fin_100 += " (lendemain)"

p_apres_arbitrage = round(p_base + p_clim, 2)


# ============================================================
# 5. CHATBOT LOCAL
# ============================================================

def repondre_question(question):

    q = question.lower().strip()

    # Normalisation des apostrophes
    q = q.replace("'", " ").replace("’", " ")

    # Salutations
    if any(mot in q for mot in [
        "bonjour", "salut", "hello",
        "bonsoir", "coucou"
    ]):
        return (
            "Bonjour ! Comment allez-vous ? "
            "Dites-moi ce que vous souhaitez comprendre "
            "et je vous aiderai avec plaisir."
        )

    # Premier président de Tunisie
    if any(mot in q for mot in [
        "premier president",
        "president de tunisie",
        "president tunisien",
        "bourguiba",
        "tunisie"
    ]):
        return (
            "Le premier président de la République tunisienne "
            "est **Habib Bourguiba**. "
            "La République tunisienne a été proclamée "
            "le 25 juillet 1957."
        )

    # Recharge à 100 %
    if (
        ("100" in q or "plein" in q)
        and any(mot in q for mot in [
            "voiture", "batterie",
            "charge", "recharge", "charger"
        ])
    ):
        return (
            f"Pour passer de **{soc_actuel}% à 100%**, "
            f"il faut fournir environ **{energie_100_kwh:.1f} kWh**, "
            f"en tenant compte du rendement de 90%.\n\n"
            f"À 3,7 kW, cela représente environ "
            f"**{duree_charge_100_h:.1f} heures**.\n\n"
            f"- Début de charge : **{heure_debut_charge}**\n"
            f"- Fin estimée : **{heure_fin_100}**."
        )

    # Questions sur la voiture
    if any(mot in q for mot in [
        "voiture", "batterie",
        "charge", "recharge", "charger",
        "vehicule", "véhicule"
    ]):
        return (
            f"La batterie est actuellement à **{soc_actuel}%** "
            f"et votre objectif est **{soc_cible}%**.\n\n"
            f"Il faut environ **{energie_cible_kwh:.1f} kWh**, "
            f"soit **{duree_charge_cible_h:.1f} heures** "
            f"sur une borne de 3,7 kW.\n\n"
            f"Le début de charge proposé est **{heure_debut_charge}**."
        )

    # Climatisation
    if any(mot in q for mot in [
        "clim", "chambre", "temperature",
        "température", "chaleur", "froid"
    ]):
        return (
            f"La température actuelle est de "
            f"**{temp_interieure}°C**.\n\n"
            f"Votre température souhaitée est de "
            f"**{temp_cible}°C**.\n\n"
            f"Le temps estimé est de **{duree_clim_min} minutes**. "
            f"Le démarrage proposé est à **{heure_demarrage_clim}** "
            f"pour atteindre la consigne vers 18h00."
        )

    # Explication de kW et kWh
    if any(mot in q for mot in [
        "kw", "puissance", "consommation",
        "energie", "énergie", "8.25", "8,25"
    ]):

        nombre = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(?:kw)?",
            q
        )

        if "8.25" in q or "8,25" in q:
            return (
                "Dans votre simulation, **8,25 kW** représente "
                "la puissance totale demandée lorsque la maison, "
                "la climatisation et la recharge de la voiture "
                "fonctionnent simultanément.\n\n"
                f"Votre contrat est de **{seuil_max:.1f} kW**. "
                f"Le dépassement est donc de "
                f"**{max(0, 8.25 - seuil_max):.2f} kW**."
            )

        if any(mot in q for mot in [
            "signifie", "veut dire",
            "c est quoi", "explique",
            "represente", "représente"
        ]):

            if nombre:
                valeur = float(
                    nombre.group(1).replace(",", ".")
                )

                return (
                    f"**{valeur:.2f} kW**, c'est une mesure "
                    "de puissance électrique.\n\n"
                    "Le kW indique la puissance utilisée "
                    "à un instant donné.\n\n"
                    "Par exemple, un appareil de 2 kW "
                    "qui fonctionne pendant une heure "
                    "consomme environ 2 kWh.\n\n"
                    "Le kWh représente donc l'énergie consommée "
                    "pendant une durée."
                )

        return (
            f"La puissance totale estimée dans votre foyer "
            f"est de **{p_totale_brute:.2f} kW**.\n\n"
            f"Votre limite contractuelle est de "
            f"**{seuil_max:.1f} kW**.\n\n"
            "Le kW mesure une puissance, tandis que le kWh "
            "mesure une quantité d'énergie consommée."
        )

    # Surcharge électrique
    if any(mot in q for mot in [
        "surcharge", "disjoncteur",
        "contrat", "abonnement",
        "depasse", "dépass", "limite"
    ]):
        if surcharge:
            return (
                f"La puissance demandée est de "
                f"**{p_totale_brute:.2f} kW**, "
                f"alors que votre contrat autorise "
                f"**{seuil_max:.1f} kW**.\n\n"
                f"Le dépassement serait de **{surcroit:.2f} kW**.\n\n"
                "Pour éviter cela, le système propose "
                "de reporter la recharge de la voiture à 23h00."
            )

        return (
            f"Bonne nouvelle ! La puissance demandée est de "
            f"**{p_totale_brute:.2f} kW**, "
            f"ce qui reste sous votre limite de "
            f"**{seuil_max:.1f} kW**.\n\n"
            "Les équipements peuvent fonctionner simultanément "
            "selon cette simulation."
        )

    # LSTM
    if any(mot in q for mot in [
        "lstm", "modele", "modèle",
        "prediction", "prédiction",
        "ia", "deep learning"
    ]):
        origine = (
            "votre modèle LSTM"
            if modele is not None
            else "la valeur de démonstration utilisée en attendant le modèle"
        )

        return (
            f"La prévision de consommation de fond est de "
            f"**{p_base:.2f} kW**.\n\n"
            f"Elle provient de {origine}. "
            "Cette prévision sert de base aux calculs "
            "de consommation du foyer."
        )

    # Questions sur la maison
    if any(mot in q for mot in [
        "maison", "foyer", "ma consommation",
        "fonctionnement", "simulation"
    ]):
        return (
            f"Actuellement, la consommation de fond est estimée "
            f"à **{p_base:.2f} kW**.\n\n"
            f"La climatisation demande **{p_clim:.2f} kW** "
            f"et la recharge **{p_ve:.2f} kW**.\n\n"
            f"La puissance totale atteint donc "
            f"**{p_totale_brute:.2f} kW**."
        )

    # Réponse par défaut
    return (
        "Je peux vous aider à comprendre les résultats "
        "de votre maison.\n\n"
        "Vous pouvez par exemple me demander :\n\n"
        "- Que signifie 8,25 kW ?\n"
        "- Quand ma voiture sera-t-elle chargée à 100 % ?\n"
        "- Pourquoi la recharge est-elle reportée à 23h ?\n"
        "- Combien de temps faut-il pour refroidir la chambre ?\n"
        "- Qu'est-ce que le modèle LSTM ?\n\n"
        "Vous pouvez aussi me poser une question générale "
        "sur ces sujets."
    )


# ============================================================
# 6. INTERFACE PRINCIPALE
# ============================================================

st.markdown("""
<div>
    <div class="page-title">Votre maison, en équilibre</div>
    <div class="page-description">
        Un aperçu simple de votre consommation, du confort et de la recharge.
    </div>
</div>
""", unsafe_allow_html=True)


# Indicateurs
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Consommation prévue",
    f"{p_base:.2f} kW",
    "Prévision LSTM"
)

c2.metric(
    "Climatisation",
    f"{p_clim:.2f} kW",
    f"{duree_clim_min} min estimées"
)

c3.metric(
    "Recharge voiture",
    f"{p_ve:.2f} kW",
    f"{duree_charge_cible_h:.1f} h vers {soc_cible}%"
)

c4.metric(
    "Puissance totale",
    f"{p_totale_brute:.2f} kW",
    (
        f"+{surcroit:.2f} kW au-dessus du contrat"
        if surcharge
        else f"{seuil_max-p_totale_brute:.2f} kW disponibles"
    ),
    delta_color="inverse"
)


# ============================================================
# 7. TABLEAU DE BORD ET CHAT
# ============================================================

col_gauche, col_droite = st.columns(
    [1.05, 1],
    gap="medium"
)


# ------------------------------------------------------------
# GAUCHE : PUISSANCE ET DÉCISION
# ------------------------------------------------------------

with col_gauche:

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">Puissance électrique</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'Votre consommation comparée à la puissance souscrite.'
            '</div>',
            unsafe_allow_html=True
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=p_totale_brute,
                number={
                    "valueformat": ".2f",
                    "suffix": " kW",
                    "font": {
                        "size": 27,
                        "color": "#263548"
                    }
                },
                gauge={
                    "axis": {
                        "range": [0, max(12, seuil_max)]
                    },
                    "bar": {
                        "color": "#B75C52" if surcharge else "#557C65"
                    },
                    "bgcolor": "#F1EFEA",
                    "borderwidth": 0,
                    "steps": [
                        {
                            "range": [0, seuil_max],
                            "color": "#E7EDE7"
                        },
                        {
                            "range": [seuil_max, max(12, seuil_max)],
                            "color": "#F5E5E1"
                        }
                    ],
                    "threshold": {
                        "line": {
                            "color": "#263548",
                            "width": 3
                        },
                        "thickness": 0.8,
                        "value": seuil_max
                    }
                }
            )
        )

        fig.update_layout(
            height=175,
            margin=dict(l=10, r=10, t=5, b=0),
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        if surcharge:

            st.markdown(
                f"""
                <div class="warning">
                    Dépassement potentiel : {surcroit:.2f} kW
                </div>
                <div class="decision">
                    La puissance simultanée atteint
                    <strong>{p_totale_brute:.2f} kW</strong>,
                    pour une limite de
                    <strong>{seuil_max:.1f} kW</strong>.<br><br>
                    La recharge du véhicule est proposée à
                    <strong>23h00</strong>.<br>
                    La puissance de la maison et de la climatisation
                    serait alors de
                    <strong>{p_apres_arbitrage:.2f} kW</strong>,
                    avec une marge de
                    <strong>{seuil_max-p_apres_arbitrage:.2f} kW</strong>.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="good">
                    Tout peut fonctionner en même temps
                </div>
                <div class="decision">
                    La puissance totale est de
                    <strong>{p_totale_brute:.2f} kW</strong>.
                    Elle respecte votre contrat de
                    <strong>{seuil_max:.1f} kW</strong>.<br><br>
                    Il reste une marge de
                    <strong>{seuil_max-p_totale_brute:.2f} kW</strong>.
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div class="small-note">
                Climatisation : démarrage estimé à {heure_demarrage_clim}.
                Recharge : début proposé à {heure_debut_charge}.
            </div>
            """,
            unsafe_allow_html=True
        )


# ------------------------------------------------------------
# DROITE : CHATBOT
# ------------------------------------------------------------

with col_droite:

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">Une question ?</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'Je suis là pour vous aider à comprendre votre consommation '
            'et les décisions du système.'
            '</div>',
            unsafe_allow_html=True
        )

        # Historique du chatbot
        if "messages" not in st.session_state:

            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Bonjour ! Je peux vous aider à comprendre "
                        "la consommation, la climatisation, "
                        "la recharge de votre voiture ou "
                        "les résultats de la simulation. "
                        "Que souhaitez-vous savoir ?"
                    )
                }
            ]

        # Affichage des derniers messages sans zone de défilement interne
        for message in st.session_state.messages[-4:]:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Entrée utilisateur
        question = st.chat_input(
            "Écrivez votre question ici..."
        )

        if question:

            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            reponse = repondre_question(question)

            st.session_state.messages.append({
                "role": "assistant",
                "content": reponse
            })

            st.rerun()


# ============================================================
# 8. NOTE DE FIN
# ============================================================

st.markdown(
    """
    <div class="small-note">
        Les résultats sont des estimations basées sur les paramètres
        et hypothèses du projet, et non des mesures électriques en temps réel.
    </div>
    """,
    unsafe_allow_html=True
)
