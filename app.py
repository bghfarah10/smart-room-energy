from datetime import datetime, timedelta
import os
import re

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from tensorflow.keras.models import load_model


# ============================================================
# 1. CONFIGURATION ET STYLE
# ============================================================
st.set_page_config(
    page_title="Smart Home Energy Orchestration",
    page_icon="SH",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #F8F6F1;
        --navy: #0F1E36;
        --blue: #1D3557;
        --muted: #64748B;
        --border: #E5E0D8;
        --card: #FFFFFF;
        --green: #286348;
        --red: #A83D35;
        --sand: #F1ECE3;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--navy);
    }

    .block-container {
        max-width: 100% !important;
        padding-top: 0.9rem !important;
        padding-bottom: 0.6rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
    }

    section[data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid #263754;
    }

    section[data-testid="stSidebar"] * {
        color: #F8F6F1 !important;
    }

    section[data-testid="stSidebar"] input {
        color: var(--navy) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #F8F6F1 !important;
    }

    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 11px;
        padding: 10px 13px;
        box-shadow: 0 2px 7px rgba(15, 30, 54, 0.035);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted) !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.035em;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--navy) !important;
        font-size: 1.35rem !important;
        font-weight: 750 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card);
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 11px 13px !important;
        box-shadow: 0 2px 8px rgba(15, 30, 54, 0.025);
    }

    .page-heading {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }

    .page-title {
        color: var(--navy);
        font-size: 1.35rem;
        line-height: 1.25;
        font-weight: 800;
        margin: 0;
    }

    .page-subtitle {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 3px;
    }

    .user-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 11px;
        padding: 13px 14px;
        min-height: 112px;
        box-shadow: 0 2px 7px rgba(15, 30, 54, 0.025);
    }

    .card-category {
        color: #54708D;
        font-size: 0.69rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .card-main {
        color: var(--navy);
        font-size: 1.28rem;
        font-weight: 800;
        line-height: 1.25;
        margin-bottom: 7px;
    }

    .card-sub {
        display: inline-block;
        background: var(--sand);
        color: #586779;
        border-radius: 6px;
        padding: 4px 7px;
        font-size: 0.72rem;
        line-height: 1.35;
    }

    .panel-title {
        color: var(--navy);
        font-size: 0.93rem;
        font-weight: 750;
        margin-bottom: 6px;
    }

    .status-line {
        font-size: 0.82rem;
        line-height: 1.5;
        border-radius: 8px;
        padding: 9px 11px;
        margin-top: 4px;
    }

    .status-good {
        color: #214B38;
        background: #F0F6F2;
        border-left: 3px solid var(--green);
    }

    .status-warn {
        color: #74342F;
        background: #FBF1EF;
        border-left: 3px solid var(--red);
    }

    .stChatMessage {
        background: #FAF8F5 !important;
        border: 1px solid #EAE5DD !important;
        border-radius: 9px !important;
        padding: 7px 10px !important;
        margin-bottom: 5px !important;
        font-size: 0.86rem !important;
    }

    [data-testid="stChatInput"] {
        padding-top: 4px;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        .page-heading {
            align-items: flex-start;
            flex-direction: column;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. CHARGEMENT DU MODELE LSTM ET DES DONNEES
# ============================================================
@st.cache_resource
def charger_ia():
    """Charge le modèle et le scaler. Retourne (None, None) si absents."""
    try:
        model = load_model("modele_lstm_energie.keras")
        scaler_obj = joblib.load("scaler.pkl")
        return model, scaler_obj
    except Exception:
        return None, None


@st.cache_data
def charger_donnees():
    try:
        return pd.read_csv("consommation_horaire_propre.csv", index_col="datetime")
    except Exception:
        return None


modele, scaler = charger_ia()
df_energie = charger_donnees()


def predire_base_foyer():
    """
    Prévoit la puissance de base à partir des 24 premières valeurs horaires
    disponibles à partir de l'index 100, comme dans le jeu de données du projet.
    """
    if modele is None or scaler is None or df_energie is None:
        return 2.55, False

    try:
        if "Global_active_power" not in df_energie.columns:
            raise ValueError("Colonne Global_active_power introuvable.")

        valeurs = df_energie["Global_active_power"].iloc[100:124].dropna().values
        if len(valeurs) != 24:
            raise ValueError("Il faut 24 valeurs horaires valides pour la prédiction.")

        valeurs_scaled = scaler.transform(valeurs.reshape(-1, 1))
        entree = valeurs_scaled.reshape(1, 24, 1)
        prediction_scaled = modele.predict(entree, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
        return max(0.0, round(float(prediction), 2)), True
    except Exception:
        return 2.55, False


p_base, prediction_reelle = predire_base_foyer()


# ============================================================
# 3. PARAMETRES UTILISATEUR
# ============================================================
with st.sidebar:
    st.markdown("### Paramètres du foyer")
    st.caption("Ajustez les valeurs pour actualiser la simulation.")

    cle_secrete = ""
    try:
        cle_secrete = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        cle_secrete = os.getenv("GEMINI_API_KEY", "")

    api_key = st.text_input(
        "Clé API Gemini (facultatif)",
        value=cle_secrete,
        type="password",
        help="Sans clé API, l'assistant utilise le mode local intégré.",
    )

    st.markdown("---")
    st.markdown("#### Contrat électrique")
    seuil_max = st.slider("Puissance souscrite (kW)", 3.0, 12.0, 6.0, 0.5)

    st.markdown("#### Confort de la chambre")
    temp_interieure = st.slider("Température actuelle (°C)", 20.0, 38.0, 30.0, 0.5)
    temp_cible = st.slider("Température souhaitée (°C)", 18.0, 26.0, 21.0, 0.5)

    st.markdown("#### Véhicule électrique")
    soc_actuel = st.slider("Batterie actuelle (%)", 5, 100, 35, 5)
    soc_cible = st.slider("Objectif de charge (%)", 50, 100, 80, 5)

    st.markdown("---")
    st.caption("Hypothèses de simulation : batterie 50 kWh, rendement de charge 90 %, borne 3,7 kW.")


# ============================================================
# 4. CALCULS ENERGETIQUES ET PLANIFICATION
# ============================================================
# Climatisation : hypothèse simplifiée de 0,20 °C/min.
delta_temp = max(0.0, temp_interieure - temp_cible)
duree_clim_min = int(round(delta_temp / 0.20)) if delta_temp > 0 else 0
p_clim = 2.0 if duree_clim_min > 0 else 0.0

heure_confort = datetime.strptime("18:00", "%H:%M")
heure_demarrage_clim = (
    heure_confort - timedelta(minutes=duree_clim_min)
).strftime("%H:%M")

# Recharge VE.
capacite_batterie_kwh = 50.0
puissance_borne_kw = 3.7
rendement_charge = 0.90

energie_cible_kwh = (
    capacite_batterie_kwh * max(0, soc_cible - soc_actuel) / 100 / rendement_charge
)
duree_charge_cible_h = (
    energie_cible_kwh / puissance_borne_kw if energie_cible_kwh > 0 else 0.0
)

energie_100_kwh = (
    capacite_batterie_kwh * max(0, 100 - soc_actuel) / 100 / rendement_charge
)
duree_charge_100_h = energie_100_kwh / puissance_borne_kw if energie_100_kwh > 0 else 0.0

p_ve = puissance_borne_kw if energie_cible_kwh > 0 else 0.0
p_totale_brute = round(p_base + p_clim + p_ve, 2)
surcharge = p_totale_brute > seuil_max

# Si la charge simultanée dépasse le contrat, la voiture est décalée à 23h.
heure_debut_charge = "23:00" if surcharge else "18:00"
puissance_apres_arbitrage = round(p_base + p_clim, 2) if surcharge else p_totale_brute
marge_apres_arbitrage = round(seuil_max - puissance_apres_arbitrage, 2)

def heure_fin_charge(heure_depart, duree_heures):
    depart = datetime.strptime(heure_depart, "%H:%M")
    arrivee = depart + timedelta(minutes=round(duree_heures * 60))
    suffixe = " (lendemain)" if arrivee.date() > depart.date() else ""
    return arrivee.strftime("%H:%M") + suffixe


heure_fin_cible = heure_fin_charge(heure_debut_charge, duree_charge_cible_h)
heure_fin_100 = heure_fin_charge(heure_debut_charge, duree_charge_100_h)

# Heure indicative de disponibilité à 07h30.
heure_depart_dt = datetime.strptime(heure_debut_charge, "%H:%M")
heure_fin_cible_dt = heure_depart_dt + timedelta(minutes=round(duree_charge_cible_h * 60))
heure_limite = datetime.strptime("07:30", "%H:%M")
if heure_debut_charge == "23:00":
    heure_limite += timedelta(days=1)
voiture_prete_0730 = heure_fin_cible_dt <= heure_limite

# ============================================================
# 5. OUTILS POUR L'ASSISTANT : GEMINI + MODE LOCAL
# ============================================================
def appeler_gemini(message_utilisateur, contexte_systeme, key):
    """
    Essaie plusieurs modèles Gemini compatibles. Le premier modèle disponible
    qui répond correctement est utilisé. Une erreur API déclenche le mode local.
    """
    modeles_a_essayer = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
    ]

    prompt_complet = (
        f"{contexte_systeme}\n\n"
        "Réponds en français, de façon naturelle, précise et concise. "
        "Explique les termes techniques simplement. Ne prétends pas qu'une action "
        "physique a été exécutée : il s'agit d'une simulation de supervision.\n\n"
        f"Question de l'utilisateur : {message_utilisateur}"
    )

    erreurs = []
    for nom_modele in modeles_a_essayer:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{nom_modele}:generateContent?key={key}"
        )
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt_complet}]}]},
                timeout=20,
            )

            if response.status_code == 200:
                contenu = response.json()
                return contenu["candidates"][0]["content"]["parts"][0]["text"], nom_modele

            erreurs.append(f"{nom_modele}: HTTP {response.status_code}")
            # Un 404 signifie souvent que le modèle n'est pas disponible pour cette clé/API.
            if response.status_code not in (400, 404, 429, 500, 503):
                break
        except requests.RequestException as exc:
            erreurs.append(f"{nom_modele}: {exc}")
            break
        except (KeyError, IndexError, TypeError, ValueError):
            erreurs.append(f"{nom_modele}: réponse API inattendue")

    return None, "; ".join(erreurs)


def reponse_locale(question):
    """Assistant local à base de règles, utilisable sans connexion Gemini."""
    q = question.lower().strip()
    q_normalise = (
        q.replace("é", "e").replace("è", "e").replace("ê", "e")
         .replace("à", "a").replace("ù", "u").replace("ç", "c")
    )

    # Compréhension des questions qui contiennent une valeur de puissance.
    valeurs_kw = re.findall(r"(\d+(?:[.,]\d+)?)\s*k\s*w", q_normalise)
    if valeurs_kw and any(mot in q_normalise for mot in [
        "signifie", "signification", "veut dire", "c'est quoi", "correspond",
        "comprendre", "kw", "puissance"
    ]):
        valeur = float(valeurs_kw[0].replace(",", "."))
        energie_une_heure = valeur
        minutes_30 = valeur * 0.5
        comparaison = (
            f"Cette valeur dépasse le plafond configuré de {seuil_max:.1f} kW."
            if valeur > seuil_max
            else f"Cette valeur reste sous le plafond configuré de {seuil_max:.1f} kW."
        )
        return (
            f"{valeur:g} kW signifie une puissance électrique de {valeur:g} kilowatts. "
            f"Cela correspond à une consommation d'énergie de {energie_une_heure:g} kWh "
            f"si cette puissance est maintenue pendant une heure, ou de {minutes_30:g} kWh "
            f"pendant 30 minutes. {comparaison} "
            "Le kW mesure une puissance à un instant donné, tandis que le kWh mesure "
            "une quantité d'énergie consommée sur une durée."
        )

    if any(mot in q_normalise for mot in ["100%", "100 pourcent", "plein", "batterie pleine"]):
        return (
            f"Pour passer de {soc_actuel}% à 100%, il faut fournir environ "
            f"{energie_100_kwh:.1f} kWh, en tenant compte du rendement estimé de "
            f"{rendement_charge:.0%}. À 3,7 kW, cela représente environ "
            f"{duree_charge_100_h:.1f} heures. Avec le planning actuel, la charge "
            f"commencerait à {heure_debut_charge} et se terminerait vers {heure_fin_100}."
        )

    if any(mot in q_normalise for mot in [
        "voiture", "batterie", "recharge", "charger", "vehicule", "autonomie"
    ]):
        if soc_actuel >= soc_cible:
            return (
                f"La batterie est déjà à {soc_actuel}%, soit au niveau de l'objectif "
                f"de {soc_cible}%. Aucune recharge supplémentaire n'est nécessaire "
                "pour atteindre cet objectif."
            )
        return (
            f"La batterie est actuellement à {soc_actuel}% et l'objectif est "
            f"{soc_cible}%. Il faut environ {energie_cible_kwh:.1f} kWh, soit "
            f"{duree_charge_cible_h:.1f} heures sur la borne de 3,7 kW. "
            f"Début prévu : {heure_debut_charge}. Fin estimée : {heure_fin_cible}. "
            f"{'La voiture atteindra l’objectif avant 07h30.' if voiture_prete_0730 else 'Avec les paramètres actuels, l’objectif ne sera pas atteint avant 07h30.'}"
        )

    if any(mot in q_normalise for mot in [
        "clim", "climatisation", "chambre", "temperature", "chaleur", "frais"
    ]):
        if duree_clim_min == 0:
            return (
                f"La température actuelle est de {temp_interieure:.1f} °C. "
                f"Elle est déjà égale ou inférieure à la consigne de {temp_cible:.1f} °C. "
                "La climatisation n'est donc pas nécessaire selon cette simulation."
            )
        return (
            f"La pièce est actuellement à {temp_interieure:.1f} °C et la consigne est "
            f"{temp_cible:.1f} °C. Avec l'hypothèse de refroidissement de 0,20 °C par "
            f"minute, il faut environ {duree_clim_min} minutes. Pour atteindre la "
            f"consigne à 18h00, le démarrage théorique est prévu à {heure_demarrage_clim}."
        )

    if any(mot in q_normalise for mot in [
        "surcharge", "disjoncteur", "puissance totale", "courant", "coupure",
        "puissance appelee", "reseau", "kw"
    ]):
        if surcharge:
            return (
                f"Si les trois équipements fonctionnent en même temps, la puissance "
                f"estimée atteint {p_totale_brute:.2f} kW, au-dessus du plafond de "
                f"{seuil_max:.1f} kW. Le planning proposé décale donc la recharge du "
                f"véhicule à 23h00. La puissance estimée pendant le fonctionnement de "
                f"la maison et de la climatisation est alors {puissance_apres_arbitrage:.2f} kW, "
                f"soit une marge de {marge_apres_arbitrage:.2f} kW."
            )
        return (
            f"La puissance totale estimée est de {p_totale_brute:.2f} kW pour un plafond "
            f"de {seuil_max:.1f} kW. La marge disponible est de "
            f"{seuil_max - p_totale_brute:.2f} kW. Selon les hypothèses de la simulation, "
            "les équipements peuvent fonctionner simultanément sans dépasser le plafond."
        )

    if any(mot in q_normalise for mot in ["lstm", "deep learning", "intelligence artificielle", "modele", "prediction"]):
        statut = "chargé et utilisé" if prediction_reelle else "indisponible : la valeur de secours est utilisée"
        return (
            f"Le modèle LSTM sert à prévoir la consommation de base du foyer à partir "
            f"d'une séquence de 24 valeurs horaires. Dans cette exécution, le modèle est "
            f"{statut}. La prévision affichée est de {p_base:.2f} kW. "
            "Cette valeur représente la charge de base estimée, avant l'ajout de la climatisation et de la recharge du véhicule."
        )

    if any(mot in q_normalise for mot in ["bonjour", "salut", "bonsoir", "hello"]):
        return (
            "Bonjour. Je peux vous renseigner sur la consommation du foyer, la puissance "
            "souscrite, la climatisation, la recharge du véhicule et les décisions de "
            "répartition de la puissance."
        )

    return (
        f"Je peux vous aider à comprendre la situation énergétique actuelle. "
        f"La puissance totale simulée est de {p_totale_brute:.2f} kW, pour une limite "
        f"de {seuil_max:.1f} kW. Vous pouvez me demander par exemple ce que signifie "
        "une valeur en kW, quand lancer la climatisation, combien de temps prendra "
        "la recharge ou pourquoi le système décale la charge."
    )


# ============================================================
# 6. CONTEXTE TRANSMIS A GEMINI
# ============================================================
contexte_ia = f"""
Tu es l'assistant de supervision d'un système domotique énergétique multi-agents.
Tu réponds en français, avec un ton professionnel, naturel et accessible.
Tu peux répondre aux questions générales, mais pour les questions sur la maison,
utilise prioritairement les données ci-dessous. Ne fabrique pas de mesures.
Il s'agit d'une simulation : ne prétends pas avoir réellement allumé un appareil.

DONNEES ACTUELLES :
- Prévision de puissance de base par LSTM : {p_base:.2f} kW.
- Modèle LSTM réellement disponible : {"oui" if prediction_reelle else "non, valeur de secours utilisée"}.
- Puissance souscrite : {seuil_max:.1f} kW.
- Température actuelle : {temp_interieure:.1f} °C.
- Température cible : {temp_cible:.1f} °C.
- Durée estimée de climatisation : {duree_clim_min} minutes.
- Démarrage théorique de la climatisation pour une consigne à 18h : {heure_demarrage_clim}.
- Batterie du véhicule : {soc_actuel}%.
- Objectif de batterie : {soc_cible}%.
- Durée estimée de recharge vers l'objectif : {duree_charge_cible_h:.2f} heures.
- Durée estimée de recharge vers 100% : {duree_charge_100_h:.2f} heures.
- Puissance totale si tout fonctionne en même temps : {p_totale_brute:.2f} kW.
- Dépassement du plafond en fonctionnement simultané : {"oui" if surcharge else "non"}.
- Début de charge planifié : {heure_debut_charge}.
- Fin estimée de charge vers l'objectif : {heure_fin_cible}.
- Fin estimée de charge à 100% : {heure_fin_100}.

EXPLICATIONS A RESPECTER :
- Le kW est une unité de puissance. Le kWh est une unité d'énergie.
- Une puissance de X kW maintenue pendant une heure correspond à X kWh.
- La puissance totale est une estimation calculée à partir des hypothèses du projet.
- Si la puissance simultanée dépasse la puissance souscrite, le système propose de
  décaler la recharge du véhicule à 23h00.
- Présente les heures et durées de façon claire. Si une donnée dépend d'une hypothèse,
  précise qu'il s'agit d'une estimation.
"""


# ============================================================
# 7. INTERFACE PRINCIPALE : CARTES + JAUGE + CHAT
# ============================================================
st.markdown(
    """
    <div class="page-heading">
        <div>
            <div class="page-title">Smart Home Energy Orchestration</div>
            <div class="page-subtitle">Supervision énergétique prédictive et coordination multi-agents</div>
        </div>
        <div style="font-size:0.76rem;color:#64748B;">Tableau de bord de supervision</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not prediction_reelle:
    st.info(
        "Le modèle LSTM ou ses fichiers de données n'ont pas été chargés. "
        "La valeur de secours de 2,55 kW est utilisée pour la démonstration. "
        "Vérifiez la présence des trois fichiers du modèle dans le même dossier que app.py."
    )

# Cartes orientées utilisateur.
c1, c2, c3, c4 = st.columns(4, gap="small")

with c1:
    st.markdown(
        f"""
        <div class="user-card">
            <div class="card-category">Consommation de la maison</div>
            <div class="card-main">{"Activité estimée" if prediction_reelle else "Estimation de secours"}</div>
            <div class="card-sub">Prévision LSTM : {p_base:.2f} kW</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="user-card">
            <div class="card-category">Confort de la chambre</div>
            <div class="card-main">{duree_clim_min} min nécessaires</div>
            <div class="card-sub">Démarrage estimé à {heure_demarrage_clim} pour 18h00</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    statut_voiture = "Objectif atteint" if soc_actuel >= soc_cible else (
        "Prête avant 07h30" if voiture_prete_0730 else f"Fin vers {heure_fin_cible}"
    )
    st.markdown(
        f"""
        <div class="user-card">
            <div class="card-category">Recharge du véhicule</div>
            <div class="card-main">{statut_voiture}</div>
            <div class="card-sub">{duree_charge_cible_h:.1f} h estimées pour atteindre {soc_cible}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    statut_reseau = "Répartition nécessaire" if surcharge else "Réseau équilibré"
    couleur_statut = "#A83D35" if surcharge else "#286348"
    detail_reseau = (
        f"Recharge reportée à {heure_debut_charge}"
        if surcharge else f"Marge disponible : {seuil_max - p_totale_brute:.2f} kW"
    )
    st.markdown(
        f"""
        <div class="user-card" style="border-left:4px solid {couleur_statut};">
            <div class="card-category">Sécurité électrique</div>
            <div class="card-main" style="color:{couleur_statut};">{statut_reseau}</div>
            <div class="card-sub">{detail_reseau}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

col_supervision, col_assistant = st.columns([1, 1.15], gap="medium")

# -------------------- JAUGE ET DECISION --------------------
with col_supervision:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Équilibre de charge du foyer</div>', unsafe_allow_html=True)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=p_totale_brute,
                number={
                    "valueformat": ".2f",
                    "suffix": " kW",
                    "font": {"size": 29, "color": "#0F1E36", "family": "Inter"},
                },
                gauge={
                    "axis": {
                        "range": [0, max(12, seuil_max + 1)],
                        "tickwidth": 1,
                        "tickcolor": "#AAB4C2",
                        "nticks": 7,
                    },
                    "bar": {"color": "#A83D35" if surcharge else "#1D3557"},
                    "bgcolor": "#F1ECE3",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, seuil_max], "color": "#E8E4DC"},
                        {"range": [seuil_max, max(12, seuil_max + 1)], "color": "#F5DEDA"},
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
            height=185,
            margin=dict(l=12, r=12, t=5, b=5),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter"},
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.caption(
            f"Puissance souscrite : {seuil_max:.1f} kW · "
            f"Puissance appelée avant arbitrage : {p_totale_brute:.2f} kW"
        )

        if surcharge:
            st.markdown(
                f"""
                <div class="status-line status-warn">
                    <strong>Dépassement potentiel : {p_totale_brute - seuil_max:.2f} kW.</strong><br>
                    La consommation simultanée dépasserait la puissance souscrite.
                    Le système propose de reporter la recharge du véhicule à 23h00.
                    La puissance estimée pendant le fonctionnement de la maison et de la
                    climatisation serait alors de {puissance_apres_arbitrage:.2f} kW,
                    soit une marge de {marge_apres_arbitrage:.2f} kW.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="status-line status-good">
                    <strong>Fonctionnement simultané possible selon la simulation.</strong><br>
                    La puissance estimée ({p_totale_brute:.2f} kW) reste sous le plafond
                    de {seuil_max:.1f} kW. La marge disponible est de
                    {seuil_max - p_totale_brute:.2f} kW.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "Les résultats sont des estimations basées sur les paramètres et hypothèses "
            "du projet, et non des mesures électriques en temps réel."
        )

# -------------------- ASSISTANT CONVERSATIONNEL --------------------
with col_assistant:
    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
                <div class="panel-title" style="margin:0;">Assistant de supervision</div>
                <div style="font-size:0.72rem;color:#64748B;">Mode Gemini ou local</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Bonjour. Je suis l'assistant de supervision énergétique. "
                        "Vous pouvez me poser des questions sur la consommation, la "
                        "puissance souscrite, la climatisation ou la recharge du véhicule."
                    ),
                }
            ]

        # Zone de conversation avec défilement interne, pour limiter la hauteur de la page.
        chat_box = st.container(height=270)

        for message in st.session_state.messages[-12:]:
            with chat_box.chat_message(message["role"]):
                st.markdown(message["content"])

        if not api_key:
            st.caption("Assistant local actif : aucune clé API n'est nécessaire.")
        else:
            st.caption("Clé API renseignée : Gemini sera utilisé si un modèle est disponible.")

# Le champ de saisie Streamlit est placé après les colonnes afin de rester utilisable.
prompt = st.chat_input("Écrivez votre question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    if api_key:
        with st.spinner("Préparation de la réponse..."):
            reponse_gemini, info_gemini = appeler_gemini(prompt, contexte_ia, api_key)

        if reponse_gemini:
            reponse = reponse_gemini
            st.session_state.mode_assistant = f"Gemini ({info_gemini})"
        else:
            reponse = reponse_locale(prompt)
            reponse += (
                "\n\n*Le service Gemini n'a pas répondu. La réponse est fournie "
                "par l'assistant local.*"
            )
            st.session_state.mode_assistant = "Mode local"
    else:
        reponse = reponse_locale(prompt)
        st.session_state.mode_assistant = "Mode local"

    st.session_state.messages.append({"role": "assistant", "content": reponse})
    st.rerun()

# Petit indicateur de disponibilité sous l'interface.
mode_actuel = st.session_state.get(
    "mode_assistant",
    "Gemini disponible sous réserve de compatibilité" if api_key else "Mode local",
)
st.caption(f"Assistant : {mode_actuel}")
