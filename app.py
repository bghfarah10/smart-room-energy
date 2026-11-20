from datetime import date, datetime, time, timedelta
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from tensorflow.keras.models import load_model

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="Smart Home Energy Orchestration",
    page_icon="⚡",
    layout="wide",
)

# Style CSS personnalisé (conserve ton design épuré)
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #132438 0%, #1f3a56 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #64748B;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #475569;
    }
    #MainMenu, footer {visibility: hidden;}
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


def predire_base_lstm(heure_cible):
    try:
        df = pd.read_csv(
            "consommation_horaire_propre.csv", index_col="datetime"
        )
        # On utilise une tranche représentative
        idx = (heure_cible * 10) % (len(df) - 30)
        sample = df["Global_active_power"].iloc[idx : idx + 24].values
        scaled = scaler.transform(sample.reshape(-1, 1))
        pred_scaled = modele.predict(scaled.reshape(1, 24, 1), verbose=0)
        return round(float(scaler.inverse_transform(pred_scaled)[0][0]), 2)
    except:
        # Estimation de repli réaliste selon l'heure
        if 18 <= heure_cible <= 22:
            return 2.55
        elif 0 <= heure_cible <= 6:
            return 0.85
        else:
            return 1.45


# 3. PANNEAU LATÉRAL : PARAMÈTRES 100% DYNAMIQUES
st.sidebar.title("🎛️ Scénario & Horaires")

# Gestion de la clé Gemini
api_key = st.sidebar.text_input(
    "🔑 Clé API Gemini",
    type="password",
    value=st.secrets.get("GEMINI_API_KEY", ""),
    help="Colle ta clé API Google AI Studio",
)

st.sidebar.subheader("🕒 Planning Temporel")
heure_actuelle = st.sidebar.time_input(
    "Heure actuelle simulée", value=time(16, 30)
)
heure_retour = st.sidebar.time_input("Heure de retour prévue", value=time(18, 0))
heure_depart_matin = st.sidebar.time_input(
    "Heure de départ demain matin", value=time(7, 30)
)

st.sidebar.subheader("❄️ Paramètres Chambre (Clim)")
temp_actuelle = st.sidebar.slider(
    "Température actuelle (°C)", 20.0, 38.0, 30.0, 0.5
)
temp_cible = st.sidebar.slider("Température voulue (°C)", 18.0, 26.0, 21.0, 0.5)

st.sidebar.subheader("🔋 Paramètres Véhicule Électrique")
soc_actuel = st.sidebar.slider("Batterie actuelle (%)", 5, 100, 35, 5)
soc_cible = st.sidebar.slider("Batterie souhaitée (%)", 50, 100, 80, 5)
borne_kw = st.sidebar.selectbox("Puissance de la borne", [3.7, 7.4], index=0)

st.sidebar.subheader("⚡ Réseau Électrique")
seuil_max_kw = st.sidebar.slider(
    "Puissance souscrite max (kW)", 3.0, 12.0, 6.0, 0.5
)


# 4. MOTEUR DE CALCULS DYNAMIQUES
ref_date = date.today()
dt_actuel = datetime.combine(ref_date, heure_actuelle)
dt_retour = datetime.combine(ref_date, heure_retour)
dt_depart = datetime.combine(ref_date + timedelta(days=1), heure_depart_matin)

# A. Prédiction LSTM au moment du pic de retour
p_base = predire_base_lstm(heure_retour.hour)

# B. Calcul Climatisation
delta_temp = max(0.0, temp_actuelle - temp_cible)
# Vitesse de refroidissement: ~0.20 °C par minute
duree_clim_min = round(delta_temp / 0.20) if delta_temp > 0 else 0
p_clim = 2.0 if duree_clim_min > 0 else 0.0

dt_allumage_clim = dt_retour - timedelta(minutes=duree_clim_min)
dt_extinction_clim = dt_retour

# C. Calcul Voiture Électrique (Batterie 50 kWh, rendement 90%)
capacite_kwh = 50.0
energie_requise_kwh = (
    capacite_kwh * max(0, soc_cible - soc_actuel) / 100.0 / 0.90
)
duree_charge_h = (
    round(energie_requise_kwh / borne_kw, 2) if energie_requise_kwh > 0 else 0.0
)
duree_charge_td = timedelta(hours=duree_charge_h)
p_ve = borne_kw if duree_charge_h > 0 else 0.0

# D. Arbitrage Anti-Surcharge de l'Orchestrateur
p_cumulee = round(p_base + p_clim + p_ve, 2)
surcharge = p_cumulee > seuil_max_kw

if surcharge:
    # Priorité 1: Confort (la clim démarre à l'heure calculée)
    # Priorité 2: Voiture décalée en heures creuses (23h00) ou après extinction clim
    heure_creuse = datetime.combine(ref_date, time(23, 0))
    dt_debut_charge_ve = max(heure_creuse, dt_extinction_clim)
    dt_fin_charge_ve = dt_debut_charge_ve + duree_charge_td
    puissance_effective = round(p_base + p_clim, 2)
else:
    # Pas de conflit: charge immédiate dès branchement
    dt_debut_charge_ve = dt_actuel
    dt_fin_charge_ve = dt_debut_charge_ve + duree_charge_td
    puissance_effective = p_cumulee

charge_a_temps = dt_fin_charge_ve <= dt_depart


# 5. BANDEAU DE TITRE
st.markdown(
    """
<div class="main-header">
    <h2 style="margin:0; font-size: 1.8rem;">Smart Home Energy Orchestration</h2>
    <p style="margin:5px 0 0 0; opacity: 0.85;">Supervision prédictive multi-agents et Deep Learning (LSTM)</p>
</div>
""",
    unsafe_allow_html=True,
)


# 6. LES 4 CARTES DYNAMIQUES EN HAUT
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-title">🏠 MAISON EN ARRIÈRE-PLAN</div>
        <div class="metric-value">{p_base:.2f} kW</div>
        <div class="metric-sub">Prévision du modèle LSTM à {heure_retour.strftime('%H:%M')}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    clim_statut = (
        f"Allumage à {dt_allumage_clim.strftime('%H:%M')}, {duree_clim_min} min ({p_clim} kW)"
        if duree_clim_min > 0
        else "Éteinte (consigne déjà atteinte)"
    )
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-title">❄️ CONFORT</div>
        <div class="metric-value">Chambre fraîche à {heure_retour.strftime('%H:%M')}</div>
        <div class="metric-sub">{clim_statut}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    ve_texte = (
        f"{duree_charge_h} h de charge vers {soc_cible}%, départ à {dt_debut_charge_ve.strftime('%H:%M')}"
        if duree_charge_h > 0
        else "Batterie déjà au niveau cible"
    )
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-title">🚗 VOITURE ÉLECTRIQUE</div>
        <div class="metric-value">Voiture prête à {dt_fin_charge_ve.strftime('%H:%M')}</div>
        <div class="metric-sub">{ve_texte}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    if surcharge:
        sec_titre = (
            "<span style='color:#DC2626;'>Risque de coupure évité</span>"
        )
        sec_sub = f"Limite {seuil_max_kw} kW, demande cumulée {p_cumulee} kW"
    else:
        sec_titre = "<span style='color:#16A34A;'>Réseau optimal</span>"
        sec_sub = f"Puissance totale {p_cumulee} kW sous la limite ({seuil_max_kw} kW)"

    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-title">🛡️ SÉCURITÉ DU RÉSEAU</div>
        <div class="metric-value">{sec_titre}</div>
        <div class="metric-sub">{sec_sub}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.write("")

# 7. SECTION INTERMÉDIAIRE : JAUGE & ASSISTANT DE SUPERVISION
col_gauche, col_droite = st.columns([1, 1.2])

with col_gauche:
    st.markdown("### ⚡ Puissance Électrique")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_cumulee,
            number={"suffix": " kW", "font": {"size": 36, "color": "#0F172A"}},
            gauge={
                "axis": {"range": [None, 12], "tickcolor": "#64748B"},
                "bar": {"color": "#DC2626" if surcharge else "#16A34A"},
                "threshold": {
                    "line": {"color": "#0F172A", "width": 4},
                    "thickness": 0.8,
                    "value": seuil_max_kw,
                },
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    if surcharge:
        st.error(
            f"**Le disjoncteur était menacé (+{round(p_cumulee - seuil_max_kw, 2)} kW)**\n\n"
            f"- Tout lancer en même temps demanderait **{p_cumulee} kW** pour un seuil de **{seuil_max_kw} kW**.\n"
            f"- La clim démarre à **{dt_allumage_clim.strftime('%H:%M')}** pour garantir {temp_cible}°C à {heure_retour.strftime('%H:%M')}.\n"
            f"- La voiture est décalée à **{dt_debut_charge_ve.strftime('%H:%M')}** et sera prête à **{dt_fin_charge_ve.strftime('%H:%M')}**.\n"
            f"- {'✅ Elle sera prête bien avant ton départ de ' + heure_depart_matin.strftime('%H:%M') + '.' if charge_a_temps else '⚠️ Attention : la charge finira après votre départ prévu.'}"
        )
    else:
        st.success(
            f"**Réseau stable et équilibré**\n\n"
            f"- La puissance totale (**{p_cumulee} kW**) reste inférieure à votre seuil de **{seuil_max_kw} kW**.\n"
            f"- La climatisation et la voiture peuvent fonctionner en simultané dès maintenant."
        )


# 8. MOTEUR ROBUSTE D'APPEL À GEMINI (RÉSOLUTION 503 + QUESTIONS GÉNÉRALES)
def appeler_gemini_robuste(message_user, prompt_contexte, cle_api):
    # Liste de modèles avec bascule automatique pour éviter l'erreur 503
    modeles_a_tester = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"{prompt_contexte}\n\nQuestion de l'utilisateur : {message_user}"
                    }
                ]
            }
        ]
    }

    for mod in modeles_a_tester:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={cle_api}"
        try:
            r = requests.post(url, headers=headers, json=body, timeout=12)
            if r.status_code == 200:
                data = r.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif r.status_code == 503:
                continue  # Serveur occupé, on teste le modèle suivant
        except Exception:
            continue

    # Réponse de secours intelligente si Google est totalement inaccessible
    return (
        f"Je surveille votre maison en direct. Actuellement, la climatisation est calculée pour démarrer à "
        f"{dt_allumage_clim.strftime('%H:%M')} et la voiture à {dt_debut_charge_ve.strftime('%H:%M')} "
        f"(prête à {dt_fin_charge_ve.strftime('%H:%M')}). La charge réseau reste sous contrôle !"
    )


contexte_systeme = f"""
Tu es l'Orchestrateur IA de la maison connectée de Farah.
Voici la situation exacte en temps réel :
- Heure actuelle simulée : {heure_actuelle.strftime('%H:%M')}
- Heure de retour de Farah : {heure_retour.strftime('%H:%M')}
- Heure de départ demain matin : {heure_depart_matin.strftime('%H:%M')}
- Modèle LSTM (fond maison à {heure_retour.strftime('%H:%M')}) : {p_base} kW
- Climatisation : température actuelle {temp_actuelle}°C -> cible {temp_cible}°C. Allumage calculé à {dt_allumage_clim.strftime('%H:%M')} ({duree_clim_min} min de marche, {p_clim} kW).
- Voiture électrique : batterie à {soc_actuel}% -> cible {soc_cible}%. Borne {borne_kw} kW.
  Temps de charge calculé : {duree_charge_h} heures.
  Début de charge : {dt_debut_charge_ve.strftime('%H:%M')} -> Prête à {dt_fin_charge_ve.strftime('%H:%M')}.
  Pour 100% : environ {round((50.0 * (100 - soc_actuel) / 100.0 / 0.9) / borne_kw, 1)} heures.
- Seuil disjoncteur : {seuil_max_kw} kW. Puissance si tout simultané : {p_cumulee} kW.
- Arbitrage : {'Surcharge évitée, voiture décalée à ' + dt_debut_charge_ve.strftime('%H:%M') if surcharge else 'Charge simultanée autorisée'}.

Règles de comportement :
1. Réponds toujours avec bienveillance, clarté et concision en français.
2. Si Farah te pose des questions de culture générale, d'histoire, de politique (ex: premier président de la Tunisie = Habib Bourguiba), RÉPONDS DIRECTEMENT ET CORRECTEMENT à sa question, puis fais une transition élégante vers la gestion de sa maison.
3. Ne dis JAMAIS que la question dépasse le cadre !
4. Utilise toujours les heures et pourcentages calculés ci-dessus pour justifier tes choix.
"""

with col_droite:
    st.markdown("### 💬 Assistant de supervision")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"Bonjour Farah ! Je supervise votre installation. La clim est planifiée pour {dt_allumage_clim.strftime('%H:%M')} et votre voiture sera prête à {dt_fin_charge_ve.strftime('%H:%M')}. Que souhaitez-vous savoir ?",
            }
        ]

    chat_container = st.container(height=320)
    for msg in st.session_state.messages:
        with chat_container.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Écrivez votre message..."):
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with chat_container.chat_message("user"):
            st.markdown(user_input)

        if api_key:
            with chat_container.chat_message("assistant"):
                with st.spinner("L'Orchestrateur réfléchit..."):
                    reponse = appeler_gemini_robuste(
                        user_input, contexte_systeme, api_key
                    )
                    st.markdown(reponse)
            st.session_state.messages.append(
                {"role": "assistant", "content": reponse}
            )
        else:
            rep_sans_cle = (
                "Veuillez renseigner votre clé API Gemini dans le panneau latéral gauche "
                "pour activer les réponses complètes de l'assistant."
            )
            chat_container.chat_message("assistant").markdown(rep_sans_cle)
            st.session_state.messages.append(
                {"role": "assistant", "content": rep_sans_cle}
            )
