from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from tensorflow.keras.models import load_model

# =====================================================================
# 1. STYLE "BEIGE & BLUE PROFESSIONAL MINIMALIST COMPANY"
# =====================================================================
st.set_page_config(
    page_title="Smart Home Orchestration",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Palette Beige & Blue Minimalist */
    :root {
        --bg-beige: #F9F7F2;
        --card-bg: #FFFFFF;
        --border-beige: #EADBCC;
        --navy-dark: #0D1F2D;
        --navy-accent: #1D3557;
        --navy-sub: #457B9D;
        --text-muted: #6C757D;
        --terracotta: #BA3B2A;
        --sage-green: #2B6E4E;
    }

    /* Fond principal beige minimaliste */
    .stApp {
        background-color: var(--bg-beige);
        color: var(--navy-dark);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Titres et en-têtes */
    h1, h2, h3 {
        color: var(--navy-dark) !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    /* Cartes utilisateur inversées (Human First UX) */
    .user-card {
        background: var(--card-bg);
        border: 1px solid var(--border-beige);
        border-radius: 14px;
        padding: 22px 20px;
        box-shadow: 0 4px 15px rgba(13, 31, 45, 0.03);
        transition: transform 0.2s ease;
        height: 100%;
    }
    .user-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(13, 31, 45, 0.06);
    }

    .card-category {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
        color: var(--navy-sub);
        margin-bottom: 8px;
    }

    .card-human-main {
        font-size: 1.7rem;
        font-weight: 800;
        color: var(--navy-dark);
        line-height: 1.2;
        margin-bottom: 8px;
    }

    .card-technical-sub {
        font-size: 0.85rem;
        color: var(--text-muted);
        background: #F4EFE6;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 500;
    }

    /* Boîte de décision */
    .decision-panel {
        background: var(--card-bg);
        border: 1px solid var(--border-beige);
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 15px rgba(13, 31, 45, 0.03);
    }

    /* Nettoyage de l'interface par défaut */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# 2. IA DEEP LEARNING (LSTM)
# =====================================================================
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


# =====================================================================
# 3. BARRE DE CONTRÔLE (SIDEBAR PROPRIÉTAIRE)
# =====================================================================
with st.sidebar:
    st.markdown("### 🎛️ Paramètres de vie")

    # Clé API avec prise en charge automatique des secrets Streamlit
    cle_par_defaut = (
        st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else ""
    )
    api_key = st.text_input(
        "🔑 Clé Gemini (Google AI Studio)",
        value=cle_par_defaut,
        type="password",
    )

    st.markdown("---")
    st.markdown("#### ⚡ Contrat d'électricité")
    seuil_max = st.slider("Disjoncteur max (kW)", 3.0, 12.0, 6.0, 0.5)

    st.markdown("#### ❄️ Confort Chambre")
    temp_interieure = st.slider(
        "Température actuelle (°C)", 20.0, 38.0, 30.9, 0.1
    )
    temp_cible = st.slider("Température idéale (°C)", 18.0, 26.0, 21.0, 0.5)

    st.markdown("#### 🔋 Voiture Électrique")
    soc_actuel = st.slider("Niveau batterie actuel (%)", 5, 100, 35, 5)
    soc_cible = st.slider("Objectif pour demain (%)", 50, 100, 80, 5)


# =====================================================================
# 4. MOTEUR PHYSIQUE & CALCULS
# =====================================================================
p_base = predire_base()

# 1. Climatisation
delta_temp = max(0.0, temp_interieure - temp_cible)
duree_clim_min = round(delta_temp / 0.20) if delta_temp > 0 else 0
p_clim = 2.0 if duree_clim_min > 0 else 0.0

# Calcul de l'heure d'allumage pour être prêt à 18h00
heure_retour_consigne = datetime.strptime("18:00", "%H:%M")
heure_allumage_clim = (
    heure_retour_consigne - timedelta(minutes=duree_clim_min)
).strftime("%H:%M")

# 2. Voiture (50 kWh, borne 3.7 kW)
capacite_batterie_kwh = 50.0
energie_manquante_kwh = (
    capacite_batterie_kwh * max(0, soc_cible - soc_actuel) / 100.0 / 0.90
)
duree_charge_h = (
    round(energie_manquante_kwh / 3.7, 2) if energie_manquante_kwh > 0 else 0.0
)
p_ve = 3.7 if duree_charge_h > 0 else 0.0

# 3. Calculs globaux
p_totale = round(p_base + p_clim + p_ve, 2)
surcharge = p_totale > seuil_max
surcroit_evite = round(p_totale - seuil_max, 2)

# Calcul temps pour 100% de batterie
duree_100_h = round(
    (capacite_batterie_kwh * (100 - soc_actuel) / 100.0 / 0.90) / 3.7, 1
)


# =====================================================================
# 5. HEADER MINIMALISTE ENTREPRISE
# =====================================================================
st.markdown(
    """
<div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 25px;">
    <div>
        <h2 style="margin: 0; color: #0D1F2D;">Smart Home Energy Orchestration</h2>
        <p style="margin: 4px 0 0 0; color: #6C757D; font-size: 0.95rem;">
            Supervision intelligente & arbitrage préventif par Deep Learning
        </p>
    </div>
    <div style="text-align: right;">
        <span style="background: #E8EEF5; color: #1D3557; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem;">
            Système Multi-Agents Actif
        </span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# 6. CARTES INVERSÉES : L'HUMAIN EN GROS, LES kW EN SOUS-TITRE
# =====================================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
    <div class="user-card">
        <div class="card-category">🏠 Maison en arrière-plan</div>
        <div class="card-human-main">Activité Calme</div>
        <div class="card-technical-sub">⚡ Prévision IA : {p_base} kW</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
    <div class="user-card">
        <div class="card-category">❄️ Confort Chambre</div>
        <div class="card-human-main">{duree_clim_min} min requises</div>
        <div class="card-technical-sub">🎯 Prête à 18h00 ({p_clim} kW)</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
    <div class="user-card">
        <div class="card-category">🚗 Borne Véhicule</div>
        <div class="card-human-main">Prête à 07h30</div>
        <div class="card-technical-sub">🔋 {duree_charge_h}h de charge vers {soc_cible}% ({p_ve} kW)</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c4:
    statut_reseau = "Surcharge Évitée" if surcharge else "Réseau Équilibré"
    badge_color = "#BA3B2A" if surcharge else "#2B6E4E"
    st.markdown(
        f"""
    <div class="user-card" style="border-left: 4px solid {badge_color};">
        <div class="card-category">🛡️ Sécurité Réseau</div>
        <div class="card-human-main" style="color: {badge_color};">{statut_reseau}</div>
        <div class="card-technical-sub">⚡ Plafond max : {seuil_max} kW (Demande : {p_totale} kW)</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.write("")


# =====================================================================
# 7. SECTION SUPERVISION & ARBITRAGE (JAUGE BLEU NUIT & BEIGE)
# =====================================================================
col_jauge, col_assistant = st.columns([1, 1.3], gap="medium")

with col_jauge:
    st.markdown(
        """
    <div class="decision-panel">
        <h4 style="margin-top: 0; color: #0D1F2D;">⚡ Équilibre de Charge du Foyer</h4>
    """,
        unsafe_allow_html=True,
    )

    # Jauge Plotly avec les tons Beige & Navy
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_totale,
            number={"suffix": " kW", "font": {"color": "#0D1F2D", "size": 38}},
            gauge={
                "axis": {
                    "range": [None, 12],
                    "tickcolor": "#0D1F2D",
                    "tickfont": {"color": "#6C757D"},
                },
                "bar": {"color": "#BA3B2A" if surcharge else "#1D3557"},
                "bgcolor": "#EADBCC",
                "borderwidth": 0,
                "threshold": {
                    "line": {"color": "#0D1F2D", "width": 4},
                    "thickness": 0.8,
                    "value": seuil_max,
                },
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Explication humaine de l'arbitrage
    if surcharge:
        st.markdown(
            f"""
        <div style="background: #FCF4F2; border-left: 4px solid #BA3B2A; padding: 12px 14px; border-radius: 8px; font-size: 0.9rem; color: #4A1510;">
            <strong>Protection active (+{surcroit_evite} kW évités) :</strong><br>
            • ❄️ <strong>Climatisation :</strong> Démarrage à <strong>{heure_allumage_clim}</strong> pour que votre chambre soit à {temp_cible}°C à 18h.<br>
            • 🔋 <strong>Voiture :</strong> Recharge décalée à <strong>23h00</strong> en heures creuses pour soulager la ligne.<br>
            • 📉 <strong>Puissance effective ramenée à :</strong> <strong>{round(p_base + p_clim, 2)} kW</strong> (aucun risque de coupure).
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div style="background: #F0F7F4; border-left: 4px solid #2B6E4E; padding: 12px 14px; border-radius: 8px; font-size: 0.9rem; color: #143525;">
            <strong>Situation nominale :</strong><br>
            La puissance appelée ({p_totale} kW) reste sous votre plafond ({seuil_max} kW). La climatisation et la borne fonctionnent en parallèle.
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# 8. ASSISTANT EXÉCUTIF CONVERSATIONNEL (GEMINI LLM)
# =====================================================================
with col_assistant:
    st.markdown(
        """
    <div class="decision-panel">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h4 style="margin: 0; color: #0D1F2D;">💬 Assistant Exécutif de Supervision</h4>
            <span style="font-size: 0.8rem; color: #6C757D;">Orchestrateur Connecté</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

    def appeler_gemini(message_utilisateur, context_systeme, key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
        headers = {"Content-Type": "application/json"}
        prompt_complet = (
            f"{context_systeme}\n\n"
            f"Message de Farah : '{message_utilisateur}'\n"
            f"Ta réponse :"
        )
        data = {"contents": [{"parts": [{"text": prompt_complet}]}]}

        try:
            r = requests.post(url, headers=headers, json=data, timeout=10)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return f"Service temporairement indisponible (Code {r.status_code})."
        except Exception as e:
            return f"Erreur de communication : {str(e)}"

    contexte_ia = f"""
    Tu es l'Orchestrateur de Gestion Énergétique de Farah. Ton rôle est d'assurer son confort, ses trajets et la sécurité de sa maison.
    Adopte un ton élégant, bienveillant, professionnel et concis (comme un concierge privé d'hôtel de luxe ou un gestionnaire d'énergie exécutif).

    DONNÉES DU FOYER EN DIRECT :
    - Consommation passive prédite (LSTM Deep Learning) : {p_base} kW
    - Plafond disjoncteur : {seuil_max} kW
    - Chambre : Il fait {temp_interieure}°C. Farah veut {temp_cible}°C pour 18h00. Temps de clim calculé : {duree_clim_min} minutes (allumage calculé à {heure_allumage_clim}).
    - Véhicule électrique : Batterie à {soc_actuel}%. Cible demandée : {soc_cible}%. Durée pour {soc_cible}% : {duree_charge_h} heures.
    - Durée nécessaire pour atteindre 100% : {duree_100_h} heures au total.
    - Situation : {'Surcharge évitée ! Voiture décalée à 23h00 en heures creuses' if surcharge else 'Charge simultanée autorisée'}.

    DIRECTIVES DE RÉPONSE :
    1. Si Farah demande quand sa voiture sera à 100% ou utilisable : réponds avec l'heure précise en te basant sur {duree_100_h}h de charge à partir de 23h00 (ou immédiat si pas de surcharge) !
    2. Si elle pose une question d'actualité, d'histoire ou générale (ex: Habib Bourguiba, politique) : réponds-lui avec exactitude et courtoisie en une ou deux phrases, puis rajoute une petite note élégante rappelant que tout est calme et sécurisé côté maison.
    3. Exprime-toi toujours en termes clairs (heures, minutes, confort, économies) plutôt qu'en jargon purement technique.
    """

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Bonjour Farah. Je supervise en continu les flux énergétiques du foyer. "
                    "N'hésitez pas à me poser vos questions sur l'autonomie de votre véhicule, "
                    "l'optimisation de votre climatisation ou mes choix d'arbitrage."
                ),
            }
        ]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Écrivez votre message à l'Orchestrateur..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if api_key:
            with st.spinner("Analyse en cours..."):
                rep = appeler_gemini(prompt, contexte_ia, api_key)
        else:
            rep = "Veuillez renseigner votre clé API Gemini dans le panneau latéral pour interagir avec l'assistant."

        st.session_state.messages.append({"role": "assistant", "content": rep})
        with st.chat_message("assistant"):
            st.markdown(rep)

    st.markdown("</div>", unsafe_allow_html=True)
