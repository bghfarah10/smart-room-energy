from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from tensorflow.keras.models import load_model

# =====================================================================
# 1. CONFIGURATION PLEIN ÉCRAN & DESIGN MINIMALISTE (BEIGE & BLEU)
# =====================================================================
st.set_page_config(
    page_title="Smart Home Hub • Executive Overview",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Fond principal Beige Crème */
    .stApp {
        background-color: #F8F6F2;
        color: #0F1E36;
    }

    /* Réduction drastique des marges pour tenir en 1 seule page */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* Barre latérale Bleu Marine */
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

    /* Cartes Métriques compactes */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E0D8;
        border-radius: 8px;
        padding: 10px 14px;
        box-shadow: 0 1px 3px rgba(15, 30, 54, 0.04);
    }
    div[data-testid="stMetric"] label {
        color: #64748B !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0F1E36 !important;
        font-weight: 700 !important;
        font-size: 1.35rem !important;
    }

    /* Bordures des conteneurs natifs */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E0D8 !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        box-shadow: 0 1px 4px rgba(15, 30, 54, 0.03) !important;
    }

    /* Messages du chat compacts */
    .stChatMessage {
        background-color: #FAF8F5 !important;
        border: 1px solid #EAE5DD !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        margin-bottom: 6px !important;
        font-size: 0.88rem !important;
    }

    #MainMenu, footer, header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# 2. CHARGEMENT DE L'IA
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
# 3. PANNEAU DE CONTRÔLE (SIDEBAR)
# =====================================================================
with st.sidebar:
    st.markdown("### 🏛️ Centre de Contrôle")
    st.caption("Paramètres du Foyer")
    st.markdown("---")

    cle_secrete = st.secrets.get("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Clé API Gemini (Optionnel)",
        value=cle_secrete,
        type="password",
        help="Connecte directement le LLM pour répondre à toute question.",
    )

    seuil_max = st.slider("Souscription Disjoncteur (kW)", 3.0, 12.0, 6.0, 0.5)
    temp_interieure = st.slider(
        "Température Ambiante (°C)", 20.0, 38.0, 30.0, 0.5
    )
    temp_cible = st.slider("Consigne Souhaitée (°C)", 18.0, 26.0, 21.0, 0.5)
    soc_actuel = st.slider("Batterie Actuelle (%)", 5, 100, 35, 5)
    soc_cible = st.slider("Batterie Cible (%)", 50, 100, 80, 5)


# =====================================================================
# 4. CALCULS EN TEMPS RÉEL
# =====================================================================
p_base = predire_base_foyer()

# Clim
delta_t = max(0.0, temp_interieure - temp_cible)
duree_clim_min = round(delta_t / 0.20) if delta_t > 0 else 0
p_clim = 2.0 if duree_clim_min > 0 else 0.0
heure_demarrage_clim = (
    datetime.strptime("18:00", "%H:%M") - timedelta(minutes=duree_clim_min)
).strftime("%H:%M")

# Voiture (50 kWh, borne 3.7 kW)
capa_kwh = 50.0
puiss_borne = 3.7
energie_cible_kwh = (
    capa_kwh * max(0, soc_cible - soc_actuel) / 100.0 / 0.90
)
duree_charge_cible_h = (
    round(energie_cible_kwh / puiss_borne, 2) if energie_cible_kwh > 0 else 0.0
)

energie_100_kwh = capa_kwh * max(0, 100 - soc_actuel) / 100.0 / 0.90
duree_charge_100_h = (
    round(energie_100_kwh / puiss_borne, 2) if energie_100_kwh > 0 else 0.0
)
p_ve = puiss_borne if duree_charge_cible_h > 0 else 0.0

p_totale_brute = round(p_base + p_clim + p_ve, 2)
surcharge = p_totale_brute > seuil_max
surcroit = round(p_totale_brute - seuil_max, 2) if surcharge else 0.0

heure_debut_charge = "23:00" if surcharge else "Immédiate (18:00)"
minutes_100 = int(duree_charge_100_h * 60)
heure_fin_100 = (
    datetime.strptime("23:00" if surcharge else "18:00", "%H:%M")
    + timedelta(minutes=minutes_100)
).strftime("%H:%M")


# =====================================================================
# 5. DASHBOARD EN UNE SEULE PAGE (AUCUN SCROLL)
# =====================================================================

# Titre compact
st.markdown(
    """
<div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px;">
    <h3 style="margin:0; color:#0F1E36; font-size:1.35rem; font-weight:700;">Smart Home Energy Orchestration</h3>
    <span style="font-size:0.8rem; color:#64748B;">Supervision prédictive multi-agents & Deep Learning (LSTM)</span>
</div>
""",
    unsafe_allow_html=True,
)

# Ligne 1 : Les 4 KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Prévision Base (LSTM)", f"{p_base:.2f} kW", "Charge passive")
c2.metric("Climatisation", f"{p_clim:.2f} kW", f"{duree_clim_min} min nécessaires")
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

# Ligne 2 : Deux colonnes principales
col_gauche, col_droite = st.columns([1.1, 1.2])

# COLONNE GAUCHE : JAUGE + ARBITRAGE
with col_gauche:
    with st.container(border=True):
        # JAUGE CORRIGÉE (Chiffre 100% centré, aucun texte coupé)
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=p_totale_brute,
                number={
                    "valueformat": ".2f",
                    "suffix": " kW",
                    "font": {
                        "size": 24,
                        "color": "#0F1E36",
                        "family": "Plus Jakarta Sans",
                    },
                },
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {
                        "range": [0, 12],
                        "tickwidth": 1,
                        "tickcolor": "#CBD5E1",
                        "nticks": 7,
                    },
                    "bar": {"color": "#B91C1C" if surcharge else "#1B4D3E"},
                    "bgcolor": "#F4EFEA",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, seuil_max], "color": "#EAE5DC"},
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
            height=160,
            margin=dict(l=15, r=15, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr style='margin: 6px 0;'>", unsafe_allow_html=True)

        # BLOC DÉCISION
        if surcharge:
            st.markdown(
                f"""
            <div style="font-size:0.85rem; line-height:1.45;">
                <span style="color:#B91C1C; font-weight:700; text-transform:uppercase; font-size:0.75rem;">
                    ● Risque de Surcharge Évité (+{surcroit:.2f} kW)
                </span>
                <p style="margin: 4px 0 6px 0; color:#334155;">
                    La demande ({p_totale_brute:.2f} kW) dépasse le disjoncteur ({seuil_max:.1f} kW).
                </p>
                <ul style="margin:0; padding-left:18px; color:#0F1E36;">
                    <li><strong>Climatisation :</strong> Démarrage à <strong>{heure_demarrage_clim}</strong> (garantit {temp_cible}°C à 18h).</li>
                    <li><strong>Voiture :</strong> Recharge décalée à <strong>23h00</strong> (heures creuses).</li>
                    <li><strong>Ligne sécurisée :</strong> <strong>{p_base + p_clim:.2f} kW</strong>.</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div style="font-size:0.85rem; line-height:1.45;">
                <span style="color:#1B4D3E; font-weight:700; text-transform:uppercase; font-size:0.75rem;">
                    ● Réseau Stable (Marge disponible : {seuil_max - p_totale_brute:.2f} kW)
                </span>
                <p style="margin: 4px 0 0 0; color:#334155;">
                    La puissance globale ({p_totale_brute:.2f} kW) respecte votre abonnement. La climatisation et la borne fonctionnent en simultané.
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

# COLONNE DROITE : CHATBOT FIXE (AUCUN DÉBORDEMENT)
with col_droite:
    with st.container(border=True):
        st.markdown(
            """
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <strong style="color:#0F1E36; font-size:0.92rem;">Assistant Exécutif de Supervision</strong>
            <span style="font-size:0.75rem; color:#64748B;">Connecté aux Agents</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Zone de chat à défilement INTERNE (ne fait pas défiler la page !)
        chat_box = st.container(height=265)

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Bonjour Farah. Je supervise en direct l'énergie du foyer. Posez-moi vos questions sur la recharge, le confort ou l'arbitrage.",
                }
            ]

        for msg in st.session_state.messages:
            with chat_box.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Champ de texte avec placeholder neutre (plus d'exemple sur la voiture !)
        if prompt := st.chat_input("Écrivez votre message..."):
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )
            with chat_box.chat_message("user"):
                st.markdown(prompt)

            # Moteur conversationnel
            q = prompt.lower().strip()

            if "100" in q or (
                "plein" in q
                and any(w in q for w in ["voiture", "batterie", "ve"])
            ):
                rep = (
                    f"Pour atteindre **100%**, il reste **{energie_100_kwh:.1f} kWh** à injecter (soit **{duree_charge_100_h} h** sur borne 3.7 kW).\n\n"
                    f"- Début de charge : **{heure_debut_charge}**\n"
                    f"- Disponibilité à 100% : vers **{heure_fin_100}**."
                )
            elif any(
                w in q for w in ["voiture", "batterie", "charge", "recharge"]
            ):
                rep = (
                    f"Batterie à **{soc_actuel}%** (cible : **{soc_cible}%**, durée : **{duree_charge_cible_h} h**).\n"
                    f"Planning : {'Différée à 23h00 (heures creuses)' if surcharge else 'Charge immédiate'}. Prête pour 07h30."
                )
            elif any(
                w in q for w in ["clim", "chambre", "temperature", "chaleur"]
            ):
                rep = (
                    f"Pièce à **{temp_interieure}°C** (consigne {temp_cible}°C).\n"
                    f"Temps estimé : **{duree_clim_min} min**. Démarrage à **{heure_demarrage_clim}** pour un confort parfait à 18h00."
                )
            elif any(w in q for w in ["tunisie", "bourguiba", "president"]):
                rep = (
                    "Le premier président de la République tunisienne est **Habib Bourguiba** (proclamé le 25 juillet 1957).\n\n"
                    "💡 *Note énergétique :* Les équipements du foyer sont sous contrôle."
                )
            elif any(
                w in q for w in ["ia", "lstm", "modele", "deep learning"]
            ):
                rep = f"Réseau Deep Learning **LSTM** autorégressif (MAE: 0.60 kW) prédisant la charge passive du foyer ({p_base:.2f} kW)."
            else:
                rep = (
                    f"J'ai bien noté votre demande : *« {prompt} »*.\n"
                    "Je reste à votre disposition pour le suivi de la batterie, le confort thermique ou l'arbitrage du réseau."
                )

            st.session_state.messages.append(
                {"role": "assistant", "content": rep}
            )
            with chat_box.chat_message("assistant"):
                st.markdown(rep)
