import re
from datetime import datetime, timedelta

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# =====================================================================
# 0. CONSTANTES MÉTIER (modifiables ici, au même endroit)
# =====================================================================
MODELE_GEMINI = "gemini-2.5-flash"   # change ici si Google renomme le modèle
CAPA_BATTERIE_KWH = 50.0
PUISSANCE_BORNE_KW = 3.7
RENDEMENT_CHARGE = 0.90
PUISSANCE_CLIM_KW = 2.0
VITESSE_CLIM_C_PAR_MIN = 0.20        # °C gagnés par minute
HEURE_RETOUR = (18, 0)               # confort exigé à 18h00
HEURE_CREUSE = (23, 0)               # début charge décalée
ECHEANCE_VE = (7, 30)                # voiture prête demain à 07h30
P_BASE_SECOURS = 2.55                # utilisée si le modèle LSTM est absent

# =====================================================================
# 1. PAGE & DESIGN (BEIGE & BLEU, UNE SEULE PAGE SANS SCROLL)
# =====================================================================
st.set_page_config(
    page_title="Smart Home Energy Orchestration",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp { background-color: #F8F6F2; color: #0F1E36; }

/* Marges réduites : tout tient dans l'écran */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* Sidebar bleu marine (champs lisibles) */
section[data-testid="stSidebar"] {
    background-color: #0F1E36 !important;
    border-right: 1px solid #1E2D4A;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] small { color: #F8F6F2 !important; }
section[data-testid="stSidebar"] input {
    color: #0F1E36 !important;
    background-color: #F8F6F2 !important;
}

/* Cartes "humain d'abord" */
.user-card {
    background: #FFFFFF;
    border: 1px solid #E5E0D8;
    border-radius: 10px;
    padding: 10px 14px;
    box-shadow: 0 1px 4px rgba(15, 30, 54, 0.04);
    min-height: 104px;
}
.card-category {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em;
    font-weight: 700; color: #457B9D; margin-bottom: 4px;
}
.card-human-main {
    font-size: 1.3rem; font-weight: 800; color: #0F1E36;
    line-height: 1.2; margin-bottom: 6px;
}
.card-technical-sub {
    font-size: 0.74rem; color: #64748B; background: #F4EFE6;
    padding: 3px 9px; border-radius: 20px; display: inline-block; font-weight: 500;
}

/* Conteneurs natifs */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #E5E0D8 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
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

/* On masque le menu mais on garde le bouton de la sidebar */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""",
    unsafe_allow_html=True,
)


def html(s: str) -> str:
    """Supprime l'indentation pour que Markdown n'interprète pas le HTML comme du code."""
    return "".join(line.strip() for line in s.splitlines())


# =====================================================================
# 2. IA LSTM (chargement tolérant aux pannes)
# =====================================================================
@st.cache_resource
def charger_ia():
    try:
        from tensorflow.keras.models import load_model

        model = load_model("modele_lstm_energie.keras")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception:
        return None, None


modele, scaler = charger_ia()


@st.cache_data
def predire_base_foyer():
    """Retourne (puissance_kW, modele_actif)."""
    if modele is None or scaler is None:
        return P_BASE_SECOURS, False
    try:
        df = pd.read_csv("consommation_horaire_propre.csv", index_col="datetime")
        sample = df["Global_active_power"].iloc[100:124].values
        scaled = scaler.transform(sample.reshape(-1, 1))
        pred_scaled = modele.predict(scaled.reshape(1, 24, 1), verbose=0)
        return round(float(scaler.inverse_transform(pred_scaled)[0][0]), 2), True
    except Exception:
        return P_BASE_SECOURS, False


# =====================================================================
# 3. PANNEAU DE CONTRÔLE (SIDEBAR)
# =====================================================================
with st.sidebar:
    st.markdown("### 🏛️ Centre de Contrôle")
    st.caption("Paramètres du foyer")
    st.markdown("---")

    try:
        cle_secrete = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        cle_secrete = ""
    api_key = st.text_input(
        "Clé API Gemini (optionnel)",
        value=cle_secrete,
        type="password",
        help="Sans clé, l'assistant répond en mode local (sujets énergie uniquement).",
    )

    seuil_max = st.slider("Souscription disjoncteur (kW)", 3.0, 12.0, 6.0, 0.5)
    temp_interieure = st.slider("Température ambiante (°C)", 20.0, 38.0, 30.0, 0.5)
    temp_cible = st.slider("Consigne souhaitée (°C)", 18.0, 26.0, 21.0, 0.5)
    soc_actuel = st.slider("Batterie actuelle (%)", 5, 100, 35, 5)
    soc_cible = st.slider("Batterie cible (%)", 50, 100, 80, 5)


# =====================================================================
# 4. MOTEUR DE CALCUL
# =====================================================================
JOUR0 = datetime(2000, 1, 1)


def at(h, m=0, jour=0):
    return JOUR0 + timedelta(days=jour, hours=h, minutes=m)


def fmt(dt):
    """HH:MM, avec « (demain) » si on a passé minuit."""
    return dt.strftime("%H:%M") + (" (demain)" if dt.day > 1 else "")


p_base, ia_active = predire_base_foyer()

# --- Climatisation (prête à 18h00)
delta_t = max(0.0, temp_interieure - temp_cible)
duree_clim_min = round(delta_t / VITESSE_CLIM_C_PAR_MIN) if delta_t > 0 else 0
p_clim = PUISSANCE_CLIM_KW if duree_clim_min > 0 else 0.0
heure_retour = at(*HEURE_RETOUR)
debut_clim = heure_retour - timedelta(minutes=duree_clim_min)

# --- Voiture
energie_cible_kwh = CAPA_BATTERIE_KWH * max(0, soc_cible - soc_actuel) / 100 / RENDEMENT_CHARGE
duree_charge_h = round(energie_cible_kwh / PUISSANCE_BORNE_KW, 2) if energie_cible_kwh > 0 else 0.0
energie_100_kwh = CAPA_BATTERIE_KWH * max(0, 100 - soc_actuel) / 100 / RENDEMENT_CHARGE
duree_100_h = round(energie_100_kwh / PUISSANCE_BORNE_KW, 2) if energie_100_kwh > 0 else 0.0
p_ve = PUISSANCE_BORNE_KW if duree_charge_h > 0 else 0.0

# --- Bilan réseau
p_totale = round(p_base + p_clim + p_ve, 2)
surcharge = p_totale > seuil_max
surcroit = round(p_totale - seuil_max, 2) if surcharge else 0.0
marge = round(seuil_max - p_totale, 2)

# Arbitrage : si surcharge, la voiture est décalée en heures creuses
debut_ve = at(*HEURE_CREUSE) if surcharge else at(*HEURE_RETOUR)
fin_ve = debut_ve + timedelta(hours=duree_charge_h)
fin_100 = debut_ve + timedelta(hours=duree_100_h)
echeance = at(*ECHEANCE_VE, jour=1)
ve_a_temps = duree_charge_h == 0 or fin_ve <= echeance
debut_ve_requis = echeance - timedelta(hours=duree_charge_h)

# Pics après arbitrage (soirée : base+clim / nuit : base+voiture)
pic_apres = max(p_base + p_clim, p_base + p_ve) if surcharge else p_totale
critique = surcharge and pic_apres > seuil_max

if not surcharge:
    statut, couleur = "Réseau équilibré", "#2B6E4E"
elif critique:
    statut, couleur = "Surcharge critique", "#B91C1C"
else:
    statut, couleur = "Surcharge évitée", "#1D3557"

if p_base < 0.4 * seuil_max:
    activite = "Activité calme"
elif p_base < 0.7 * seuil_max:
    activite = "Activité modérée"
else:
    activite = "Activité élevée"


# =====================================================================
# 5. ASSISTANT : MOTEUR LOCAL + GEMINI
# =====================================================================
def has(q, tokens, mots):
    return any((m in q) if len(m) > 3 else (m in tokens) for m in mots)


def repondre_local(prompt):
    q = prompt.lower().strip()
    tokens = set(re.findall(r"\w+", q))
    voiture = has(q, tokens, ["voiture", "batterie", "charge", "recharge", "borne", "ve"])

    if has(q, tokens, ["100", "plein", "complet"]) and voiture:
        return (
            f"Pour atteindre **100 %**, il faut **{energie_100_kwh:.1f} kWh**, soit **{duree_100_h} h** sur la borne.\n\n"
            f"- Début de charge : **{fmt(debut_ve)}**\n"
            f"- Batterie pleine vers : **{fmt(fin_100)}**"
        )
    if has(q, tokens, ["clim", "chambre", "temperature", "chaleur", "frais", "fraiche"]):
        if duree_clim_min == 0:
            return f"La pièce est déjà à **{temp_interieure}°C**, sous votre consigne de **{temp_cible}°C**. Aucune climatisation nécessaire."
        return (
            f"Pièce à **{temp_interieure}°C** pour une consigne de **{temp_cible}°C**.\n\n"
            f"Il faut **{duree_clim_min} min** : allumage à **{fmt(debut_clim)}** pour une chambre fraîche à 18h00."
        )
    if voiture:
        if duree_charge_h == 0:
            return f"Batterie à **{soc_actuel}%** : l'objectif de **{soc_cible}%** est déjà atteint."
        planning = "décalée à 23h00 (heures creuses)" if surcharge else "immédiate (dès 18h00)"
        etat = "✅ Prête pour 07h30." if ve_a_temps else f"⚠️ Fin vers {fmt(fin_ve)} : trop tard pour 07h30."
        return (
            f"Batterie à **{soc_actuel}%** → cible **{soc_cible}%** en **{duree_charge_h} h**.\n\n"
            f"Charge {planning}, terminée vers **{fmt(fin_ve)}**. {etat}"
        )
    if has(q, tokens, ["surcharge", "disjoncteur", "securite", "coupure", "courant", "arbitrage"]):
        if not surcharge:
            return f"Aucun risque : **{p_totale:.2f} kW** appelés pour **{seuil_max:.1f} kW** souscrits (marge **{marge:.2f} kW**)."
        return (
            f"La demande brute ({p_totale:.2f} kW) dépasse le disjoncteur ({seuil_max:.1f} kW) de **{surcroit:.2f} kW**. "
            f"J'ai donc décalé la voiture à 23h00 : pic ramené à **{pic_apres:.2f} kW**."
        )
    if has(q, tokens, ["ia", "lstm", "modele", "deep", "prevision"]):
        src = "calculée par le modèle" if ia_active else "estimée (modèle non chargé)"
        return f"Réseau **LSTM** autorégressif (MAE ≈ 0.60 kW) : charge passive du foyer {src} à **{p_base:.2f} kW**."
    if has(q, tokens, ["bonjour", "salut", "hello", "bonsoir"]):
        return "Bonjour ! Je suis prêt à vous renseigner sur la voiture, la climatisation ou la sécurité du réseau."
    return (
        "Je peux vous renseigner sur la **voiture**, la **climatisation**, la **sécurité du réseau** ou le **modèle IA**. "
        "Pour les questions générales, ajoutez une clé Gemini dans le panneau latéral."
    )


def contexte_gemini():
    decision = (
        f"Surcharge évitée : voiture décalée à {fmt(debut_ve)} (heures creuses)."
        if surcharge else "Pas de surcharge : charge immédiate autorisée."
    )
    return f"""Tu es l'Orchestrateur de Gestion Énergétique de Farah (maison connectée).
Ton : élégant, bienveillant, concis (3 phrases max), en français, en termes d'heures, de confort et de sécurité plutôt qu'en jargon.

DONNÉES EN DIRECT :
- Consommation passive prédite (LSTM) : {p_base} kW
- Disjoncteur : {seuil_max} kW | Puissance appelée brute : {p_totale} kW
- Chambre : {temp_interieure}°C, consigne {temp_cible}°C pour 18h00. Clim : {duree_clim_min} min, allumage à {fmt(debut_clim)}.
- Voiture : batterie {soc_actuel}%, cible {soc_cible}% en {duree_charge_h} h, prête vers {fmt(fin_ve)} (échéance 07h30).
- Pour 100 % : {duree_100_h} h de charge, batterie pleine vers {fmt(fin_100)}.
- Décision : {decision}

RÈGLES :
1. Si on te demande quand la voiture sera pleine ou utilisable, donne l'heure précise ci-dessus.
2. Pour une question générale (histoire, actualité, culture), réponds correctement en une ou deux phrases, puis ajoute une courte note rassurante sur l'état de la maison.
3. N'invente aucune donnée absente de la liste ci-dessus."""


def appeler_gemini(messages, key):
    """Retourne (texte, None) ou (None, message_erreur)."""
    contents = []
    for m in messages[-9:]:
        role = "user" if m["role"] == "user" else "model"
        if not contents and role == "model":
            continue  # Gemini exige de commencer par un message utilisateur
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELE_GEMINI}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": contexte_gemini()}]},
        "contents": contents,
    }
    try:
        r = requests.post(
            url,
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            json=payload,
            timeout=15,
        )
        if r.status_code != 200:
            return None, f"code {r.status_code}"
        return r.json()["candidates"][0]["content"]["parts"][0]["text"], None
    except Exception as e:
        return None, type(e).__name__


# =====================================================================
# 6. DASHBOARD
# =====================================================================
st.markdown(
    html(
        """
<div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px;">
<h3 style="margin:0; color:#0F1E36; font-size:1.35rem; font-weight:700;">Smart Home Energy Orchestration</h3>
<span style="font-size:0.8rem; color:#64748B;">Supervision prédictive multi-agents &amp; Deep Learning (LSTM)</span>
</div>
"""
    ),
    unsafe_allow_html=True,
)

# ---- Ligne 1 : cartes (l'humain en gros, les kW en discret)
c1, c2, c3, c4 = st.columns(4)

with c1:
    src = "Prévision LSTM" if ia_active else "Valeur de secours"
    st.markdown(
        html(f"""
<div class="user-card">
<div class="card-category">🏠 Maison en arrière-plan</div>
<div class="card-human-main">{activite}</div>
<div class="card-technical-sub">⚡ {src} : {p_base:.2f} kW</div>
</div>"""),
        unsafe_allow_html=True,
    )

with c2:
    if duree_clim_min > 0:
        principal = "Fraîche à 18h00"
        sub = f"❄️ Allumage {fmt(debut_clim)} · {duree_clim_min} min · {p_clim:.1f} kW"
    else:
        principal = "Déjà confortable"
        sub = f"❄️ {temp_interieure}°C ≤ consigne {temp_cible}°C · 0 kW"
    st.markdown(
        html(f"""
<div class="user-card">
<div class="card-category">Confort chambre</div>
<div class="card-human-main">{principal}</div>
<div class="card-technical-sub">{sub}</div>
</div>"""),
        unsafe_allow_html=True,
    )

with c3:
    if duree_charge_h == 0:
        principal = "Aucune charge requise"
        sub = f"🔋 {soc_actuel}% ≥ cible {soc_cible}%"
        etiq_couleur = "#0F1E36"
    elif ve_a_temps:
        principal = f"Prête à {fmt(fin_ve).replace(' (demain)', '')}"
        sub = f"🔋 {duree_charge_h} h vers {soc_cible}% · départ {fmt(debut_ve).replace(' (demain)', '')} · {p_ve:.1f} kW"
        etiq_couleur = "#0F1E36"
    else:
        principal = f"Prête à {fmt(fin_ve).replace(' (demain)', '')}"
        sub = f"⚠️ Après 07h30 · démarrer avant {fmt(debut_ve_requis)}"
        etiq_couleur = "#B91C1C"
    st.markdown(
        html(f"""
<div class="user-card">
<div class="card-category">🚗 Borne véhicule</div>
<div class="card-human-main" style="color:{etiq_couleur};">{principal}</div>
<div class="card-technical-sub">{sub}</div>
</div>"""),
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        html(f"""
<div class="user-card" style="border-left:4px solid {couleur};">
<div class="card-category">🛡️ Sécurité réseau</div>
<div class="card-human-main" style="color:{couleur};">{statut}</div>
<div class="card-technical-sub">⚡ Plafond {seuil_max:.1f} kW · demande {p_totale:.2f} kW</div>
</div>"""),
        unsafe_allow_html=True,
    )

st.write("")

# ---- Ligne 2 : jauge + arbitrage | assistant
col_gauche, col_droite = st.columns([1.1, 1.2], gap="medium")

with col_gauche:
    with st.container(border=True):
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=p_totale,
                number={
                    "valueformat": ".2f",
                    "suffix": " kW",
                    "font": {"size": 24, "color": "#0F1E36", "family": "Plus Jakarta Sans"},
                },
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 12], "tickwidth": 1, "tickcolor": "#CBD5E1", "nticks": 7},
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
        try:
            st.plotly_chart(fig, width="stretch")
        except TypeError:  # anciennes versions de Streamlit
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr style='margin:6px 0;'>", unsafe_allow_html=True)

        if not surcharge:
            st.markdown(
                html(f"""
<div style="font-size:0.85rem; line-height:1.45;">
<span style="color:#2B6E4E; font-weight:700; text-transform:uppercase; font-size:0.75rem;">● Situation nominale (marge : {marge:.2f} kW)</span>
<p style="margin:4px 0 0 0; color:#334155;">La puissance globale ({p_totale:.2f} kW) respecte votre abonnement. Climatisation et borne fonctionnent en simultané.</p>
</div>"""),
                unsafe_allow_html=True,
            )
        else:
            titre = "Surcharge critique" if critique else f"Surcharge évitée (+{surcroit:.2f} kW)"
            alerte = (
                f"<li style='color:#B91C1C;'><strong>Attention :</strong> même après arbitrage, le pic ({pic_apres:.2f} kW) dépasse le disjoncteur. Augmentez la souscription ou réduisez la charge.</li>"
                if critique else ""
            )
            st.markdown(
                html(f"""
<div style="font-size:0.85rem; line-height:1.45;">
<span style="color:{couleur}; font-weight:700; text-transform:uppercase; font-size:0.75rem;">● {titre}</span>
<p style="margin:4px 0 6px 0; color:#334155;">La demande ({p_totale:.2f} kW) dépasse le disjoncteur ({seuil_max:.1f} kW).</p>
<ul style="margin:0; padding-left:18px; color:#0F1E36;">
<li><strong>Climatisation :</strong> allumage à <strong>{fmt(debut_clim)}</strong> (chambre à {temp_cible}°C à 18h).</li>
<li><strong>Voiture :</strong> recharge décalée à <strong>{HEURE_CREUSE[0]}h{HEURE_CREUSE[1]:02d}</strong> (heures creuses), terminée vers <strong>{fmt(fin_ve)}</strong>.</li>
<li><strong>Pic après arbitrage :</strong> <strong>{pic_apres:.2f} kW</strong>.</li>
{alerte}
</ul>
</div>"""),
                unsafe_allow_html=True,
            )

with col_droite:
    with st.container(border=True):
        mode = "Gemini" if api_key else "Mode local"
        st.markdown(
            html(f"""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
<strong style="color:#0F1E36; font-size:0.92rem;">💬 Assistant exécutif de supervision</strong>
<span style="font-size:0.75rem; color:#64748B;">{mode} · connecté aux agents</span>
</div>"""),
            unsafe_allow_html=True,
        )

        # Défilement INTERNE : la page ne bouge jamais
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

        if prompt := st.chat_input("Écrivez votre message..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_box.chat_message("user"):
                st.markdown(prompt)

            with chat_box.chat_message("assistant"):
                rep, erreur = None, None
                if api_key:
                    with st.spinner("Analyse en cours..."):
                        rep, erreur = appeler_gemini(st.session_state.messages, api_key)
                if rep is None:
                    rep = repondre_local(prompt)
                    if erreur:
                        rep += f"\n\n*(Gemini indisponible : {erreur}, réponse en mode local.)*"
                st.markdown(rep)

            st.session_state.messages.append({"role": "assistant", "content": rep})
