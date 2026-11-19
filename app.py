import math
import re
from datetime import datetime, timedelta

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# =====================================================================
# 0. CONSTANTES MÉTIER
# =====================================================================
# Modèles Gemini essayés dans l'ordre (le premier qui répond est mémorisé)
MODELES_GEMINI = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-1.5-flash",
]
API_BASE = "https://generativelanguage.googleapis.com/v1beta"

CAPA_BATTERIE_KWH = 50.0
PUISSANCE_BORNE_KW = 3.7
RENDEMENT_CHARGE = 0.90
PUISSANCE_CLIM_KW = 2.0
VITESSE_CLIM_C_PAR_MIN = 0.20
HEURE_RETOUR = (18, 0)
HEURE_CREUSE = (23, 0)
ECHEANCE_VE = (7, 30)
P_BASE_SECOURS = 2.55

# =====================================================================
# 1. PAGE & DESIGN (beige et bleu, une seule page, sans icônes)
# =====================================================================
st.set_page_config(
    page_title="Smart Home Energy Orchestration",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background-color: #F8F6F2; color: #0F1E36; }

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* Sidebar */
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

/* Cartes */
.user-card {
    background: #FFFFFF;
    border: 1px solid #E5E0D8;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 1px 4px rgba(15, 30, 54, 0.04);
    min-height: 104px;
}
.card-category {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em;
    font-weight: 700; color: #5B7A99; margin-bottom: 4px;
}
.card-human-main {
    font-size: 1.28rem; font-weight: 800; color: #0F1E36;
    line-height: 1.2; margin-bottom: 6px;
}
.card-technical-sub { font-size: 0.78rem; color: #64748B; font-weight: 500; }

/* Conteneurs natifs */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #E5E0D8 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    box-shadow: 0 1px 4px rgba(15, 30, 54, 0.03) !important;
}

/* Chat sans avatars, avec un petit nom d'interlocuteur */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }
.stChatMessage {
    background-color: #FAF8F5 !important;
    border: 1px solid #EAE5DD !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin-bottom: 6px !important;
    font-size: 0.88rem !important;
}
.who {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: #5B7A99; margin-bottom: 2px;
}

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
# 2. MODÈLE LSTM
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
# 3. SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### Centre de contrôle")
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
        help="Sans clé, l'assistant répond en mode local, sur les sujets de la maison uniquement.",
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


def hm(dt):
    """Heure au format 05h45."""
    return dt.strftime("%Hh%M")


def fmt(dt):
    """05h45, avec « demain » si on a passé minuit."""
    return hm(dt) + (" demain" if dt.day > 1 else "")


p_base, ia_active = predire_base_foyer()

# Climatisation (fraîcheur exigée à 18h00)
delta_t = max(0.0, temp_interieure - temp_cible)
duree_clim_min = round(delta_t / VITESSE_CLIM_C_PAR_MIN) if delta_t > 0 else 0
p_clim = PUISSANCE_CLIM_KW if duree_clim_min > 0 else 0.0
debut_clim = at(*HEURE_RETOUR) - timedelta(minutes=duree_clim_min)

# Voiture
energie_cible_kwh = CAPA_BATTERIE_KWH * max(0, soc_cible - soc_actuel) / 100 / RENDEMENT_CHARGE
duree_charge_h = round(energie_cible_kwh / PUISSANCE_BORNE_KW, 2) if energie_cible_kwh > 0 else 0.0
energie_100_kwh = CAPA_BATTERIE_KWH * max(0, 100 - soc_actuel) / 100 / RENDEMENT_CHARGE
duree_100_h = round(energie_100_kwh / PUISSANCE_BORNE_KW, 2) if energie_100_kwh > 0 else 0.0
p_ve = PUISSANCE_BORNE_KW if duree_charge_h > 0 else 0.0

# Bilan réseau
p_totale = round(p_base + p_clim + p_ve, 2)
surcharge = p_totale > seuil_max
surcroit = round(p_totale - seuil_max, 2) if surcharge else 0.0
marge = round(seuil_max - p_totale, 2)

# Arbitrage : voiture décalée en heures creuses si surcharge
debut_ve = at(*HEURE_CREUSE) if surcharge else at(*HEURE_RETOUR)
fin_ve = debut_ve + timedelta(hours=duree_charge_h)
fin_100 = debut_ve + timedelta(hours=duree_100_h)
echeance = at(*ECHEANCE_VE, jour=1)
ve_a_temps = duree_charge_h == 0 or fin_ve <= echeance
debut_ve_requis = echeance - timedelta(hours=duree_charge_h)

pic_apres = max(p_base + p_clim, p_base + p_ve) if surcharge else p_totale
critique = surcharge and pic_apres > seuil_max
souscription_conseillee = math.ceil(pic_apres * 2) / 2  # arrondi au 0,5 kW supérieur

if not surcharge:
    statut, couleur = "Tout est sous contrôle", "#2B6E4E"
elif critique:
    statut, couleur = "Risque de coupure", "#B91C1C"
else:
    statut, couleur = "Coupure évitée", "#1D3557"

if p_base < 0.4 * seuil_max:
    activite = "Maison tranquille"
elif p_base < 0.7 * seuil_max:
    activite = "Consommation normale"
else:
    activite = "Maison très active"


# =====================================================================
# 5. ASSISTANT : MOTEUR LOCAL + GEMINI
# =====================================================================
def has(q, tokens, mots):
    return any((m in q) if len(m) > 3 else (m in tokens) for m in mots)


def repondre_local(prompt):
    q = prompt.lower().strip()
    tokens = set(re.findall(r"\w+", q))

    if has(q, tokens, ["surcharge", "disjoncteur", "coupure", "securite", "arbitrage", "sauter"]):
        if not surcharge:
            return (
                f"Aucun souci de ce côté. La maison réclame {p_totale:.2f} kW pour {seuil_max:.1f} kW "
                f"autorisés, il reste donc {marge:.2f} kW de marge."
            )
        base = (
            f"Si tout démarrait en même temps, la maison réclamerait {p_totale:.2f} kW, "
            f"soit {surcroit:.2f} kW de trop pour votre disjoncteur ({seuil_max:.1f} kW). "
            f"J'ai donc décalé la recharge de la voiture à {hm(at(*HEURE_CREUSE))}. "
        )
        if critique:
            return base + (
                f"Malgré cela, le pic reste à {pic_apres:.2f} kW. Il faudrait une souscription "
                f"d'au moins {souscription_conseillee:.1f} kW, ou une charge plus légère."
            )
        return base + f"Le pic redescend à {pic_apres:.2f} kW, sans risque."

    if has(q, tokens, ["100", "plein", "complet", "pleine"]):
        return (
            f"Pour remplir complètement la batterie, il faut {energie_100_kwh:.1f} kWh, "
            f"soit environ {duree_100_h} h de charge. En partant à {hm(debut_ve)}, "
            f"elle sera pleine vers {fmt(fin_100)}."
        )

    if has(q, tokens, ["clim", "chambre", "temperature", "chaleur", "frais", "fraiche", "climatisation"]):
        if duree_clim_min == 0:
            return f"La pièce est à {temp_interieure}°C, déjà sous votre consigne de {temp_cible}°C. Pas besoin de climatiser."
        return (
            f"La pièce est à {temp_interieure}°C et vous la voulez à {temp_cible}°C. "
            f"Il faut compter {duree_clim_min} minutes : je lance donc la clim à {hm(debut_clim)} "
            f"pour qu'elle soit fraîche à 18h00."
        )

    if has(q, tokens, ["voiture", "batterie", "recharge", "borne", "ve", "vehicule"]) or "charge" in tokens:
        if duree_charge_h == 0:
            return f"La batterie est à {soc_actuel}%, votre objectif de {soc_cible}% est déjà atteint."
        quand = f"à {hm(debut_ve)}, en heures creuses" if surcharge else "tout de suite"
        fin = (
            f"Elle sera prête vers {fmt(fin_ve)}, largement avant 07h30."
            if ve_a_temps
            else f"Elle ne sera prête que vers {fmt(fin_ve)}, trop tard pour 07h30. Il faudrait démarrer avant {fmt(debut_ve_requis)}."
        )
        return (
            f"La batterie est à {soc_actuel}% et doit monter à {soc_cible}%, soit {duree_charge_h} h de charge. "
            f"La recharge démarre {quand}. {fin}"
        )

    if has(q, tokens, ["kw", "kilowatt", "puissance", "watt", "appelee", "signifie", "veut", "dire"]):
        parts = [f"{p_base:.2f} kW pour le fonctionnement habituel de la maison"]
        if p_clim:
            parts.append(f"{p_clim:.1f} kW pour la climatisation")
        if p_ve:
            parts.append(f"{p_ve:.1f} kW pour la borne de recharge")
        detail = ", ".join(parts[:-1]) + (" et " if len(parts) > 1 else "") + parts[-1]
        verdict = (
            f"C'est au-dessus de votre limite de {seuil_max:.1f} kW, d'où l'arbitrage."
            if surcharge
            else f"C'est sous votre limite de {seuil_max:.1f} kW."
        )
        return (
            f"Le kilowatt (kW) mesure la puissance que la maison tire à un instant donné. "
            f"Ici, {p_totale:.2f} kW correspondent à la somme de : {detail}. {verdict}"
        )

    if has(q, tokens, ["ia", "lstm", "modele", "deep", "prevision", "intelligence"]):
        etat = (
            f"Il estime la consommation de fond de la maison à {p_base:.2f} kW."
            if ia_active
            else f"Le modèle n'est pas chargé en ce moment, j'utilise une valeur de secours de {p_base:.2f} kW."
        )
        return f"Le modèle LSTM apprend les habitudes de consommation heure par heure (erreur moyenne d'environ 0,60 kW). {etat}"

    if has(q, tokens, ["bonjour", "salut", "hello", "bonsoir"]):
        return "Bonjour Farah. Dites-moi ce que vous voulez savoir sur la voiture, la clim ou la sécurité du réseau."

    return (
        "Je peux vous parler de la voiture, de la climatisation, de la sécurité du réseau ou du modèle de prévision. "
        "Pour les autres sujets, il faut renseigner une clé Gemini dans le panneau de gauche."
    )


def contexte_gemini():
    decision = (
        f"La demande dépasse le disjoncteur : la voiture est décalée à {hm(debut_ve)} en heures creuses."
        if surcharge
        else "Pas de surcharge : la voiture charge dès 18h00."
    )
    return f"""Tu es l'assistant de gestion d'énergie de la maison de Farah. Tu parles comme un gestionnaire de maison attentionné et compétent : naturel, chaleureux, direct.

STYLE :
- Français, 2 à 4 phrases, prose simple.
- Aucun emoji, aucun symbole décoratif, aucun titre, pas de listes sauf si on te les demande.
- Parle d'heures, de confort et de sécurité plutôt que de jargon technique ; explique un terme technique si Farah le demande.

DONNÉES EN DIRECT :
- Consommation habituelle prévue par le modèle LSTM : {p_base} kW
- Disjoncteur : {seuil_max} kW. Puissance totale réclamée : {p_totale} kW (dont clim {p_clim} kW et borne {p_ve} kW).
- Chambre : {temp_interieure}°C, consigne {temp_cible}°C pour 18h00. Clim : {duree_clim_min} min, allumage à {hm(debut_clim)}.
- Voiture : batterie à {soc_actuel}%, cible {soc_cible}% en {duree_charge_h} h, prête vers {fmt(fin_ve)} (échéance 07h30).
- Pour 100 % : {duree_100_h} h de charge, batterie pleine vers {fmt(fin_100)}.
- Décision : {decision}
- Pic après arbitrage : {pic_apres:.2f} kW.

RÈGLES :
1. Si on demande quand la voiture sera pleine ou utilisable, donne l'heure précise ci-dessus.
2. Pour une question générale (histoire, culture, actualité), réponds correctement et brièvement, puis glisse en une phrase l'état de la maison.
3. N'invente aucune donnée absente de la liste."""


def _post_gemini(nom, key, payload):
    return requests.post(
        f"{API_BASE}/models/{nom}:generateContent",
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        json=payload,
        timeout=20,
    )


def _detail_erreur(r):
    try:
        return r.json()["error"]["message"][:140]
    except Exception:
        return f"code {r.status_code}"


def appeler_gemini(messages, key):
    """Retourne (texte, None) ou (None, message_erreur)."""
    contents = []
    for m in messages[-9:]:
        role = "user" if m["role"] == "user" else "model"
        if not contents and role == "model":
            continue
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    payload = {
        "systemInstruction": {"parts": [{"text": contexte_gemini()}]},
        "contents": contents,
    }

    deja_ok = st.session_state.get("modele_gemini_ok")
    candidats = [deja_ok] if deja_ok else list(MODELES_GEMINI)
    derniere_erreur = "aucune réponse"

    def essayer(liste):
        nonlocal derniere_erreur
        for nom in liste:
            try:
                r = _post_gemini(nom, key, payload)
            except Exception as e:
                derniere_erreur = type(e).__name__
                return None, derniere_erreur, True
            if r.status_code == 200:
                try:
                    texte = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    derniere_erreur = "réponse vide"
                    continue
                st.session_state["modele_gemini_ok"] = nom
                return texte, None, False
            derniere_erreur = _detail_erreur(r)
            if r.status_code != 404:   # clé invalide, quota... inutile d'insister
                return None, derniere_erreur, True
        return None, derniere_erreur, False

    texte, err, stop = essayer(candidats)
    if texte or stop:
        return texte, err

    # Dernier recours : demander à Google quels modèles ce compte peut utiliser
    try:
        r = requests.get(f"{API_BASE}/models", headers={"x-goog-api-key": key}, timeout=15)
        if r.status_code == 200:
            dispo = [
                m["name"].split("/")[-1]
                for m in r.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
                and "flash" in m["name"]
                and "lite" not in m["name"]
                and m["name"].split("/")[-1] not in candidats
            ]
            texte, err, _ = essayer(dispo[:3])
            if texte:
                return texte, None
    except Exception:
        pass
    return None, derniere_erreur


def afficher(container, role, contenu):
    nom = "Vous" if role == "user" else "Assistant"
    with container.chat_message(role):
        st.markdown(f"<div class='who'>{nom}</div>", unsafe_allow_html=True)
        st.markdown(contenu)


# =====================================================================
# 6. DASHBOARD
# =====================================================================
st.markdown(
    html(
        """
<div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px;">
<h3 style="margin:0; color:#0F1E36; font-size:1.35rem; font-weight:700;">Smart Home Energy Orchestration</h3>
<span style="font-size:0.8rem; color:#64748B;">Supervision prédictive multi-agents et Deep Learning (LSTM)</span>
</div>
"""
    ),
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    src = "Prévision du modèle LSTM" if ia_active else "Valeur de secours"
    st.markdown(
        html(f"""
<div class="user-card">
<div class="card-category">Maison en arrière-plan</div>
<div class="card-human-main">{activite}</div>
<div class="card-technical-sub">{src} : {p_base:.2f} kW</div>
</div>"""),
        unsafe_allow_html=True,
    )

with c2:
    if duree_clim_min > 0:
        principal = "Chambre fraîche à 18h00"
        sub = f"Allumage à {hm(debut_clim)}, {duree_clim_min} min de clim ({p_clim:.1f} kW)"
    else:
        principal = "Chambre déjà confortable"
        sub = f"{temp_interieure}°C, consigne de {temp_cible}°C : clim inutile"
    st.markdown(
        html(f"""
<div class="user-card">
<div class="card-category">Confort</div>
<div class="card-human-main">{principal}</div>
<div class="card-technical-sub">{sub}</div>
</div>"""),
        unsafe_allow_html=True,
    )

with c3:
    couleur_ve = "#0F1E36"
    if duree_charge_h == 0:
        principal = "Rien à recharger"
        sub = f"Batterie à {soc_actuel}%, cible de {soc_cible}% atteinte"
    elif ve_a_temps:
        principal = f"Voiture prête à {hm(fin_ve)}"
        sub = f"{duree_charge_h} h de charge vers {soc_cible}%, départ à {hm(debut_ve)} ({p_ve:.1f} kW)"
    else:
        principal = f"Voiture prête à {hm(fin_ve)}"
        sub = f"Trop tard pour 07h30, démarrer avant {fmt(debut_ve_requis)}"
        couleur_ve = "#B91C1C"
    st.markdown(
        html(f"""
<div class="user-card">
<div class="card-category">Voiture électrique</div>
<div class="card-human-main" style="color:{couleur_ve};">{principal}</div>
<div class="card-technical-sub">{sub}</div>
</div>"""),
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        html(f"""
<div class="user-card" style="border-left:4px solid {couleur};">
<div class="card-category">Sécurité du réseau</div>
<div class="card-human-main" style="color:{couleur};">{statut}</div>
<div class="card-technical-sub">Limite de {seuil_max:.1f} kW, demande de {p_totale:.2f} kW</div>
</div>"""),
        unsafe_allow_html=True,
    )

st.write("")

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
        except TypeError:
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr style='margin:6px 0;'>", unsafe_allow_html=True)

        if not surcharge:
            st.markdown(
                html(f"""
<div style="font-size:0.88rem; line-height:1.5; color:#334155;">
<div style="color:{couleur}; font-weight:700; margin-bottom:4px;">Rien à signaler</div>
La maison réclame {p_totale:.2f} kW pour {seuil_max:.1f} kW autorisés : il reste {marge:.2f} kW de marge.
La clim et la recharge de la voiture peuvent fonctionner en même temps.
</div>"""),
                unsafe_allow_html=True,
            )
        else:
            titre = (
                "Le disjoncteur reste menacé"
                if critique
                else "J'ai évité une coupure de courant"
            )
            if critique:
                suite = (
                    f"Même après décalage, le pic atteint {pic_apres:.2f} kW pour {seuil_max:.1f} kW autorisés. "
                    f"Passer à {souscription_conseillee:.1f} kW de souscription règlerait le problème."
                )
            else:
                suite = f"Le pic redescend à {pic_apres:.2f} kW, sous votre limite."
            st.markdown(
                html(f"""
<div style="font-size:0.88rem; line-height:1.5; color:#334155;">
<div style="color:{couleur}; font-weight:700; margin-bottom:4px;">{titre}</div>
Tout lancer en même temps demanderait {p_totale:.2f} kW, soit {surcroit:.2f} kW de plus que votre disjoncteur ({seuil_max:.1f} kW).
La clim démarre à {hm(debut_clim)} pour que la chambre soit à {temp_cible}°C à 18h00.
La voiture ne recharge qu'à partir de {hm(debut_ve)}, en heures creuses, et sera prête vers {fmt(fin_ve)}.
{suite}
</div>"""),
                unsafe_allow_html=True,
            )

with col_droite:
    with st.container(border=True):
        mode = "Gemini" if api_key else "Mode local"
        st.markdown(
            html(f"""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
<strong style="color:#0F1E36; font-size:0.92rem;">Assistant de supervision</strong>
<span style="font-size:0.75rem; color:#64748B;">{mode}</span>
</div>"""),
            unsafe_allow_html=True,
        )

        chat_box = st.container(height=265)

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Bonjour Farah. Je suis là pour vous expliquer ce que fait la maison : la recharge de la voiture, la fraîcheur de la chambre ou la protection du disjoncteur. Que voulez-vous savoir ?",
                }
            ]

        for msg in st.session_state.messages:
            afficher(chat_box, msg["role"], msg["content"])

        if prompt := st.chat_input("Écrivez votre message..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            afficher(chat_box, "user", prompt)

            rep, erreur = None, None
            if api_key:
                with chat_box:
                    with st.spinner("Je réfléchis..."):
                        rep, erreur = appeler_gemini(st.session_state.messages, api_key)
            if rep is None:
                rep = repondre_local(prompt)
            st.session_state.messages.append({"role": "assistant", "content": rep})

            affichage = rep
            if erreur:
                affichage += f"\n\n*Gemini ne répond pas ({erreur}). Réponse en mode local.*"
            afficher(chat_box, "assistant", affichage)
