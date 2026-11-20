import math
import re
from datetime import datetime, timedelta

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

# =====================================================================
# 0. CONSTANTES MÉTIER
# =====================================================================
MODELES_GEMINI = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
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
# 1. PAGE & DESIGN
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

/* Fond sable + trame de plan d'architecte très discrète */
.stApp {
    color: #0F1E36;
    background-color: #EDE5D6;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='44' height='44'%3E%3Cpath d='M44 0H0V44' fill='none' stroke='%230F1E36' stroke-opacity='0.055' stroke-width='1'/%3E%3C/svg%3E"),
        linear-gradient(180deg, #E9E0CE 0%, #F3EDE1 55%, #EDE5D6 100%);
}

.block-container {
    padding-top: 0.9rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F1E36 0%, #162B4A 100%) !important;
    border-right: 2px solid #B08D57;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] small { color: #F3EDE1 !important; }
section[data-testid="stSidebar"] input {
    color: #0F1E36 !important;
    background-color: #F3EDE1 !important;
}
section[data-testid="stSidebar"] button p { color: #0F1E36 !important; }

/* Bannière */
.banner {
    display: flex; justify-content: space-between; align-items: center;
    background: linear-gradient(100deg, #0F1E36 0%, #1B3A5C 100%);
    border-radius: 12px;
    padding: 12px 22px;
    margin-bottom: 12px;
    border-bottom: 3px solid #B08D57;
    box-shadow: 0 3px 10px rgba(15, 30, 54, 0.18);
}
.banner h3 { margin: 0; color: #F6F1E7; font-size: 1.3rem; font-weight: 700; }
.banner p { margin: 2px 0 0 0; color: #B9C4D3; font-size: 0.8rem; }

/* Cartes */
.user-card {
    background: #FFFFFF;
    border: 1px solid #D6CBB8;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 2px 8px rgba(15, 30, 54, 0.10);
    min-height: 108px;
}
.card-row { display: flex; gap: 12px; align-items: flex-start; }
.icon-badge {
    flex: 0 0 38px; height: 38px; border-radius: 50%;
    background: #E1E9F2; color: #1D3557;
    display: flex; align-items: center; justify-content: center;
}
.icon-badge svg { width: 20px; height: 20px; fill: none; stroke: currentColor;
    stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
.card-category {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.09em;
    font-weight: 700; color: #3F5F82; margin-bottom: 3px;
}
.card-human-main { font-size: 1.25rem; font-weight: 800; line-height: 1.2; margin-bottom: 5px; }
.card-technical-sub { font-size: 0.78rem; color: #475569; font-weight: 500; }

/* Conteneurs natifs */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #D6CBB8 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    box-shadow: 0 2px 8px rgba(15, 30, 54, 0.10) !important;
}

/* Chat : sans avatars, messages bien distincts */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }
.stChatMessage {
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin-bottom: 6px !important;
    font-size: 0.88rem !important;
    background-color: #F7F3EB !important;
    border: 1px solid #E2D9C8 !important;
    border-left: 3px solid #B08D57 !important;
}
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
    background-color: #E1E9F2 !important;
    border: 1px solid #C9D6E6 !important;
    border-left: 3px solid #1D3557 !important;
}
.who {
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: #3F5F82; margin-bottom: 2px;
}

#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""",
    unsafe_allow_html=True,
)


def html(s: str) -> str:
    """Aplatit le HTML (sans coller les phrases) pour éviter le rendu en bloc de code."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


ICONES = {
    "maison": '<path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/>',
    "thermo": '<path d="M14 14.8V5a2 2 0 0 0-4 0v9.8a4 4 0 1 0 4 0z"/>',
    "voiture": '<path d="M5 16l1.5-5a2 2 0 0 1 1.9-1.4h7.2a2 2 0 0 1 1.9 1.4L19 16"/><rect x="3" y="16" width="18" height="4" rx="1.5"/><path d="M7 18h.01M17 18h.01"/>',
    "bouclier": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
}


def carte(icone, categorie, principal, sub, couleur="#0F1E36", accent="#1D3557"):
    return html(f"""
<div class="user-card" style="border-top:3px solid {accent};">
<div class="card-row">
<div class="icon-badge"><svg viewBox="0 0 24 24">{ICONES[icone]}</svg></div>
<div>
<div class="card-category">{categorie}</div>
<div class="card-human-main" style="color:{couleur};">{principal}</div>
<div class="card-technical-sub">{sub}</div>
</div>
</div>
</div>""")


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
# 3. GEMINI (essai de plusieurs modèles + diagnostic lisible)
# =====================================================================
def _post_gemini(nom, key, payload):
    return requests.post(
        f"{API_BASE}/models/{nom}:generateContent",
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        json=payload,
        timeout=25,
    )


def _detail_erreur(r):
    try:
        msg = r.json()["error"]["message"]
    except Exception:
        msg = ""
    return f"code {r.status_code}" + (f" : {msg[:110]}" if msg else "")


def appeler_gemini(messages, key, systeme=""):
    """Retourne (texte, None) ou (None, raison). Mémorise l'état dans la session."""
    contents = []
    for m in messages[-9:]:
        role = "user" if m["role"] == "user" else "model"
        if not contents and role == "model":
            continue
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    payload = {"contents": contents}
    if systeme:
        payload["systemInstruction"] = {"parts": [{"text": systeme}]}

    deja_ok = st.session_state.get("modele_gemini_ok")
    candidats = [deja_ok] if deja_ok else list(MODELES_GEMINI)
    erreurs = []

    def essayer(liste):
        for nom in liste:
            try:
                r = _post_gemini(nom, key, payload)
            except Exception as e:
                erreurs.append(f"réseau ({type(e).__name__})")
                return None, True
            if r.status_code == 200:
                try:
                    data = r.json()
                    parts = data["candidates"][0]["content"]["parts"]
                    texte = "".join(p.get("text", "") for p in parts).strip()
                except Exception:
                    texte = ""
                if texte:
                    st.session_state["modele_gemini_ok"] = nom
                    return texte, False
                erreurs.append(f"{nom} : réponse vide ou bloquée")
                continue
            erreurs.append(_detail_erreur(r))
            if r.status_code in (400, 401, 403):  # clé refusée : inutile d'insister
                return None, True
        return None, False

    texte, stop = essayer(candidats)
    if not texte and not stop:
        # Dernier recours : demander à Google quels modèles ce compte peut utiliser
        try:
            r = requests.get(f"{API_BASE}/models", headers={"x-goog-api-key": key}, timeout=15)
            if r.status_code == 200:
                dispo = [
                    m["name"].split("/")[-1]
                    for m in r.json().get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                    and "flash" in m["name"]
                    and m["name"].split("/")[-1] not in candidats
                ]
                texte, _ = essayer(dispo[:3])
        except Exception:
            pass

    if texte:
        st.session_state["gemini_statut"] = ("ok", st.session_state.get("modele_gemini_ok", ""))
        return texte, None
    non404 = [e for e in erreurs if "code 404" not in e]
    raison = (non404 or erreurs or ["aucune réponse"])[0]
    st.session_state["gemini_statut"] = ("erreur", raison)
    return None, raison


# =====================================================================
# 4. SIDEBAR
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
    if api_key and st.button("Tester la connexion Gemini"):
        t, e = appeler_gemini([{"role": "user", "content": "Réponds uniquement : ok."}], api_key)
        st.caption("Connexion réussie." if t else f"Échec : {e}")

    seuil_max = st.slider("Souscription disjoncteur (kW)", 3.0, 12.0, 6.0, 0.5)
    temp_interieure = st.slider("Température ambiante (°C)", 20.0, 38.0, 30.0, 0.5)
    temp_cible = st.slider("Consigne souhaitée (°C)", 18.0, 26.0, 21.0, 0.5)
    soc_actuel = st.slider("Batterie actuelle (%)", 5, 100, 35, 5)
    soc_cible = st.slider("Batterie cible (%)", 50, 100, 80, 5)


# =====================================================================
# 5. MOTEUR DE CALCUL
# =====================================================================
JOUR0 = datetime(2000, 1, 1)


def at(h, m=0, jour=0):
    return JOUR0 + timedelta(days=jour, hours=h, minutes=m)


def hm(dt):
    return dt.strftime("%Hh%M")


def fmt(dt):
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
souscription_conseillee = math.ceil(pic_apres * 2) / 2

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
# 6. ASSISTANT : MOTEUR LOCAL + CONTEXTE GEMINI
# =====================================================================
def has(q, tokens, mots):
    return any((m in q) if len(m) > 3 else (m in tokens) for m in mots)


def repondre_local(prompt, raison=None):
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

    if raison:
        return (
            "Cette question dépasse le cadre de la maison et je n'ai pas réussi à joindre Gemini "
            f"({raison}). Utilisez le bouton « Tester la connexion Gemini » dans le panneau de gauche pour vérifier la clé."
        )
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


def afficher(container, role, contenu):
    nom = "Vous" if role == "user" else "Assistant"
    with container.chat_message(role):
        st.markdown(f"<div class='who'>{nom}</div>", unsafe_allow_html=True)
        st.markdown(contenu)


def entete_chat(api_key):
    if not api_key:
        etat = "Mode local"
    else:
        s = st.session_state.get("gemini_statut")
        if s is None:
            etat = "Gemini (clé saisie)"
        elif s[0] == "ok":
            etat = f"Gemini connecté ({s[1]})"
        else:
            etat = "Gemini injoignable, mode local"
    return html(f"""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
<strong style="color:#0F1E36; font-size:0.92rem;">Assistant de supervision</strong>
<span style="font-size:0.75rem; color:#475569;">{etat}</span>
</div>""")


# =====================================================================
# 7. DASHBOARD
# =====================================================================
st.markdown(
    html("""
<div class="banner">
<div>
<h3>Smart Home Energy Orchestration</h3>
<p>Supervision prédictive multi-agents et Deep Learning (LSTM)</p>
</div>
<svg width="170" height="54" viewBox="0 0 170 54" fill="none" stroke="#C9B58F" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M8 30 L38 8 L68 30"/><path d="M14 27 V48 H62 V27"/><rect x="26" y="32" width="10" height="16"/><rect x="44" y="32" width="10" height="9"/>
<path d="M82 40 C92 40 94 16 105 16 S118 42 129 32 S146 14 162 22" stroke="#7FA6CF"/>
<path d="M82 48 H164" stroke-opacity="0.5"/>
</svg>
</div>
"""),
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    src = "Prévision du modèle LSTM" if ia_active else "Valeur de secours"
    st.markdown(
        carte("maison", "Maison en arrière-plan", activite, f"{src} : {p_base:.2f} kW"),
        unsafe_allow_html=True,
    )

with c2:
    if duree_clim_min > 0:
        principal = "Chambre fraîche à 18h00"
        sub = f"Allumage à {hm(debut_clim)}, {duree_clim_min} min de clim ({p_clim:.1f} kW)"
    else:
        principal = "Chambre déjà confortable"
        sub = f"{temp_interieure}°C, consigne de {temp_cible}°C : clim inutile"
    st.markdown(carte("thermo", "Confort", principal, sub), unsafe_allow_html=True)

with c3:
    couleur_ve, accent_ve = "#0F1E36", "#1D3557"
    if duree_charge_h == 0:
        principal = "Rien à recharger"
        sub = f"Batterie à {soc_actuel}%, cible de {soc_cible}% atteinte"
    elif ve_a_temps:
        principal = f"Voiture prête à {hm(fin_ve)}"
        sub = f"{duree_charge_h} h de charge vers {soc_cible}%, départ à {hm(debut_ve)} ({p_ve:.1f} kW)"
    else:
        principal = f"Voiture prête à {hm(fin_ve)}"
        sub = f"Trop tard pour 07h30, démarrer avant {fmt(debut_ve_requis)}"
        couleur_ve, accent_ve = "#B91C1C", "#B91C1C"
    st.markdown(
        carte("voiture", "Voiture électrique", principal, sub, couleur_ve, accent_ve),
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        carte(
            "bouclier",
            "Sécurité du réseau",
            statut,
            f"Limite de {seuil_max:.1f} kW, demande de {p_totale:.2f} kW",
            couleur,
            couleur,
        ),
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
                    "axis": {"range": [0, 12], "tickwidth": 1, "tickcolor": "#94A3B8", "nticks": 7},
                    "bar": {"color": "#B91C1C" if surcharge else "#1B4D3E"},
                    "bgcolor": "#F4EFEA",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, seuil_max], "color": "#E3DACA"},
                        {"range": [seuil_max, 12], "color": "#F6CFCB"},
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
<p style="margin:0 0 4px 0;">La maison réclame {p_totale:.2f} kW pour {seuil_max:.1f} kW autorisés : il reste {marge:.2f} kW de marge.</p>
<p style="margin:0;">La clim et la recharge de la voiture peuvent fonctionner en même temps.</p>
</div>"""),
                unsafe_allow_html=True,
            )
        else:
            titre = "Le disjoncteur reste menacé" if critique else "J'ai évité une coupure de courant"
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
<p style="margin:0 0 4px 0;">Tout lancer en même temps demanderait {p_totale:.2f} kW, soit {surcroit:.2f} kW de plus que votre disjoncteur ({seuil_max:.1f} kW).</p>
<p style="margin:0 0 4px 0;">La clim démarre à {hm(debut_clim)} pour que la chambre soit à {temp_cible}°C à 18h00. La voiture ne recharge qu'à partir de {hm(debut_ve)}, en heures creuses, et sera prête vers {fmt(fin_ve)}.</p>
<p style="margin:0;">{suite}</p>
</div>"""),
                unsafe_allow_html=True,
            )

with col_droite:
    with st.container(border=True):
        entete_slot = st.empty()
        entete_slot.markdown(entete_chat(api_key), unsafe_allow_html=True)

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
                        rep, erreur = appeler_gemini(
                            st.session_state.messages, api_key, contexte_gemini()
                        )
            if rep is None:
                rep = repondre_local(prompt, erreur)
            st.session_state.messages.append({"role": "assistant", "content": rep})
            afficher(chat_box, "assistant", rep)
            entete_slot.markdown(entete_chat(api_key), unsafe_allow_html=True)

# Fait défiler la zone de chat jusqu'au dernier message
try:
    components.html(
        """
<script>
(function () {
  const d = window.parent.document;
  function go() {
    const msgs = d.querySelectorAll('.stChatMessage');
    if (!msgs.length) return;
    let el = msgs[msgs.length - 1];
    while (el) {
      const oy = window.parent.getComputedStyle(el).overflowY;
      if ((oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight) {
        el.scrollTop = el.scrollHeight;
        break;
      }
      el = el.parentElement;
    }
  }
  setTimeout(go, 100); setTimeout(go, 500);
})();
</script>
""",
        height=0,
    )
except Exception:
    pass
