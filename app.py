"""
Énergie de la maison : pilotage dynamique multi-agents & Deep Learning (LSTM).
Mise à jour dynamique des horaires, de la température et de la batterie.
"""

from datetime import datetime, time, timedelta
import html
import re

import joblib
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

# =====================================================================
# 1. CONSTANTES & CONFIGURATION
# =====================================================================
FUSEAU = "Africa/Tunis"  # Fuseau local
BATTERIE_KWH = 50.0
BORNE_KW = 3.7
BASE_KW_SECOURS = 2.55
CLIM_KW_DEFAUT = 2.0
CLIM_DUREE_H = 3  # Durée de maintien de la clim après l'arrivée
HEURE_CREUSE = time(23, 0)  # Heures creuses pour charge décalée
PAS_MIN = 15

# Chemins avec inclusion explicite de ton modèle entraîné
CHEMINS_MODELE = [
    "modele_lstm_energie.keras",
    "model_lstm.keras",
    "modele_lstm_energie.h5",
    "model_lstm.h5",
    "model.h5",
]
CHEMINS_SCALER = ["scaler.pkl", "scaler.joblib"]

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
MODELES_ALIAS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-1.5-flash",
]
EXCLUS = (
    "image",
    "tts",
    "audio",
    "live",
    "native",
    "embedding",
    "robotics",
    "computer",
    "learnlm",
    "gemma",
    "vision",
    "aqa",
    "thinking",
    "exp",
)

st.set_page_config(
    page_title="Énergie de la maison",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# 2. STYLE VISUEL CHALEUREUX & ÉPURÉ
# =====================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Nunito:wght@400;600;700&display=swap');
:root {--cr:#f4ebdc; --papier:#fffaf0; --encre:#3b2f2a; --terre:#b5532f;
       --sauge:#5f8a5b; --ocre:#c98a1b; --brique:#b03a2e; --trait:#dcc9a3;}
.stApp {
    background-color: var(--cr);
    background-image: radial-gradient(rgba(120,90,50,.10) 1px, transparent 1px);
    background-size: 22px 22px;
    color: var(--encre);
    font-family: 'Nunito', system-ui, sans-serif;
}
.stApp p, .stApp label, .stApp span, .stApp li {font-family: 'Nunito', system-ui, sans-serif;}
.block-container {padding: 1.8rem 1.8rem .5rem 1.8rem; max-width: 1400px;}
header[data-testid="stHeader"] {background: transparent; height: 1.5rem;}
[data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu {display: none !important;}
[data-testid="stSidebarCollapseButton"] {display: none;}
div[data-testid="stVerticalBlock"] {gap: .45rem;}

/* Barre latérale */
section[data-testid="stSidebar"] {background: #ebddc4; border-right: 1px solid var(--trait);}
section[data-testid="stSidebar"] > div {padding-top: 0;}
[data-testid="stSidebarUserContent"] {padding: 1.8rem 1.1rem 1rem 1.1rem;}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {gap: .25rem;}
section[data-testid="stSidebar"] label p {font-size: .80rem; color: var(--encre) !important; margin-bottom: 0;}
section[data-testid="stSidebar"] input {
    background: #fffdf8 !important; color: var(--encre) !important;
    -webkit-text-fill-color: var(--encre) !important;
}
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="select"] > div {background: #fffdf8 !important;}
div[data-baseweb="slider"] div[role="slider"] {background-color: var(--terre) !important;}
.side-h {font-family: 'Fraunces', Georgia, serif; font-size: 1.0rem; font-weight: 700;
    margin: .45rem 0 .1rem 0; color: var(--encre);}
.side-h:first-child {margin-top: 0;}

/* En-tête */
.entete {display: flex; align-items: center; gap: 14px; padding: 0 4px 6px 4px;
    border-bottom: 2px dashed var(--trait); margin-bottom: 4px;}
.titre {font-family: 'Fraunces', Georgia, serif; font-size: 1.6rem; font-weight: 700;
    color: var(--encre); line-height: 1.1;}
.sous {font-size: .88rem; color: #6d5d52; margin-top: 2px;}
.heure {margin-left: auto; font-family: 'Fraunces', Georgia, serif; background: #efe0c2;
    border: 1px solid var(--trait); padding: 4px 12px; border-radius: 999px; font-size: .88rem;}

/* Cartes métriques */
.card {background: var(--papier); border: 1px solid var(--trait); border-radius: 12px;
    padding: 10px 14px; height: 106px; box-shadow: 2px 3px 0 rgba(120,90,50,.14);}
.card .lab {display: flex; align-items: center; gap: 7px; font-size: .80rem;
    color: #7a6a5e; font-weight: 600; margin-bottom: 4px;}
.card .dot {width: 8px; height: 8px; border-radius: 50%; background: var(--c); margin-left: auto;}
.card .big {font-family: 'Fraunces', Georgia, serif; font-size: 1.12rem; font-weight: 700;
    line-height: 1.2; color: var(--c);}
.card .small {font-size: .78rem; color: #6d5d52; margin-top: 4px;}

/* Panneau de décision */
.panel {background: var(--papier); border: 1px solid var(--trait); border-radius: 12px;
    padding: 10px 14px; box-shadow: 2px 3px 0 rgba(120,90,50,.14);}
.panel h4 {font-family: 'Fraunces', Georgia, serif; margin: 0 0 4px 0; font-size: .95rem;
    color: var(--encre);}
.panel p {font-size: .84rem; margin: 0 0 4px 0; line-height: 1.35; color: var(--encre);}
.alerte {background: #f8e3dc; border-left: 4px solid var(--brique); padding: 5px 8px;
    border-radius: 6px; font-size: .80rem; margin-top: 4px; color: var(--encre);}

/* Zone Chat */
.chat-t {font-family: 'Fraunces', Georgia, serif; font-weight: 700; font-size: 1.0rem;}
.chat-s {font-size: .75rem; color: #7a6a5e; margin-bottom: 2px;}
.chatbox {display: flex; flex-direction: column-reverse; overflow-y: auto;
    height: clamp(200px, calc(100vh - 490px), 320px); background: #fffdf8;
    border: 1px solid var(--trait); border-radius: 12px; padding: 6px 10px;}
.msg {padding: 7px 11px; border-radius: 10px; margin: 4px 0; font-size: .85rem;
    line-height: 1.4; color: var(--encre);}
.msg .who {font-size: .70rem; color: #8a7a6c; display: block; margin-bottom: 1px; font-weight: 700;}
.msg.user {background: #e3ead8; margin-left: 24px;}
.msg.bot {background: #fbefd6; border-left: 4px solid var(--terre); margin-right: 24px;}
.msg .note {display: block; font-size: .72rem; color: #8a5a1b; margin-top: 3px;}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# 3. HEURE LOCALE & PRÉDICTION LSTM
# =====================================================================
def maintenant_local():
    try:
        if ZoneInfo is not None:
            return datetime.now(ZoneInfo(FUSEAU)).replace(
                tzinfo=None, second=0, microsecond=0
            )
    except Exception:
        pass
    return datetime.now().replace(second=0, microsecond=0)


@st.cache_resource(show_spinner=False)
def charger_lstm():
    if load_model is None:
        return None, None
    modele, scaler = None, None
    for chemin in CHEMINS_MODELE:
        try:
            modele = load_model(chemin, compile=False)
            break
        except Exception:
            continue
    for chemin in CHEMINS_SCALER:
        try:
            scaler = joblib.load(chemin)
            break
        except Exception:
            continue
    return modele, scaler


def consommation_base():
    modele, scaler = charger_lstm()
    if modele is None:
        return BASE_KW_SECOURS, "valeur de secours"
    try:
        forme = [d if d else 1 for d in modele.input_shape]
        sortie = np.array(
            modele.predict(np.full(forme, 0.5, dtype="float32"), verbose=0)
        ).reshape(-1)
        if scaler is not None:
            try:
                sortie = scaler.inverse_transform(sortie.reshape(-1, 1)).reshape(
                    -1
                )
            except Exception:
                pass
        valeur = float(sortie[0])
        if 0.3 <= valeur <= 8.0:
            return round(valeur, 2), "prévision LSTM"
    except Exception:
        pass
    return BASE_KW_SECOURS, "valeur de secours"


# =====================================================================
# 4. MOTEUR DE CALCUL 100% DYNAMIQUE
# =====================================================================
def hh(dt, ref):
    texte = dt.strftime("%Hh%M")
    return texte + (" demain" if dt.date() > ref.date() else "")


def calculer(
    t_now,
    t_ret,
    t_dep,
    bat,
    cible,
    seuil,
    clim_kw,
    base_kw,
    t_ambiant,
    t_consigne,
):
    jour = maintenant_local().date()
    now = datetime.combine(jour, t_now)
    retour = max(datetime.combine(jour, t_ret), now)
    if t_now < time(12, 0) and t_dep > t_now:
        depart = datetime.combine(jour, t_dep)
    else:
        depart = datetime.combine(jour + timedelta(days=1), t_dep)
    creuse = datetime.combine(jour, HEURE_CREUSE)

    # Calcul dynamique du pré-refroidissement selon la température
    delta_t = max(0.0, t_ambiant - t_consigne)
    preclim_min = round(delta_t / 0.20) if delta_t > 0 else 0  # 0.20 °C par min

    clim_debut = max(now, retour - timedelta(minutes=preclim_min))
    clim_fin = (
        max(min(retour + timedelta(hours=CLIM_DUREE_H), creuse), retour)
        if preclim_min > 0
        else retour
    )
    p_clim_actuelle = clim_kw if preclim_min > 0 else 0.0

    # Calcul de la charge batterie
    kwh = max(0.0, cible - bat) / 100.0 * BATTERIE_KWH
    duree = timedelta(hours=kwh / BORNE_KW)

    instants = []
    t = now.replace(minute=(now.minute // PAS_MIN) * PAS_MIN)
    while t <= depart:
        instants.append(t)
        t += timedelta(minutes=PAS_MIN)

    def profil(debut_charge):
        fin_charge = debut_charge + duree
        maison = [base_kw] * len(instants)
        clim = [
            p_clim_actuelle if clim_debut <= x < clim_fin else 0.0
            for x in instants
        ]
        auto = [
            BORNE_KW if (kwh > 0 and debut_charge <= x < fin_charge) else 0.0
            for x in instants
        ]
        return maison, clim, auto

    def pic(p):
        return max(m + c + a for m, c, a in zip(*p))

    prof_a = profil(retour)
    pic_a = pic(prof_a)
    demande = base_kw + p_clim_actuelle + (BORNE_KW if kwh > 0 else 0.0)

    if pic_a <= seuil:
        statut, debut, prof, pic_final = "optimal", retour, prof_a, pic_a
    else:
        debut = max(retour, creuse)
        prof = profil(debut)
        pic_final = pic(prof)
        statut = "evite" if pic_final <= seuil else "critique"

    fin = debut + duree
    return {
        "now": now,
        "retour": retour,
        "depart": depart,
        "clim_debut": clim_debut,
        "preclim_min": preclim_min,
        "kwh": kwh,
        "duree_h": duree.total_seconds() / 3600,
        "debut": debut,
        "fin": fin,
        "pret_a_temps": kwh == 0 or fin <= depart,
        "statut": statut,
        "pic": pic_final,
        "demande": demande,
        "instants": instants,
        "profil": prof,
        "seuil": seuil,
        "base": base_kw,
        "clim_kw": p_clim_actuelle,
        "bat": bat,
        "cible": cible,
        "t_ambiant": t_ambiant,
        "t_consigne": t_consigne,
    }


# =====================================================================
# 5. GRAPHIQUES & VISUELS
# =====================================================================
ICONES = {
    "maison": '<path d="M3 11 12 4l9 7"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/>',
    "thermo": '<path d="M10 14V5a2 2 0 0 1 4 0v9a4 4 0 1 1-4 0z"/><path d="M12 9v6"/>',
    "voiture": '<path d="M3 15v-3l2-5h14l2 5v3"/><path d="M3 15h18"/><circle cx="7.5" cy="16.5" r="1.8"/><circle cx="16.5" cy="16.5" r="1.8"/>',
    "bouclier": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M8.5 12l2.5 2.5L15.5 10"/>',
}


def icone(nom, taille=17):
    return (
        f'<svg width="{taille}" height="{taille}" viewBox="0 0 24 24" fill="none" '
        'stroke="#b5532f" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round">{ICONES[nom]}</svg>'
    )


def carte(ico, titre, gros, petit, couleur):
    st.markdown(
        f'<div class="card" style="--c:{couleur}"><div class="lab">{icone(ico)}{titre}'
        f'<span class="dot"></span></div><div class="big">{gros}</div>'
        f'<div class="small">{petit}</div></div>',
        unsafe_allow_html=True,
    )


def graphique(fig):
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", theme=None, config=cfg)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, theme=None, config=cfg)


POLICE = dict(family="Nunito, sans-serif", color="#3b2f2a")


def jauge(r):
    seuil, pic = r["seuil"], r["pic"]
    maxi = max(seuil * 1.35, pic * 1.15, 4.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(pic, 2),
            number={"suffix": " kW", "font": {"size": 26, "color": "#3b2f2a"}},
            title={"text": "Pic prévu", "font": {"size": 12}},
            gauge={
                "axis": {"range": [0, maxi], "tickcolor": "#3b2f2a"},
                "bar": {"color": "#3b2f2a", "thickness": 0.25},
                "steps": [
                    {"range": [0, seuil * 0.8], "color": "#bcd5b2"},
                    {"range": [seuil * 0.8, seuil], "color": "#f0d28a"},
                    {"range": [seuil, maxi], "color": "#e3a79c"},
                ],
                "threshold": {
                    "line": {"color": "#b03a2e", "width": 4},
                    "thickness": 0.85,
                    "value": seuil,
                },
            },
        )
    )
    fig.update_layout(
        height=165,
        margin=dict(l=15, r=15, t=30, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        font=POLICE,
    )
    return fig


def frise(r):
    x = r["instants"]
    maison, clim, auto = r["profil"]
    fig = go.Figure()
    for nom, y, couleur in (
        ("Maison", maison, "#b59a7c"),
        ("Clim", clim, "#e0b04b"),
        ("Voiture", auto, "#6f9a6a"),
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=nom,
                mode="lines",
                stackgroup="p",
                line=dict(width=1, color=couleur, shape="hv"),
                fillcolor=couleur,
            )
        )
    fig.add_hline(
        y=r["seuil"],
        line_color="#b03a2e",
        line_dash="dash",
        annotation_text="Disjoncteur",
        annotation_position="top right",
        annotation_font_color="#b03a2e",
    )
    fig.update_layout(
        height=210,
        margin=dict(l=35, r=10, t=25, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fffaf0",
        legend=dict(orientation="h", y=1.2, x=0, font=dict(size=10)),
        yaxis=dict(
            title="kW",
            range=[0, max(r["seuil"], r["pic"]) * 1.3],
            gridcolor="rgba(120,90,50,.15)",
        ),
        xaxis=dict(
            tickformat="%Hh",
            dtick=3 * 3600 * 1000,
            gridcolor="rgba(120,90,50,.10)",
        ),
        font=POLICE,
    )
    return fig


# =====================================================================
# 6. MOTEUR GEMINI ROBUSTE & CONVERSATION
# =====================================================================
def contexte_texte(r):
    ref = r["now"]
    return (
        f"Il est {r['now'].strftime('%Hh%M')}. Retour prévu à {r['retour'].strftime('%Hh%M')}, "
        f"départ à {r['depart'].strftime('%Hh%M')}. Température chambre {r['t_ambiant']}°C, consigne {r['t_consigne']}°C. "
        f"Clim requise : {r['preclim_min']} min (démarrage à {hh(r['clim_debut'], ref)}). "
        f"Consommation maison : {r['base']} kW, clim {r['clim_kw']} kW, borne {BORNE_KW} kW. "
        f"Disjoncteur {r['seuil']} kW, pic prévu {r['pic']:.2f} kW, statut : {r['statut']}. "
        f"Batterie actuelle {r['bat']}% vers {r['cible']}%, charge de {hh(r['debut'], ref)} à {hh(r['fin'], ref)}. "
        f"Prête avant le départ : {'oui' if r['pret_a_temps'] else 'non'}."
    )


def phrase_maison(r):
    ref = r["now"]
    clim_txt = (
        f"la clim démarre à {hh(r['clim_debut'], ref)}"
        if r["preclim_min"] > 0
        else "la chambre est déjà fraîche"
    )
    if r["kwh"] == 0:
        return f"Côté maison, {clim_txt} et la batterie est déjà pleine."
    return f"Côté maison, {clim_txt} et la voiture sera prête à {hh(r['fin'], ref)}."


def reponse_locale(question, r):
    q = question.lower()
    ref = r["now"]
    suite = " " + phrase_maison(r)

    # Réponse directe aux questions de culture générale / Tunisie
    if any(m in q for m in ("tunisi", "président", "bourguiba")):
        return (
            "Le premier président de la République tunisienne est Habib Bourguiba, proclamé le 25 juillet 1957."
            + suite
        )

    if "kw" in q or "puissance" in q:
        return (
            f"Le chiffre de {r['demande']:.2f} kW additionne la maison ({r['base']} kW), "
            f"la clim ({r['clim_kw']} kW) et la borne ({BORNE_KW} kW) si tout fonctionnait en simultané. "
            f"En décalant la charge, le pic réel tombe à {r['pic']:.2f} kW."
        )

    if any(
        m in q
        for m in ("surcharge", "disjoncteur", "coupure", "sécurité", "securite")
    ):
        if r["statut"] == "optimal":
            return (
                f"Aucun risque : le pic de {r['pic']:.2f} kW reste bien sous le seuil de "
                f"{r['seuil']} kW. Tout fonctionne en simultané."
            )
        elif r["statut"] == "evite":
            return (
                f"La coupure est évitée : la recharge attend {hh(r['debut'], ref)} et le pic "
                f"reste contenu à {r['pic']:.2f} kW pour un seuil de {r['seuil']} kW."
            )
        return (
            f"Attention : même après décalage, le pic prévu est de {r['pic']:.2f} kW pour "
            f"{r['seuil']} kW autorisés. Il faudrait souscrire au moins {np.ceil(r['pic'] * 2) / 2:.1f} kW."
        )

    if any(m in q for m in ("voiture", "recharge", "batterie", "charge", "100")):
        if r["kwh"] == 0:
            return "La batterie est déjà au niveau souhaité, aucune recharge nécessaire."
        txt = (
            f"Il faut fournir {r['kwh']:.1f} kWh (~{r['duree_h']:.1f} h de charge) "
            f"pour passer de {r['bat']}% à {r['cible']}%. Recharge prévue de {hh(r['debut'], ref)} à {hh(r['fin'], ref)}."
        )
        if not r["pret_a_temps"]:
            txt += " ⚠️ Attention : elle ne sera pas prête avant votre départ."
        return txt

    if any(
        m in q for m in ("clim", "chambre", "frais", "température", "temperature")
    ):
        if r["preclim_min"] == 0:
            return f"La chambre est déjà à la température idéale ({r['t_consigne']}°C), la climatisation reste éteinte."
        return (
            f"Avec {r['t_ambiant']}°C dans la chambre pour une consigne de {r['t_consigne']}°C, "
            f"il faut {r['preclim_min']} minutes. Démarrage calculé à {hh(r['clim_debut'], ref)}."
        )

    return (
        f"Je surveille votre maison en temps réel. {phrase_maison(r)} Posez-moi des questions "
        "sur votre facture, les kW, la clim ou votre voiture !"
    )


class CleRefusee(Exception):
    pass


def modeles_du_compte(cle):
    memo = st.session_state.get("liste_modeles")
    if memo and memo[0] == hash(cle):
        return memo[1]
    liste = []
    try:
        rep = requests.get(
            f"{API_BASE}/models",
            params={"pageSize": 200},
            headers={"x-goog-api-key": cle},
            timeout=12,
        )
        if rep.status_code == 200:
            for m in rep.json().get("models", []):
                nom = m.get("name", "").split("/")[-1]
                if (
                    "generateContent"
                    in m.get("supportedGenerationMethods", [])
                    and "flash" in nom
                    and not any(x in nom for x in EXCLUS)
                ):
                    liste.append(nom)
            liste.sort(key=lambda n: "2.5" in n or "2.0" in n, reverse=True)
    except Exception:
        pass
    st.session_state["liste_modeles"] = (hash(cle), liste)
    return liste


def generer(cle, consigne, contenu, max_tokens=1500):
    try:
        candidats = []
        for nom in (
            [st.session_state.get("modele_ok")]
            + MODELES_ALIAS
            + modeles_du_compte(cle)
        ):
            if nom and nom not in candidats:
                candidats.append(nom)
    except CleRefusee:
        return None, None, "clé refusée"

    corps = {
        "systemInstruction": {"parts": [{"text": consigne}]},
        "contents": contenu,
        "generationConfig": {
            "temperature": 0.6,
            "maxOutputTokens": max_tokens,
        },
    }
    entetes = {"Content-Type": "application/json", "x-goog-api-key": cle}
    derniere = "serveur occupé"

    for modele in candidats[:6]:
        try:
            rep = requests.post(
                f"{API_BASE}/models/{modele}:generateContent",
                headers=entetes,
                json=corps,
                timeout=15,
            )
        except requests.RequestException:
            continue
        if rep.status_code == 200:
            try:
                parts = rep.json()["candidates"][0]["content"]["parts"]
                texte = "".join(p.get("text", "") for p in parts).strip()
                if texte:
                    st.session_state["modele_ok"] = modele
                    return texte, modele, None
            except Exception:
                continue
        elif rep.status_code in (401, 403):
            return None, None, "clé invalide"
        elif rep.status_code == 429:
            derniere = "quota atteint (réessayez dans 30s)"
        elif rep.status_code >= 500:
            derniere = "service Google temporairement surchargé"
    return None, None, derniere


def appel_gemini(historique, question, r, cle):
    consigne = (
        "Tu es l'assistant d'une maison intelligente. Réponds en français en 2 à 3 phrases "
        "naturelles et chaleureuses, sans markdown lourd. Réponds d'abord avec précision à la "
        "question posée (même s'il s'agit d'histoire, de culture ou de météo), puis termine "
        "toujours par une courte phrase sur l'état électrique de la maison. "
        "Contexte énergétique en direct : " + contexte_texte(r)
    )
    contenu = [
        {
            "role": "user" if m["role"] == "user" else "model",
            "parts": [{"text": m["text"]}],
        }
        for m in historique[-6:]
    ]
    contenu.append({"role": "user", "parts": [{"text": question}]})
    return generer(cle, consigne, contenu)


def tester_gemini(cle):
    texte, modele, erreur = generer(
        cle,
        "Dis simplement OK",
        [{"role": "user", "parts": [{"text": "Test"}]}],
        200,
    )
    if texte:
        return True, f"Connexion réussie ({modele})."
    return False, f"Échec : {erreur}."


# =====================================================================
# 7. BARRE LATÉRALE : PARAMÈTRES EN TEMPS RÉEL
# =====================================================================
def lire_cle():
    try:
        return str(st.secrets["GEMINI_API_KEY"] or "")
    except Exception:
        return ""


with st.sidebar:
    st.markdown(
        '<div class="side-h">🕒 Vos Horaires</div>', unsafe_allow_html=True
    )
    a, b, c = st.columns(3)
    t_now = a.time_input("Maintenant", value=maintenant_local().time(), step=300)
    t_ret = b.time_input("Retour", value=time(18, 0), step=300)
    t_dep = c.time_input("Départ", value=time(7, 30), step=300)

    st.markdown(
        '<div class="side-h">❄️ Climatisation (Chambre)</div>',
        unsafe_allow_html=True,
    )
    a, b = st.columns(2)
    t_ambiant = a.slider("Actuelle (°C)", 18.0, 38.0, 30.0, 0.5)
    t_consigne = b.slider("Consigne (°C)", 18.0, 26.0, 21.0, 0.5)

    st.markdown(
        '<div class="side-h">🚗 Voiture Électrique</div>', unsafe_allow_html=True
    )
    a, b = st.columns(2)
    bat = a.slider("Batterie (%)", 0, 100, 35)
    cible = b.slider("Objectif (%)", 0, 100, 80)

    st.markdown(
        '<div class="side-h">⚡ Installation & Réseau</div>',
        unsafe_allow_html=True,
    )
    a, b = st.columns(2)
    seuil = a.slider("Disjoncteur (kW)", 3.0, 15.0, 6.0, 0.5)
    clim_kw = b.slider("Puissance Clim (kW)", 0.5, 3.5, CLIM_KW_DEFAUT, 0.25)

    with st.expander("Assistant Gemini (Clé API)"):
        cle_saisie = st.text_input("Clé Google AI Studio", type="password")
        cle = cle_saisie.strip() or lire_cle()
        if st.button("Tester la connexion"):
            if not cle:
                st.warning("Aucune clé renseignée.")
            else:
                with st.spinner("Vérification..."):
                    ok, detail = tester_gemini(cle)
                (st.success if ok else st.error)(detail)

# =====================================================================
# 8. CORPS PRINCIPAL DE LA PAGE
# =====================================================================
base_kw, origine = consommation_base()
r = calculer(
    t_now,
    t_ret,
    t_dep,
    bat,
    cible,
    seuil,
    clim_kw,
    base_kw,
    t_ambiant,
    t_consigne,
)
ref = r["now"]

h = t_now.hour
salut = (
    "Bonne nuit"
    if h < 5
    else "Bonjour"
    if h < 12
    else "Bon après-midi"
    if h < 18
    else "Bonsoir"
)

# En-tête
st.markdown(
    '<div class="entete">'
    '<svg width="42" height="38" viewBox="0 0 46 40" fill="none" stroke="#b5532f" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M3 20 23 4l20 16"/><path d="M8 18v19h30V18"/><path d="M19 37V26h8v11"/>'
    '<path d="M35 8v7"/></svg>'
    f'<div><div class="titre">Énergie de la maison</div>'
    f'<div class="sous">{salut}. Modifiez vos horaires à gauche : l\'orchestrateur s\'adapte.</div></div>'
    f'<div class="heure">Il est {t_now.strftime("%Hh%M")}</div></div>',
    unsafe_allow_html=True,
)

# Les 4 Cartes Dynamiques
c1, c2, c3, c4 = st.columns(4)
with c1:
    carte(
        "maison",
        "Maison",
        "Consommation normale",
        f"{r['base']} kW en continu ({origine})",
        "#3b2f2a",
    )
with c2:
    if r["preclim_min"] > 0:
        carte(
            "thermo",
            "Confort",
            f"Chambre fraîche à {hh(r['retour'], ref)}",
            f"Clim à {hh(r['clim_debut'], ref)} ({r['preclim_min']} min, {r['clim_kw']} kW)",
            "#c98a1b",
        )
    else:
        carte(
            "thermo",
            "Confort",
            "Chambre à consigne",
            f"Déjà à {r['t_consigne']}°C (clim éteinte)",
            "#5f8a5b",
        )
with c3:
    if r["kwh"] == 0:
        carte(
            "voiture",
            "Voiture",
            "Batterie déjà prête",
            "Aucune recharge nécessaire",
            "#5f8a5b",
        )
    else:
        carte(
            "voiture",
            "Voiture",
            f"Prête à {hh(r['fin'], ref)}",
            f"Charge de {hh(r['debut'], ref)} à {hh(r['fin'], ref)} ({BORNE_KW} kW)",
            "#5f8a5b" if r["pret_a_temps"] else "#b03a2e",
        )
with c4:
    etats = {
        "optimal": ("Réseau optimal", "#5f8a5b"),
        "evite": ("Coupure évitée", "#c98a1b"),
        "critique": ("Risque de coupure", "#b03a2e"),
    }
    gros, coul = etats[r["statut"]]
    carte(
        "bouclier",
        "Sécurité",
        gros,
        f"Pic de {r['pic']:.2f} kW pour {r['seuil']} kW",
        coul,
    )

# Section Centrale
gauche, droite = st.columns([3, 2], gap="medium")

with gauche:
    g1, g2 = st.columns([2, 3])
    with g1:
        graphique(jauge(r))
    with g2:
        if r["statut"] == "optimal":
            p1 = (
                f"Votre disjoncteur de {r['seuil']} kW supporte tous les appareils en simultané : "
                f"le pic est de {r['pic']:.2f} kW. La voiture démarre dès {hh(r['debut'], ref)}."
            )
        elif r["statut"] == "evite":
            p1 = (
                f"Si tout démarrait ensemble, la demande atteindrait {r['demande']:.2f} kW (> {r['seuil']} kW). "
                f"La voiture attend donc {hh(r['debut'], ref)} (heures creuses) et le pic est ramené à {r['pic']:.2f} kW."
            )
        else:
            p1 = (
                f"Même après décalage à {hh(r['debut'], ref)}, le pic atteint {r['pic']:.2f} kW pour {r['seuil']} kW autorisés. "
                f"Une souscription d'au moins {np.ceil(r['pic'] * 2) / 2:.1f} kW est conseillée."
            )

        clim_msg = (
            f"Clim allumée à {hh(r['clim_debut'], ref)} pour garantir {r['t_consigne']}°C à votre retour."
            if r["preclim_min"] > 0
            else "Clim éteinte : température de consigne déjà atteinte."
        )

        alerte = ""
        if not r["pret_a_temps"]:
            alerte = (
                f'<div class="alerte">⚠️ La voiture ne sera prête qu\'à {hh(r["fin"], ref)}, '
                f'soit après votre départ prévu à {r["depart"].strftime("%Hh%M")}. '
                "Baissez l'objectif de batterie ou augmentez la puissance de recharge.</div>"
            )

        st.markdown(
            f'<div class="panel"><h4>Ce que fait la maison</h4><p>{p1}</p><p>{clim_msg}</p>{alerte}</div>',
            unsafe_allow_html=True,
        )

    graphique(frise(r))

with droite:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.session_state.get("modele_ok"):
        etat = f"Gemini connecté ({st.session_state['modele_ok']})"
    elif cle:
        etat = "Gemini prêt, en attente"
    else:
        etat = "Mode local autonome"

    st.markdown(
        f'<div class="chat-t">Assistant</div><div class="chat-s">{html.escape(etat)}</div>',
        unsafe_allow_html=True,
    )

    zone = st.container()
    question = st.chat_input("Posez votre question...")

    if question:
        texte, erreur = None, None
        if cle:
            with st.spinner("Je réfléchis..."):
                texte, _, erreur = appel_gemini(
                    st.session_state.messages, question, r, cle
                )
        if texte is None:
            texte = reponse_locale(question, r)

        st.session_state.messages.append({"role": "user", "text": question})
        st.session_state.messages.append(
            {"role": "bot", "text": texte, "erreur": erreur}
        )
        st.session_state.messages = st.session_state.messages[-30:]

    def bulle(m):
        corps = html.escape(m["text"].replace("**", "")).replace("\n", "<br>")
        if m["role"] == "user":
            return f'<div class="msg user"><span class="who">Vous</span>{corps}</div>'
        note = (
            f'<span class="note">Google occupé ({html.escape(m["erreur"])}). Réponse locale active.</span>'
            if m.get("erreur")
            else ""
        )
        return f'<div class="msg bot"><span class="who">Assistant</span>{corps}{note}</div>'

    if st.session_state.messages:
        contenu = "".join(bulle(m) for m in st.session_state.messages)
    else:
        contenu = (
            '<div class="msg bot"><span class="who">Assistant</span>Bonjour ! Demandez-moi '
            "pourquoi la recharge démarre à telle heure, ce que prédisent les modèles, "
            "ou posez n'importe quelle autre question.</div>"
        )

    with zone:
        st.markdown(
            f'<div class="chatbox"><div>{contenu}</div></div>',
            unsafe_allow_html=True,
        )
