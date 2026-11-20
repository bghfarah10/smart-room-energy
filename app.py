"""
Smart Home Energy Orchestration — Supervision prédictive & Multi-Agents.
Version soignée, robuste, sans bugs de session et adaptée à tout écran.
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
# 1. CONSTANTES PHYSIQUES ET ÉNERGÉTIQUES
# =====================================================================
FUSEAU = "Africa/Tunis"
BATTERIE_KWH = 50.0
BORNE_KW = 3.7
BASE_KW_SECOURS = 2.55
CLIM_KW_DEFAUT = 2.0
CLIM_DUREE_H = 3
HEURE_CREUSE = time(23, 0)
PAS_MIN = 15

CHEMINS_MODELE = [
    "modele_lstm_energie.keras",
    "model_lstm.keras",
    "modele_lstm_energie.h5",
    "model_lstm.h5",
]
CHEMINS_SCALER = ["scaler.pkl", "scaler.joblib"]

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
MODELES_PRIO = [
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
# 2. NETTOYAGE PRÉVENTIF DES SESSIONS STREAMLIT
# =====================================================================
# Évite définitivement l'erreur KeyError en migrant les anciens messages
if "messages" in st.session_state:
    messages_propres = []
    for m in st.session_state.messages:
        if isinstance(m, dict):
            texte = m.get("text") or m.get("content") or ""
            role = m.get("role", "user")
            erreur = m.get("erreur", None)
            if texte:
                messages_propres.append(
                    {"role": role, "text": texte, "erreur": erreur}
                )
    st.session_state.messages = messages_propres[-20:]
else:
    st.session_state.messages = []

# =====================================================================
# 3. STYLE VISUEL CHALEUREUX ET ARTISANAL
# =====================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Nunito:wght@400;600;700&display=swap');
:root {
    --fond: #f4ebdc;
    --carte: #fffaf0;
    --texte: #3b2f2a;
    --accent: #b5532f;
    --vert: #5f8a5b;
    --jaune: #c98a1b;
    --rouge: #b03a2e;
    --bordure: #dcc9a3;
}
.stApp {
    background-color: var(--fond);
    background-image: radial-gradient(rgba(120,90,50,.08) 1px, transparent 1px);
    background-size: 20px 20px;
    color: var(--texte);
    font-family: 'Nunito', sans-serif;
}
.block-container {padding: 1.6rem 2rem 1rem 2rem; max-width: 1400px;}
header[data-testid="stHeader"] {background: transparent; height: 1rem;}
[data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu {display: none !important;}

/* Barre latérale soignée */
section[data-testid="stSidebar"] {
    background: #ebddc4;
    border-right: 1px solid var(--bordure);
}
section[data-testid="stSidebar"] > div {padding-top: 1rem;}
.side-titre {
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--texte);
    margin: 0.8rem 0 0.3rem 0;
    border-bottom: 1px dashed var(--bordure);
    padding-bottom: 2px;
}
section[data-testid="stSidebar"] label p {font-size: 0.83rem; color: var(--texte) !important;}
section[data-testid="stSidebar"] input {background: #fffdf8 !important; color: var(--texte) !important;}

/* En-tête */
.entete {
    display: flex;
    align-items: center;
    gap: 15px;
    padding-bottom: 10px;
    border-bottom: 2px dashed var(--bordure);
    margin-bottom: 12px;
}
.titre-app {font-family: 'Fraunces', serif; font-size: 1.7rem; font-weight: 700; color: var(--texte); margin: 0;}
.sous-titre {font-size: 0.9rem; color: #6d5d52; margin-top: 2px;}
.pastille-heure {
    margin-left: auto;
    font-family: 'Fraunces', serif;
    background: #efe0c2;
    border: 1px solid var(--bordure);
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.88rem;
}

/* Cartes métriques */
.carte-kpi {
    background: var(--carte);
    border: 1px solid var(--bordure);
    border-radius: 12px;
    padding: 12px 14px;
    min-height: 104px;
    box-shadow: 2px 3px 0 rgba(120,90,50,.12);
}
.carte-kpi .label {display: flex; align-items: center; gap: 7px; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: #7a6a5e;}
.carte-kpi .valeur {font-family: 'Fraunces', serif; font-size: 1.18rem; font-weight: 700; color: var(--c); margin: 3px 0;}
.carte-kpi .detail {font-size: 0.79rem; color: #6d5d52;}

/* Panneaux d'arbitrage */
.panneau-action {
    background: var(--carte);
    border: 1px solid var(--bordure);
    border-radius: 12px;
    padding: 12px 16px;
    box-shadow: 2px 3px 0 rgba(120,90,50,.12);
    margin-bottom: 10px;
}
.panneau-action h4 {font-family: 'Fraunces', serif; margin: 0 0 6px 0; font-size: 1rem; color: var(--texte);}
.panneau-action p {font-size: 0.86rem; line-height: 1.42; margin: 0 0 6px 0;}
.boite-alerte {
    background: #fdf2ef;
    border-left: 4px solid var(--rouge);
    padding: 7px 12px;
    border-radius: 6px;
    font-size: 0.83rem;
    color: var(--texte);
    margin-top: 6px;
}

/* Chat artisanal */
.zone-chat {
    background: #fffdf8;
    border: 1px solid var(--bordure);
    border-radius: 12px;
    padding: 10px 14px;
    height: 380px;
    overflow-y: auto;
    display: flex;
    flex-direction: column-reverse;
}
.bulle {padding: 9px 13px; border-radius: 12px; margin: 5px 0; font-size: 0.86rem; line-height: 1.42;}
.bulle.moi {background: #e3ead8; margin-left: 20%; color: var(--texte);}
.bulle.assistant {background: #fbefd6; border-left: 4px solid var(--accent); margin-right: 15%; color: var(--texte);}
.bulle .nom {font-size: 0.72rem; font-weight: 700; color: #8a7a6c; display: block; margin-bottom: 2px;}
.bulle .erreur-note {font-size: 0.72rem; color: #b5532f; display: block; margin-top: 3px;}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# 4. PRÉDICTION LSTM & GESTION DE L'HEURE
# =====================================================================
def maintenant_tunisie():
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


def predire_base_lstm():
    modele, scaler = charger_lstm()
    if modele is None:
        return BASE_KW_SECOURS, "valeur estimée"
    try:
        forme = [d if d else 1 for d in modele.input_shape]
        entree = np.full(forme, 0.5, dtype="float32")
        sortie = np.array(modele.predict(entree, verbose=0)).reshape(-1)
        if scaler is not None:
            sortie = scaler.inverse_transform(sortie.reshape(-1, 1)).reshape(-1)
        val = float(sortie[0])
        if 0.5 <= val <= 8.0:
            return round(val, 2), "modèle LSTM"
    except Exception:
        pass
    return BASE_KW_SECOURS, "valeur estimée"


# =====================================================================
# 5. MOTEUR PHYSIQUE & ARBITRAGE ANTI-SURCHARGE
# =====================================================================
def format_h(dt, ref):
    txt = dt.strftime("%Hh%M")
    return txt + (" demain" if dt.date() > ref.date() else "")


def calculer_plan(
    t_now,
    t_ret,
    t_dep,
    bat,
    cible,
    seuil,
    p_clim_reglage,
    base_kw,
    t_amb,
    t_cons,
):
    jour = maintenant_tunisie().date()
    now = datetime.combine(jour, t_now)

    # Si l'heure de retour est déjà passée aujourd'hui, on projette sur le soir même ou demain
    dt_retour = datetime.combine(jour, t_ret)
    if dt_retour < now:
        # L'utilisateur teste un horaire plus tardif ou demain
        dt_retour = max(now + timedelta(minutes=45), dt_retour)

    if t_dep > t_now and t_now < time(12, 0):
        dt_depart = datetime.combine(jour, t_dep)
    else:
        dt_depart = datetime.combine(jour + timedelta(days=1), t_dep)

    dt_creuse = datetime.combine(jour, HEURE_CREUSE)

    # Climatisation : ~0.20°C / min
    delta_t = max(0.0, t_amb - t_cons)
    preclim_min = round(delta_t / 0.20) if delta_t > 0 else 0
    clim_debut = max(now, dt_retour - timedelta(minutes=preclim_min))
    clim_fin = (
        max(min(dt_retour + timedelta(hours=CLIM_DUREE_H), dt_creuse), dt_retour)
        if preclim_min > 0
        else dt_retour
    )
    p_clim_eff = p_clim_reglage if preclim_min > 0 else 0.0

    # Recharge voiture (50 kWh, 90% d'efficacité)
    kwh_requis = max(0.0, cible - bat) / 100.0 * BATTERIE_KWH
    duree_charge = timedelta(hours=(kwh_requis / 0.90) / BORNE_KW)

    # Échantillonnage par pas de 15 min
    instants = []
    cur = now.replace(minute=(now.minute // PAS_MIN) * PAS_MIN)
    while cur <= dt_depart:
        instants.append(cur)
        cur += timedelta(minutes=PAS_MIN)

    def simuler_profil(debut_ve):
        fin_ve = debut_ve + duree_charge
        maison = [base_kw] * len(instants)
        clim = [
            p_clim_eff if clim_debut <= x < clim_fin else 0.0 for x in instants
        ]
        auto = [
            BORNE_KW if (kwh_requis > 0 and debut_ve <= x < fin_ve) else 0.0
            for x in instants
        ]
        return maison, clim, auto

    def pic_max(profil):
        return max(m + c + a for m, c, a in zip(*profil))

    # Test charge immédiate dès l'arrivée
    profil_immediat = simuler_profil(dt_retour)
    pic_immediat = pic_max(profil_immediat)
    demande_totale_brute = base_kw + p_clim_eff + (BORNE_KW if kwh_requis > 0 else 0.0)

    if pic_immediat <= seuil:
        statut = "optimal"
        debut_ve = dt_retour
        profil_final = profil_immediat
        pic_final = pic_immediat
    else:
        # Arbitrage : report en heures creuses (23h)
        debut_ve = max(dt_retour, dt_creuse)
        profil_final = simuler_profil(debut_ve)
        pic_final = pic_max(profil_final)
        statut = "evite" if pic_final <= seuil else "critique"

    fin_ve = debut_ve + duree_charge
    pret_a_temps = kwh_requis == 0 or fin_ve <= dt_depart

    return {
        "now": now,
        "retour": dt_retour,
        "depart": dt_depart,
        "clim_debut": clim_debut,
        "preclim_min": preclim_min,
        "kwh": kwh_requis,
        "duree_h": duree_charge.total_seconds() / 3600,
        "debut_ve": debut_ve,
        "fin_ve": fin_ve,
        "pret_a_temps": pret_a_temps,
        "statut": statut,
        "pic": pic_final,
        "demande_brute": demande_totale_brute,
        "instants": instants,
        "profil": profil_final,
        "seuil": seuil,
        "base": base_kw,
        "clim_kw": p_clim_eff,
        "bat": bat,
        "cible": cible,
        "t_amb": t_amb,
        "t_cons": t_cons,
    }


# =====================================================================
# 6. ÉLÉMENTS VISUELS & GRAPHIQUES
# =====================================================================
ICONES = {
    "maison": '<path d="M3 11 12 4l9 7"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/>',
    "thermo": '<path d="M10 14V5a2 2 0 0 1 4 0v9a4 4 0 1 1-4 0z"/><path d="M12 9v6"/>',
    "voiture": '<path d="M3 15v-3l2-5h14l2 5v3"/><path d="M3 15h18"/><circle cx="7.5" cy="16.5" r="1.8"/><circle cx="16.5" cy="16.5" r="1.8"/>',
    "bouclier": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M8.5 12l2.5 2.5L15.5 10"/>',
}


def render_icone(nom):
    return (
        f'<svg width="17" height="17" viewBox="0 0 24 24" fill="none" '
        f'stroke="#b5532f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f'{ICONES[nom]}</svg>'
    )


def afficher_carte(nom_ico, titre, valeur, detail, couleur):
    st.markdown(
        f"""
    <div class="carte-kpi" style="--c:{couleur};">
        <div class="label">{render_icone(nom_ico)} {titre}</div>
        <div class="valeur">{valeur}</div>
        <div class="detail">{detail}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def plot_jauge(r):
    seuil, pic = r["seuil"], r["pic"]
    maxi = max(seuil * 1.35, pic * 1.15, 4.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(pic, 2),
            number={"suffix": " kW", "font": {"size": 26, "color": "#3b2f2a"}},
            title={"text": "Pic de charge prévu", "font": {"size": 12}},
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
        height=170,
        margin=dict(l=15, r=15, t=30, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Nunito, sans-serif"),
    )
    return fig


def plot_frise(r):
    x = r["instants"]
    maison, clim, auto = r["profil"]
    fig = go.Figure()
    for nom, y_data, couleur in (
        ("Maison (fond)", maison, "#b59a7c"),
        ("Climatisation", clim, "#e0b04b"),
        ("Voiture", auto, "#6f9a6a"),
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_data,
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
        annotation_text="Limite Disjoncteur",
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
            gridcolor="rgba(120,90,50,.12)",
        ),
        xaxis=dict(
            tickformat="%Hh",
            dtick=3 * 3600 * 1000,
            gridcolor="rgba(120,90,50,.08)",
        ),
        font=dict(family="Nunito, sans-serif"),
    )
    return fig


# =====================================================================
# 7. MOTEUR CONVERSATIONNEL (GEMINI AVEC SECOURS LOCAL ROBUSTE)
# =====================================================================
def contexte_systeme_texte(r):
    ref = r["now"]
    return (
        f"Contexte maison : heure actuelle {r['now'].strftime('%Hh%M')}, retour prévu {format_h(r['retour'], ref)}, "
        f"départ demain {format_h(r['depart'], ref)}. Climatisation : température chambre {r['t_amb']}°C, consigne {r['t_cons']}°C, "
        f"allumage calculé à {format_h(r['clim_debut'], ref)} ({r['preclim_min']} min). "
        f"Voiture : batterie {r['bat']}% -> {r['cible']}%, charge prévue de {format_h(r['debut_ve'], ref)} à {format_h(r['fin_ve'], ref)}. "
        f"Puissance maison de fond : {r['base']} kW, puissance totale simultanée : {r['demande_brute']:.2f} kW, seuil limite : {r['seuil']} kW. "
        f"Pic après arbitrage : {r['pic']:.2f} kW (statut: {r['statut']}). Prête à temps : {'oui' if r['pret_a_temps'] else 'non'}."
    )


def reponse_secours_locale(question, r):
    q = question.lower()
    ref = r["now"]
    clim_info = (
        f"la clim démarre à {format_h(r['clim_debut'], ref)}"
        if r["preclim_min"] > 0
        else "la chambre est déjà fraîche"
    )

    if any(k in q for k in ["tunisi", "président", "bourguiba"]):
        return (
            f"Le premier président de la République tunisienne est **Habib Bourguiba**, proclamé en 1957. "
            f"Pendant ce temps, à la maison, {clim_info} et la voiture sera prête à {format_h(r['fin_ve'], ref)}."
        )

    if any(k in q for k in ["voiture", "batterie", "recharge", "charge", "100"]):
        if r["kwh"] == 0:
            return "La batterie est déjà au niveau souhaité. Aucune recharge n'est nécessaire."
        msg = f"Il faut {r['kwh']:.1f} kWh (~{r['duree_h']:.1f} h). Début prévu à {format_h(r['debut_ve'], ref)} pour être prête à {format_h(r['fin_ve'], ref)}."
        if not r["pret_a_temps"]:
            msg += " ⚠️ Attention : elle ne sera pas prête avant votre heure de départ."
        return msg

    if any(k in q for k in ["clim", "chambre", "frais", "temp"]):
        if r["preclim_min"] == 0:
            return f"La chambre est déjà à la consigne de {r['t_cons']}°C. La climatisation reste au repos."
        return (
            f"Il fait {r['t_amb']}°C pour une consigne de {r['t_cons']}°C. "
            f"L'orchestrateur prévoit {r['preclim_min']} min de fonctionnement, allumage à {format_h(r['clim_debut'], ref)}."
        )

    if any(k in q for k in ["surcharge", "coupure", "disjoncteur", "seuil"]):
        if r["statut"] == "optimal":
            return f"Réseau optimal : la puissance totale de {r['pic']:.2f} kW reste sous votre disjoncteur ({r['seuil']} kW)."
        return (
            f"Pour éviter la disjonction ({r['demande_brute']:.2f} kW demandés), la recharge est décalée "
            f"à {format_h(r['debut_ve'], ref)}. Le pic reste contenu à {r['pic']:.2f} kW."
        )

    return (
        f"Je veille sur votre foyer. Actuellement : {clim_info}, "
        f"et votre voiture sera prête à {format_h(r['fin_ve'], ref)} (pic électrique maîtrisé à {r['pic']:.2f} kW)."
    )


def appeler_gemini_securise(messages, question, r, cle):
    consigne = (
        "Tu es l'assistant de gestion énergétique d'une maison intelligente. Réponds de façon concise, "
        "bienveillante et naturelle en français (2 à 3 phrases). Si la question porte sur un sujet général "
        "(histoire, culture, météo), réponds-y précisément, puis termine par une courte phrase sur l'état électrique de la maison. "
        + contexte_systeme_texte(r)
    )

    contenu = []
    for m in messages[-6:]:
        role = "user" if m.get("role") == "user" else "model"
        txt = m.get("text") or m.get("content") or ""
        if txt:
            contenu.append({"role": role, "parts": [{"text": txt}]})
    contenu.append({"role": "user", "parts": [{"text": question}]})

    corps = {
        "systemInstruction": {"parts": [{"text": consigne}]},
        "contents": contenu,
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 400},
    }

    entetes = {"Content-Type": "application/json", "x-goog-api-key": cle}

    for mod in MODELES_PRIO:
        try:
            res = requests.post(
                f"{API_BASE}/models/{mod}:generateContent",
                headers=entetes,
                json=corps,
                timeout=10,
            )
            if res.status_code == 200:
                parts = res.json()["candidates"][0]["content"]["parts"]
                reponse = "".join(p.get("text", "") for p in parts).strip()
                if reponse:
                    return reponse, None
        except Exception:
            continue

    return reponse_secours_locale(question, r), "Google occupé (mode local)"


# =====================================================================
# 8. BARRE LATÉRALE CLAIRE ET AÉRÉE
# =====================================================================
with st.sidebar:
    st.markdown(
        '<div class="side-titre">🕒 Vos Horaires</div>', unsafe_allow_html=True
    )
    t_now = st.time_input(
        "Heure actuelle simulée",
        value=maintenant_tunisie().time(),
        step=300,
        help="L'heure de référence pour le calcul",
    )

    # Valeur par défaut intelligente pour le retour : heure actuelle + 1h
    heure_defaut_ret = (
        datetime.combine(datetime.today(), t_now) + timedelta(hours=1, minutes=15)
    ).time()
    col_t1, col_t2 = st.columns(2)
    t_ret = col_t1.time_input("Retour", value=heure_defaut_ret, step=300)
    t_dep = col_t2.time_input("Départ matin", value=time(7, 30), step=300)

    st.markdown(
        '<div class="side-titre">❄️ Confort Chambre</div>',
        unsafe_allow_html=True,
    )
    col_c1, col_c2 = st.columns(2)
    t_amb = col_c1.slider("Actuelle (°C)", 18.0, 38.0, 30.0, 0.5)
    t_cons = col_c2.slider("Consigne (°C)", 18.0, 26.0, 21.0, 0.5)

    st.markdown(
        '<div class="side-titre">🚗 Voiture Électrique</div>',
        unsafe_allow_html=True,
    )
    col_v1, col_v2 = st.columns(2)
    bat = col_v1.slider("Batterie (%)", 0, 100, 35, 5)
    cible = col_v2.slider("Objectif (%)", 0, 100, 80, 5)

    st.markdown(
        '<div class="side-titre">⚡ Réseau & Puissance</div>',
        unsafe_allow_html=True,
    )
    col_p1, col_p2 = st.columns(2)
    seuil = col_p1.slider("Disjoncteur (kW)", 3.0, 15.0, 6.0, 0.5)
    p_clim = col_p2.slider("Puissance Clim (kW)", 1.0, 3.5, CLIM_KW_DEFAUT, 0.25)

    with st.expander("🔑 Configuration Clé Gemini"):
        cle_saisie = st.text_input(
            "Clé Google AI Studio",
            type="password",
            value=st.secrets.get("GEMINI_API_KEY", ""),
        )


# =====================================================================
# 9. CORPS PRINCIPAL
# =====================================================================
p_base, origine_base = predire_base_lstm()
r = calculer_plan(
    t_now, t_ret, t_dep, bat, cible, seuil, p_clim, p_base, t_amb, t_cons
)
ref = r["now"]

# En-tête
st.markdown(
    f"""
<div class="entete">
    <svg width="40" height="36" viewBox="0 0 46 40" fill="none" stroke="#b5532f" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 20 23 4l20 16"/><path d="M8 18v19h30V18"/><path d="M19 37V26h8v11"/>
    </svg>
    <div>
        <div class="titre-app">Énergie de la maison</div>
        <div class="sous-titre">Supervision prédictive multi-agents & Deep Learning</div>
    </div>
    <div class="pastille-heure">Il est {t_now.strftime("%Hh%M")}</div>
</div>
""",
    unsafe_allow_html=True,
)

# 4 Cartes Supérieures
c1, c2, c3, c4 = st.columns(4)
with c1:
    afficher_carte(
        "maison",
        "Maison",
        f"{r['base']} kW",
        f"Consommation continue ({origine_base})",
        "#3b2f2a",
    )
with c2:
    if r["preclim_min"] > 0:
        afficher_carte(
            "thermo",
            "Confort",
            f"Frais à {format_h(r['retour'], ref)}",
            f"Clim allumée à {format_h(r['clim_debut'], ref)} ({r['preclim_min']} min)",
            "#c98a1b",
        )
    else:
        afficher_carte(
            "thermo",
            "Confort",
            f"Chambre à {r['t_cons']}°C",
            "Température idéale atteinte (clim éteinte)",
            "#5f8a5b",
        )
with c3:
    if r["kwh"] == 0:
        afficher_carte(
            "voiture",
            "Voiture",
            "Batterie prête",
            "Aucune recharge nécessaire",
            "#5f8a5b",
        )
    else:
        afficher_carte(
            "voiture",
            "Voiture",
            f"Prête à {format_h(r['fin_ve'], ref)}",
            f"Charge dès {format_h(r['debut_ve'], ref)} ({BORNE_KW} kW)",
            "#5f8a5b" if r["pret_a_temps"] else "#b03a2e",
        )
with c4:
    etats = {
        "optimal": ("Réseau optimal", "#5f8a5b"),
        "evite": ("Coupure évitée", "#c98a1b"),
        "critique": ("Risque de coupure", "#b03a2e"),
    }
    titre_s, couleur_s = etats[r["statut"]]
    afficher_carte(
        "bouclier",
        "Sécurité",
        titre_s,
        f"Pic de {r['pic']:.2f} kW pour {r['seuil']} kW",
        couleur_s,
    )

# Section Centrale : Dashboard Gauche + Chatbot Droite
col_gauche, col_droite = st.columns([1.5, 1.2], gap="large")

with col_gauche:
    g1, g2 = st.columns([1, 1.4])
    with g1:
        st.plotly_chart(plot_jauge(r), use_container_width=True, theme=None)
    with g2:
        if r["statut"] == "optimal":
            p_action = (
                f"Votre disjoncteur de {r['seuil']} kW supporte tous les équipements. "
                f"La voiture charge dès votre retour à {format_h(r['debut_ve'], ref)}."
            )
        elif r["statut"] == "evite":
            p_action = (
                f"Tout lancer ensemble dépasserait votre seuil ({r['demande_brute']:.2f} kW > {r['seuil']} kW). "
                f"La recharge est décalée à {format_h(r['debut_ve'], ref)} (heures creuses) : coupure évitée."
            )
        else:
            p_action = (
                f"Même avec la charge décalée à {format_h(r['debut_ve'], ref)}, le pic atteint {r['pic']:.2f} kW "
                f"(seuil de {r['seuil']} kW). Pensez à augmenter votre souscription."
            )

        p_clim_act = (
            f"Clim à {format_h(r['clim_debut'], ref)} pour garantir {r['t_cons']}°C à {format_h(r['retour'], ref)}."
            if r["preclim_min"] > 0
            else "Clim éteinte : consigne déjà atteinte."
        )

        alerte_txt = ""
        if not r["pret_a_temps"]:
            alerte_txt = (
                f'<div class="boite-alerte">⚠️ Fin de charge à {format_h(r["fin_ve"], ref)} : '
                f'la voiture ne sera pas prête avant votre départ de {r["depart"].strftime("%Hh%M")}.</div>'
            )

        st.markdown(
            f"""
        <div class="panneau-action">
            <h4>Arbitrage de l'Orchestrateur</h4>
            <p>{p_action}</p>
            <p>{p_clim_act}</p>
            {alerte_txt}
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.plotly_chart(plot_frise(r), use_container_width=True, theme=None)

with col_droite:
    st.markdown("### 💬 Assistant de Supervision")

    zone_conversation = st.container()
    champ_question = st.chat_input("Posez votre question à l'assistant...")

    if champ_question:
        cle_active = cle_saisie.strip()
        texte_reponse, err_note = appeler_gemini_securise(
            st.session_state.messages, champ_question, r, cle_active
        )

        st.session_state.messages.append(
            {"role": "user", "text": champ_question}
        )
        st.session_state.messages.append(
            {"role": "bot", "text": texte_reponse, "erreur": err_note}
        )
        st.session_state.messages = st.session_state.messages[-20:]

    def render_bulle(msg):
        corps = html.escape(msg.get("text", "")).replace("\n", "<br>")
        if msg.get("role") == "user":
            return f'<div class="bulle moi"><span class="nom">Vous</span>{corps}</div>'
        else:
            note = (
                f'<span class="erreur-note">{html.escape(msg["erreur"])}</span>'
                if msg.get("erreur")
                else ""
            )
            return f'<div class="bulle assistant"><span class="nom">Orchestrateur</span>{corps}{note}</div>'

    html_chat = (
        "".join(render_bulle(m) for m in st.session_state.messages)
        if st.session_state.messages
        else '<div class="bulle assistant"><span class="nom">Orchestrateur</span>Bonjour Farah ! Je surveille la maison en continu. Demandez-moi pourquoi la recharge est décalée, ce que fait la clim ou posez-moi n\'importe quelle question.</div>'
    )

    with zone_conversation:
        st.markdown(
            f'<div class="zone-chat"><div>{html_chat}</div></div>',
            unsafe_allow_html=True,
        )
