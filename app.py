
import html
from datetime import datetime, time, timedelta

import joblib
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    from tensorflow.keras.models import load_model
except Exception:  # l'app démarre même sans TensorFlow
    load_model = None

# =====================================================================
# 1. CONSTANTES MÉTIER
# =====================================================================
BATTERIE_KWH = 50.0          # capacité de la batterie
BORNE_KW = 3.7               # puissance de la borne de recharge
BASE_KW_SECOURS = 2.55       # consommation de base si le LSTM est absent
CLIM_KW_DEFAUT = 2.0         # puissance de la climatisation
PRECLIM_MIN = 45             # la clim démarre 45 min avant le retour
CLIM_DUREE_H = 3             # la clim reste allumée 3 h après le retour
HEURE_CREUSE = time(23, 0)   # début de charge décalée
PAS_MIN = 15                 # pas de la frise (minutes)

CHEMINS_MODELE = ["model_lstm.keras", "model_lstm.h5", "model.h5", "lstm_model.h5"]
CHEMINS_SCALER = ["scaler.pkl", "scaler.joblib"]

MODELES_GEMINI = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
]
API_BASE = "https://generativelanguage.googleapis.com/v1beta"

st.set_page_config(page_title="Énergie de la maison", layout="wide",
                   initial_sidebar_state="expanded")

# =====================================================================
# 2. STYLE
# =====================================================================
st.markdown("""
<style>
.stApp {
    background-color: #ece2cc;
    background-image:
        linear-gradient(rgba(20,33,61,.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(20,33,61,.045) 1px, transparent 1px);
    background-size: 28px 28px;
}
.block-container {padding-top: .6rem; padding-bottom: .4rem; max-width: 1400px;}
footer, #MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
html, body, .stApp, p, label, span, div {color: #14213d;}

section[data-testid="stSidebar"] {background: #e0d3b6; border-right: 2px solid #c9a227;}
section[data-testid="stSidebar"] * {color: #14213d !important;}
section[data-testid="stSidebar"] input {
    background: #ffffff !important; color: #14213d !important;
    -webkit-text-fill-color: #14213d !important;
}
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="select"] > div {background: #ffffff !important;}

.banner {
    background: linear-gradient(100deg, #14213d 0%, #1d3461 100%);
    border-bottom: 4px solid #c9a227; border-radius: 10px;
    padding: 12px 22px; display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 10px;
}
.banner h1 {color: #f6efe0 !important; font-size: 1.45rem; margin: 0; font-weight: 700;}
.banner p {color: #d9cfae !important; margin: 2px 0 0 0; font-size: .86rem;}

.card {
    background: #fffaf0; border: 1px solid #b9a97d; border-radius: 10px;
    padding: 12px 14px 10px 14px; height: 138px;
    box-shadow: 0 3px 8px rgba(20,33,61,.15); border-top: 5px solid var(--accent);
}
.card .row {display: flex; align-items: center; gap: 9px; margin-bottom: 6px;}
.card .ico {
    width: 34px; height: 34px; border-radius: 50%; background: #dfe8f3;
    display: flex; align-items: center; justify-content: center; flex: none;
}
.card .lab {font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;
    color: #5a6478 !important; font-weight: 600;}
.card .big {font-size: 1.18rem; font-weight: 700; line-height: 1.2; color: var(--accent) !important;}
.card .small {font-size: .8rem; color: #4a5468 !important; margin-top: 5px;}

.panel {background: #fffaf0; border: 1px solid #b9a97d; border-radius: 10px;
    padding: 10px 14px; box-shadow: 0 2px 6px rgba(20,33,61,.12);}
.panel p {font-size: .88rem; margin: 0 0 6px 0; line-height: 1.4;}
.panel h4 {margin: 0 0 6px 0; font-size: .95rem; color: #14213d !important;}
.alerte {background: #fbe9e7; border-left: 4px solid #b3261e; padding: 6px 10px;
    border-radius: 4px; font-size: .85rem; margin-top: 6px;}

.chat-title {font-weight: 700; font-size: .98rem; margin-bottom: 2px;}
.chat-state {font-size: .78rem; color: #4a5468 !important; margin-bottom: 4px;}
.msg {padding: 8px 12px; border-radius: 8px; margin: 6px 0; font-size: .9rem; line-height: 1.4;}
.msg .who {font-size: .68rem; text-transform: uppercase; letter-spacing: .07em;
    color: #5a6478 !important; display: block; margin-bottom: 2px; font-weight: 600;}
.msg.user {background: #dbe6f4; margin-left: 22px;}
.msg.bot {background: #fff6e0; border-left: 4px solid #c9a227; margin-right: 22px;}
.msg.err {font-size: .78rem; color: #8a4b00 !important; margin-top: 4px; display: block;}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 3. MODÈLE LSTM (facultatif)
# =====================================================================
@st.cache_resource(show_spinner=False)
def charger_lstm():
    """Retourne (modèle, scaler) ou (None, None) si absent."""
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
    """Consommation de base en kW et origine de la valeur."""
    modele, scaler = charger_lstm()
    if modele is None:
        return BASE_KW_SECOURS, "Valeur de secours"
    try:
        forme = [d if d else 1 for d in modele.input_shape]
        entree = np.full(forme, 0.5, dtype="float32")
        sortie = np.array(modele.predict(entree, verbose=0)).reshape(-1)
        if scaler is not None:
            try:
                sortie = scaler.inverse_transform(sortie.reshape(-1, 1)).reshape(-1)
            except Exception:
                pass
        valeur = float(sortie[0])
        if 0.3 <= valeur <= 8.0:
            return round(valeur, 2), "Prévision LSTM"
    except Exception:
        pass
    return BASE_KW_SECOURS, "Valeur de secours"

# =====================================================================
# 4. MOTEUR DE CALCUL
# =====================================================================
def hh(dt, ref):
    texte = dt.strftime("%Hh%M")
    return texte + (" (demain)" if dt.date() > ref.date() else "")


def calculer(t_now, t_ret, t_dep, bat, cible, seuil, clim_kw, base_kw):
    aujourdhui = datetime.now().date()
    now = datetime.combine(aujourdhui, t_now)
    retour = max(datetime.combine(aujourdhui, t_ret), now)
    depart = datetime.combine(aujourdhui + timedelta(days=1), t_dep)
    creuse = datetime.combine(aujourdhui, HEURE_CREUSE)

    clim_debut = max(now, retour - timedelta(minutes=PRECLIM_MIN))
    clim_fin = max(min(retour + timedelta(hours=CLIM_DUREE_H), creuse), retour)

    kwh = max(0.0, cible - bat) / 100.0 * BATTERIE_KWH
    duree = timedelta(hours=kwh / BORNE_KW)

    pas = timedelta(minutes=PAS_MIN)
    instants = []
    t = now.replace(minute=(now.minute // PAS_MIN) * PAS_MIN, second=0, microsecond=0)
    while t <= depart:
        instants.append(t)
        t += pas

    def profil(debut_charge):
        fin_charge = debut_charge + duree
        maison = [base_kw] * len(instants)
        clim = [clim_kw if clim_debut <= x < clim_fin else 0.0 for x in instants]
        auto = [BORNE_KW if (kwh > 0 and debut_charge <= x < fin_charge) else 0.0
                for x in instants]
        return maison, clim, auto

    def pic(p):
        return max(m + c + a for m, c, a in zip(*p))

    debut_a = retour
    prof_a = profil(debut_a)
    pic_a = pic(prof_a)
    demande_max = base_kw + clim_kw + (BORNE_KW if kwh > 0 else 0.0)

    if pic_a <= seuil:
        statut, debut, prof, pic_final = "optimal", debut_a, prof_a, pic_a
    else:
        debut_b = max(retour, creuse)
        prof_b = profil(debut_b)
        pic_b = pic(prof_b)
        statut = "evite" if pic_b <= seuil else "critique"
        debut, prof, pic_final = debut_b, prof_b, pic_b

    fin = debut + duree
    return {
        "now": now, "retour": retour, "depart": depart,
        "clim_debut": clim_debut, "clim_fin": clim_fin,
        "kwh": kwh, "duree_h": duree.total_seconds() / 3600,
        "debut": debut, "fin": fin, "pret_a_temps": (kwh == 0) or fin <= depart,
        "statut": statut, "pic": pic_final, "demande": demande_max,
        "instants": instants, "profil": prof, "seuil": seuil,
        "base": base_kw, "clim_kw": clim_kw, "bat": bat, "cible": cible,
    }

# =====================================================================
# 5. ÉLÉMENTS VISUELS
# =====================================================================
ICONES = {
    "maison": '<path d="M3 11 12 4l9 7"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/>',
    "thermo": '<path d="M10 14V5a2 2 0 0 1 4 0v9a4 4 0 1 1-4 0z"/><path d="M12 9v6"/>',
    "voiture": '<path d="M3 15v-3l2-5h14l2 5v3"/><path d="M3 15h18"/><circle cx="7.5" cy="16.5" r="1.8"/><circle cx="16.5" cy="16.5" r="1.8"/>',
    "bouclier": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M8.5 12l2.5 2.5L15.5 10"/>',
}


def icone(nom):
    return ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1d3461" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{ICONES[nom]}</svg>')


def carte(ico, titre, gros, petit, couleur):
    st.markdown(
        f'<div class="card" style="--accent:{couleur}"><div class="row">'
        f'<div class="ico">{icone(ico)}</div><div class="lab">{titre}</div></div>'
        f'<div class="big">{gros}</div><div class="small">{petit}</div></div>',
        unsafe_allow_html=True)


def jauge(r):
    seuil, pic = r["seuil"], r["pic"]
    maxi = max(seuil * 1.35, pic * 1.15, 4.0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(pic, 2),
        number={"suffix": " kW", "font": {"size": 30, "color": "#14213d"}},
        title={"text": "Pic de puissance", "font": {"size": 13, "color": "#14213d"}},
        gauge={
            "axis": {"range": [0, maxi], "tickcolor": "#14213d"},
            "bar": {"color": "#14213d", "thickness": 0.25},
            "steps": [
                {"range": [0, seuil * 0.8], "color": "#b7d7b0"},
                {"range": [seuil * 0.8, seuil], "color": "#f2d98a"},
                {"range": [seuil, maxi], "color": "#e8a39c"},
            ],
            "threshold": {"line": {"color": "#b3261e", "width": 4},
                          "thickness": 0.85, "value": seuil},
        }))
    fig.update_layout(height=190, margin=dict(l=18, r=18, t=38, b=6),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#14213d"})
    return fig


def frise(r):
    x = r["instants"]
    maison, clim, auto = r["profil"]
    fig = go.Figure()
    fig.add_bar(x=x, y=maison, name="Maison", marker_color="#14213d")
    fig.add_bar(x=x, y=clim, name="Climatisation", marker_color="#c9a227")
    fig.add_bar(x=x, y=auto, name="Voiture", marker_color="#4f81bd")
    fig.add_hline(y=r["seuil"], line_color="#b3261e", line_dash="dash",
                  annotation_text="Disjoncteur", annotation_position="top left",
                  annotation_font_color="#b3261e")
    fig.update_layout(
        barmode="stack", height=210, bargap=0,
        margin=dict(l=40, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,250,240,.7)",
        legend=dict(orientation="h", y=1.18, x=0, font=dict(size=11)),
        yaxis=dict(title="kW", gridcolor="rgba(20,33,61,.12)"),
        xaxis=dict(tickformat="%Hh", gridcolor="rgba(20,33,61,.08)"),
        font={"color": "#14213d"})
    return fig

# =====================================================================
# 6. CHATBOT
# =====================================================================
def contexte_texte(r):
    return (
        f"Heure actuelle {hh(r['now'], r['now'])}, retour à {hh(r['retour'], r['now'])}, "
        f"départ demain à {hh(r['depart'], r['now'])}. "
        f"Consommation de base {r['base']} kW, clim {r['clim_kw']} kW, borne {BORNE_KW} kW. "
        f"Disjoncteur {r['seuil']} kW, pic prévu {r['pic']:.2f} kW, statut {r['statut']}. "
        f"Clim allumée à {hh(r['clim_debut'], r['now'])}. "
        f"Batterie {r['bat']} % vers {r['cible']} %, charge de {hh(r['debut'], r['now'])} "
        f"à {hh(r['fin'], r['now'])}, départ à l'heure : {'oui' if r['pret_a_temps'] else 'non'}."
    )


def phrase_maison(r):
    if r["kwh"] == 0:
        return (f"Côté maison, la clim s'allume à {hh(r['clim_debut'], r['now'])} "
                "et la batterie est déjà au niveau visé.")
    return (f"Côté maison, la clim s'allume à {hh(r['clim_debut'], r['now'])} "
            f"et la voiture sera prête à {hh(r['fin'], r['now'])}.")


def reponse_locale(question, r):
    q = question.lower()
    suite = " " + phrase_maison(r)
    if "président" in q or "president" in q:
        if "tunisi" in q:
            return ("Le premier président de la République tunisienne est Habib Bourguiba, "
                    "proclamé en 1957." + suite)
    if "kw" in q or "signifie" in q or "puissance" in q:
        base, clim = r["base"], r["clim_kw"]
        return (f"Le pic de {r['demande']:.2f} kW est la somme de la maison ({base} kW), "
                f"de la clim ({clim} kW) et de la borne ({BORNE_KW} kW) si tout tournait en même temps. "
                f"En décalant la voiture, le pic réel prévu est de {r['pic']:.2f} kW.")
    if any(m in q for m in ("surcharge", "disjoncteur", "coupure", "sécurité", "securite")):
        if r["statut"] == "optimal":
            return (f"Aucun risque : le pic de {r['pic']:.2f} kW reste sous votre seuil de "
                    f"{r['seuil']} kW, la voiture peut donc démarrer tout de suite.")
        if r["statut"] == "evite":
            return (f"La coupure est évitée : la voiture attend {hh(r['debut'], r['now'])} et "
                    f"le pic tombe à {r['pic']:.2f} kW pour un seuil de {r['seuil']} kW.")
        return (f"Même en décalant la voiture, le pic reste à {r['pic']:.2f} kW, au-dessus des "
                f"{r['seuil']} kW. Il faudrait une souscription d'au moins "
                f"{np.ceil(r['pic'] * 2) / 2:.1f} kW.")
    if any(m in q for m in ("voiture", "recharge", "batterie", "charge")):
        if r["kwh"] == 0:
            return "La batterie est déjà au niveau visé, aucune charge n'est nécessaire."
        txt = (f"La charge de {r['kwh']:.1f} kWh dure environ {r['duree_h']:.1f} h, "
               f"de {hh(r['debut'], r['now'])} à {hh(r['fin'], r['now'])}.")
        if not r["pret_a_temps"]:
            txt += " Attention : la voiture ne sera pas prête à l'heure de départ."
        return txt
    if any(m in q for m in ("clim", "chambre", "frais", "température", "temperature")):
        return (f"La clim démarre à {hh(r['clim_debut'], r['now'])} pour que la chambre "
                f"soit fraîche à {hh(r['retour'], r['now'])}.")
    return ("Je n'ai pas pu joindre Gemini, donc je ne peux répondre qu'aux questions sur la maison "
            "(clim, voiture, kW, coupure)." + suite)


def appel_gemini(historique, question, r, cle):
    """Retourne (texte, modèle, erreur). Bascule automatiquement entre les modèles."""
    consigne = (
        "Tu es l'assistant énergie d'une maison. Réponds en français, en 2 à 4 phrases "
        "naturelles, sans emoji ni symbole décoratif. Réponds d'abord à la question, même "
        "si elle est hors sujet, puis termine par une courte phrase sur l'état de la maison. "
        "Explique les termes techniques si on te le demande. Contexte actuel : " + contexte_texte(r)
    )
    contenu = [{"role": "user" if m["role"] == "user" else "model",
                "parts": [{"text": m["text"]}]} for m in historique[-8:]]
    contenu.append({"role": "user", "parts": [{"text": question}]})
    corps = {"systemInstruction": {"parts": [{"text": consigne}]}, "contents": contenu,
             "generationConfig": {"temperature": 0.6, "maxOutputTokens": 400}}
    entetes = {"Content-Type": "application/json", "x-goog-api-key": cle}

    candidats = []
    memo = st.session_state.get("modele_ok")
    if memo:
        candidats.append(memo)
    candidats += [m for m in MODELES_GEMINI if m not in candidats]
    derniere = "aucune réponse"
    decouverts = False

    i = 0
    while i < len(candidats):
        modele = candidats[i]
        i += 1
        try:
            rep = requests.post(f"{API_BASE}/models/{modele}:generateContent",
                                headers=entetes, json=corps, timeout=25)
        except requests.RequestException as e:
            derniere = f"réseau ({type(e).__name__})"
            continue
        if rep.status_code == 200:
            try:
                parts = rep.json()["candidates"][0]["content"]["parts"]
                texte = "".join(p.get("text", "") for p in parts).strip()
                if texte:
                    st.session_state["modele_ok"] = modele
                    return texte, modele, None
                derniere = f"{modele} : réponse vide"
            except (KeyError, IndexError, ValueError):
                derniere = f"{modele} : réponse illisible"
            continue
        try:
            msg = rep.json().get("error", {}).get("message", "")[:140]
        except ValueError:
            msg = ""
        derniere = f"{modele} : code {rep.status_code} {msg}".strip()
        if rep.status_code in (401, 403) or (rep.status_code == 400 and "API key" in msg):
            return None, None, f"clé refusée par Google ({rep.status_code})"
        # 404, 429, 500, 503 : on passe au modèle suivant
        if i == len(candidats) and not decouverts:
            decouverts = True
            try:
                lst = requests.get(f"{API_BASE}/models", headers=entetes, timeout=15).json()
                for m in lst.get("models", []):
                    nom = m["name"].split("/")[-1]
                    if ("generateContent" in m.get("supportedGenerationMethods", [])
                            and "flash" in nom and nom not in candidats):
                        candidats.append(nom)
            except Exception:
                pass
    return None, None, derniere


def tester_gemini(cle):
    try:
        rep = requests.post(
            f"{API_BASE}/models/{MODELES_GEMINI[0]}:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": cle},
            json={"contents": [{"role": "user", "parts": [{"text": "Réponds : ok"}]}]},
            timeout=15)
        if rep.status_code == 200:
            return True, "Connexion réussie."
        return False, f"Code {rep.status_code} : {rep.text[:200]}"
    except requests.RequestException as e:
        return False, f"Réseau : {e}"

# =====================================================================
# 7. BARRE LATÉRALE
# =====================================================================
def lire_cle():
    try:
        valeur = st.secrets["GEMINI_API_KEY"]
        if valeur:
            return str(valeur)
    except Exception:
        pass
    return ""


with st.sidebar:
    st.markdown("### Réglages")
    maintenant = datetime.now().replace(second=0, microsecond=0)
    t_now = st.time_input("Heure actuelle", value=maintenant.time())
    t_ret = st.time_input("Heure de retour à la maison", value=time(18, 0))
    t_dep = st.time_input("Départ demain matin", value=time(7, 30))
    bat = st.slider("Batterie actuelle (%)", 0, 100, 40)
    cible = st.slider("Batterie visée (%)", 0, 100, 100)
    seuil = st.slider("Disjoncteur (kW)", 3.0, 15.0, 6.0, 0.5)
    clim_kw = st.slider("Puissance de la clim (kW)", 0.5, 4.0, CLIM_KW_DEFAUT, 0.25)
    st.markdown("---")
    st.markdown("### Assistant")
    cle_saisie = st.text_input("Clé Gemini", type="password",
                               help="Ou ajoutez GEMINI_API_KEY dans st.secrets.")
    cle = cle_saisie.strip() or lire_cle()
    if st.button("Tester la connexion Gemini"):
        if not cle:
            st.warning("Aucune clé renseignée.")
        else:
            ok, detail = tester_gemini(cle)
            (st.success if ok else st.error)(detail)

# =====================================================================
# 8. CALCUL ET AFFICHAGE
# =====================================================================
base_kw, origine = consommation_base()
r = calculer(t_now, t_ret, t_dep, bat, cible, seuil, clim_kw, base_kw)
ref = r["now"]

st.markdown(
    '<div class="banner"><div><h1>Énergie de la maison</h1>'
    '<p>Climatisation, recharge de la voiture et disjoncteur, réglés sur vos horaires.</p></div>'
    '<svg width="120" height="44" viewBox="0 0 120 44" fill="none" stroke="#c9a227" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M6 22 28 6l22 16"/><path d="M11 20v18h34V20"/><path d="M24 38V28h8v10"/>'
    '<path d="M58 34c10 0 10-22 20-22s10 22 20 22 8-10 16-10"/></svg></div>',
    unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    carte("maison", "Maison", "Consommation normale",
          f"{r['base']} kW en continu ({origine.lower()})", "#14213d")
with c2:
    carte("thermo", "Confort", f"Chambre fraîche à {hh(r['retour'], ref)}",
          f"Clim allumée à {hh(r['clim_debut'], ref)}, {r['clim_kw']} kW", "#c9a227")
with c3:
    if r["kwh"] == 0:
        carte("voiture", "Voiture", "Batterie déjà au niveau visé",
              "Aucune charge nécessaire", "#2e7d32")
    else:
        couleur = "#2e7d32" if r["pret_a_temps"] else "#b3261e"
        carte("voiture", "Voiture", f"Prête à {hh(r['fin'], ref)}",
              f"Charge de {hh(r['debut'], ref)} à {hh(r['fin'], ref)}, {BORNE_KW} kW", couleur)
with c4:
    titres = {"optimal": ("Réseau optimal", "#2e7d32"),
              "evite": ("Coupure évitée", "#c9a227"),
              "critique": ("Risque de coupure", "#b3261e")}
    gros, coul = titres[r["statut"]]
    carte("bouclier", "Sécurité", gros,
          f"Pic de {r['pic']:.2f} kW pour {r['seuil']} kW", coul)

st.write("")
gauche, droite = st.columns([3, 2], gap="medium")

with gauche:
    g1, g2 = st.columns([2, 3])
    with g1:
        st.plotly_chart(jauge(r), use_container_width=True, config={"displayModeBar": False})
    with g2:
        if r["statut"] == "optimal":
            p1 = (f"Votre disjoncteur de {r['seuil']} kW supporte tout en même temps : "
                  f"le pic reste à {r['pic']:.2f} kW. La voiture démarre dès {hh(r['debut'], ref)}.")
        elif r["statut"] == "evite":
            p1 = (f"Si tout démarrait ensemble, la demande monterait à {r['demande']:.2f} kW, "
                  f"au-dessus des {r['seuil']} kW. La voiture attend donc {hh(r['debut'], ref)} "
                  f"et le pic tombe à {r['pic']:.2f} kW.")
        else:
            p1 = (f"Même avec la voiture décalée à {hh(r['debut'], ref)}, le pic atteint "
                  f"{r['pic']:.2f} kW, au-dessus des {r['seuil']} kW. Une souscription d'au moins "
                  f"{np.ceil(r['pic'] * 2) / 2:.1f} kW est conseillée.")
        p2 = (f"La clim s'allume à {hh(r['clim_debut'], ref)} pour une chambre fraîche à "
              f"{hh(r['retour'], ref)}.")
        alerte = ""
        if not r["pret_a_temps"]:
            alerte = (f'<div class="alerte">La voiture ne sera pas prête à {hh(r["depart"], ref)} : '
                      f'fin de charge prévue à {hh(r["fin"], ref)}. Baissez la batterie visée '
                      "ou augmentez le seuil du disjoncteur.</div>")
        st.markdown(f'<div class="panel"><h4>Décision du système</h4><p>{p1}</p><p>{p2}</p>{alerte}</div>',
                    unsafe_allow_html=True)
    st.plotly_chart(frise(r), use_container_width=True, config={"displayModeBar": False})

with droite:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    etat = (f"Gemini connecté ({st.session_state['modele_ok']})"
            if st.session_state.get("modele_ok") else
            ("Gemini prêt, en attente d'une question" if cle else "Mode local, aucune clé Gemini"))
    st.markdown(f'<div class="chat-title">Assistant</div><div class="chat-state">{html.escape(etat)}</div>',
                unsafe_allow_html=True)

    zone = st.container(height=330, border=True)
    question = st.chat_input("Posez une question sur la maison ou autre chose")

    if question:
        texte, erreur = None, None
        if cle:
            texte, modele, erreur = appel_gemini(st.session_state.messages, question, r, cle)
            if texte is None:
                st.session_state.pop("modele_ok", None)
        if texte is None:
            texte = reponse_locale(question, r)
        st.session_state.messages.append({"role": "user", "text": question})
        st.session_state.messages.append({"role": "bot", "text": texte,
                                          "erreur": erreur if cle else None})

    with zone:
        if not st.session_state.messages:
            st.markdown('<div class="msg bot"><span class="who">Assistant</span>'
                        "Bonjour. Demandez-moi pourquoi la voiture démarre à telle heure, "
                        "ce que signifient les kW, ou toute autre question.</div>",
                        unsafe_allow_html=True)
        for m in st.session_state.messages:
            corps = html.escape(m["text"]).replace("\n", "<br>")
            if m["role"] == "user":
                st.markdown(f'<div class="msg user"><span class="who">Vous</span>{corps}</div>',
                            unsafe_allow_html=True)
            else:
                note = (f'<span class="msg err">Gemini indisponible : {html.escape(m["erreur"])}. '
                        "Réponse locale.</span>") if m.get("erreur") else ""
                st.markdown(f'<div class="msg bot"><span class="who">Assistant</span>{corps}{note}</div>',
                            unsafe_allow_html=True)

    # fait défiler la zone de chat jusqu'au dernier message
    components.html(
        "<script>const d=window.parent.document;"
        "d.querySelectorAll('div').forEach(e=>{const s=getComputedStyle(e);"
        "if((s.overflowY==='auto'||s.overflowY==='scroll')&&e.scrollHeight>e.clientHeight+5"
        "&&e.clientHeight<600){e.scrollTop=e.scrollHeight;}});</script>", height=0)
