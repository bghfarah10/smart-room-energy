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
