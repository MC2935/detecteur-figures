"""
Détecteur de figures chartistes — application Streamlit.
Lancer en local :  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from patterns import analyze, HAUSSIER, _overlap

st.set_page_config(page_title="Détecteur de figures", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
VALEURS = {
    "CAC 40 (indice)": "^FCHI", "LVMH": "MC.PA", "TotalEnergies": "TTE.PA",
    "Hermès": "RMS.PA", "L'Oréal": "OR.PA", "Schneider Electric": "SU.PA",
    "Airbus": "AIR.PA", "Safran": "SAF.PA", "Sanofi": "SAN.PA",
    "Air Liquide": "AI.PA", "BNP Paribas": "BNP.PA", "AXA": "CS.PA",
    "EssilorLuxottica": "EL.PA", "Vinci": "DG.PA", "Société Générale": "GLE.PA",
    "Crédit Agricole": "ACA.PA", "Orange": "ORA.PA", "Thales": "HO.PA",
    "Capgemini": "CAP.PA", "Dassault Systèmes": "DSY.PA", "Danone": "BN.PA",
    "Pernod Ricard": "RI.PA", "Kering": "KER.PA", "Stellantis": "STLAP.PA",
    "Engie": "ENGI.PA", "Veolia": "VIE.PA", "Saint-Gobain": "SGO.PA",
    "Legrand": "LR.PA", "Michelin": "ML.PA", "Renault": "RNO.PA",
    "Publicis": "PUB.PA", "STMicroelectronics": "STMPA.PA", "Carrefour": "CA.PA",
    "Bouygues": "EN.PA", "Accor": "AC.PA", "Bureau Veritas": "BVI.PA",
    "ArcelorMittal": "MT.AS", "Eurofins": "ERF.PA",
    "Or (future)": "GC=F", "Pétrole Brent (future)": "BZ=F", "EUR/USD": "EURUSD=X",
}

# Limites Yahoo Finance pour l'historique intraday
INTERVALLES = {
    "15 min": ("15m", "60d"),
    "30 min": ("30m", "60d"),
    "1 heure": ("1h", "730d"),
    "4 heures (reconstitué)": ("1h", "730d"),
    "Jour": ("1d", "5y"),
    "Semaine": ("1wk", "10y"),
}

FIGURES = {
    "Tête-épaules (+ inversée)": "ete",
    "Doubles sommets / creux": "double",
    "Triangles": "triangle",
    "Drapeaux / fanions": "drapeau",
    "Imbalances (FVG)": "fvg",
}

DESSINS = {
    "Figures chartistes": "figures",
    "Objectifs et invalidations": "objectifs",
    "Supports / résistances": "sr",
    "Canal": "canal",
    "Obliques de tendance": "obliques",
    "Imbalances (FVG)": "fvg",
    "Moyennes mobiles 50 / 200": "mm",
}

VERT, ROUGE, GRIS = "#1a9850", "#d73027", "#888888"


@st.cache_data(ttl=300, show_spinner=False)
def charger(ticker: str, libelle: str) -> pd.DataFrame:
    interval, period = INTERVALLES[libelle]
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    try:
        df.index = df.index.tz_convert("Europe/Paris")
    except (TypeError, AttributeError):
        pass
    if libelle.startswith("4 heures"):
        df = df.resample("4h", offset="1h").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna()
    return df


def fmt(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}".replace(",", " ")


def labels(idx, libelle):
    f = "%d/%m/%y" if libelle in ("Jour", "Semaine") else "%d/%m/%y %H:%M"
    return [t.strftime(f) for t in idx]


def ratio_rr(p, last):
    if p["objectif"] is None or p["invalidation"] is None:
        return None
    if p["biais"] == HAUSSIER:
        gain, risque = p["objectif"] - last, last - p["invalidation"]
    else:
        gain, risque = last - p["objectif"], p["invalidation"] - last
    if gain <= 0 or risque <= 0:
        return None
    return gain / risque


# ---------------------------------------------------------------------------
# Graphique
# ---------------------------------------------------------------------------

BLEU, VIOLET = "#2c7bb6", "#7b3294"


def _ligne(fig, xs, ys, col, width=2, dash="solid", texte=None, pos="top right"):
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+text" if texte else "lines",
        line=dict(color=col, width=width, dash=dash),
        text=([""] * (len(xs) - 1) + [texte]) if texte else None,
        textposition=pos, textfont=dict(size=10, color=col),
        hoverinfo="skip", showlegend=False))


def figures_a_dessiner(res, montrer_hist, max_actives=3):
    """Évite un graphique illisible : au plus 3 figures actives, sans chevauchement fort
    (on garde la mieux notée), plus les figures passées si demandé."""
    actives = sorted([p for p in res["figures"] if p["active"]], key=lambda p: -p["score"])
    garde = []
    for p in actives:
        if len(garde) >= max_actives:
            break
        if all(_overlap(p, g) <= 0.5 for g in garde):
            garde.append(p)
    if montrer_hist:
        garde += [p for p in res["figures"] if not p["active"]]
    return garde


def graphique(res, libelle, n_aff, montrer_hist, dessins, max_fvg=8):
    """dessins : ensemble parmi {"figures", "objectifs", "sr", "canal", "obliques", "fvg", "mm"}."""
    df = res["data"]
    n = len(df)
    x = np.arange(n)
    lab = labels(df.index, libelle)
    w0 = max(0, n - n_aff)
    xr = n + 8          # les tracés débordent un peu à droite du dernier cours

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        text=lab, name="Cours", increasing_line_color=VERT, decreasing_line_color=ROUGE,
        showlegend=False))
    if "mm" in dessins:
        fig.add_trace(go.Scatter(x=x, y=res["ema50"], name="MM50",
                                 line=dict(width=1, color="#4575b4"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=res["ema200"], name="MM200",
                                 line=dict(width=1, color="#f46d43"), hoverinfo="skip"))

    ylo = float(df["Low"].iloc[w0:].min())
    yhi = float(df["High"].iloc[w0:].max())
    last = float(df["Close"].iloc[-1])

    # Imbalances
    if "fvg" in dessins:
        act = [f for f in res["fvg"] if f["actif"] and f["idx"] >= w0]
        act = sorted(act, key=lambda f: abs((f["haut"] + f["bas"]) / 2 - last))[:max_fvg]
        for f in act:
            col = "rgba(26,152,80,0.18)" if f["biais"] == HAUSSIER else "rgba(215,48,39,0.18)"
            fig.add_shape(type="rect", x0=f["idx"], x1=xr, y0=f["bas"], y1=f["haut"],
                          fillcolor=col, line_width=0, layer="below")

    # Supports / résistances horizontaux
    if "sr" in dessins:
        for z in res.get("niveaux", []):
            col = VERT if z["type"] == "support" else ROUGE
            x0 = max(z["debut"], w0)
            fig.add_shape(type="rect", x0=x0, x1=xr, y0=z["bas"], y1=z["haut"],
                          fillcolor=col, opacity=0.08, line_width=0, layer="below")
            _ligne(fig, [x0, xr], [z["niveau"]] * 2, col, 1.5, "dash",
                   f"{z['type'].capitalize()} {fmt(z['niveau'])} ({z['contacts']} contacts)",
                   "top left" if z["type"] == "résistance" else "bottom left")

    # Canal
    c = res.get("canal")
    if "canal" in dessins and c:
        x0 = max(c["debut"], w0 - 20)
        xs = [x0, xr]
        haut = [c["pente"] * i + c["b_haut"] for i in xs]
        bas = [c["pente"] * i + c["b_bas"] for i in xs]
        milieu = [(h + b) / 2 for h, b in zip(haut, bas)]
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=haut + bas[::-1], fill="toself",
                                 fillcolor="rgba(123,50,148,0.06)", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        _ligne(fig, xs, haut, VIOLET, 2, texte=f"{c['nom']} ({c['etat']})", pos="top left")
        _ligne(fig, xs, bas, VIOLET, 2)
        _ligne(fig, xs, milieu, VIOLET, 1, "dot")

    # Obliques de tendance
    if "obliques" in dessins:
        for o in res.get("obliques", []):
            col = BLEU
            y_end = o["y1"] + o["pente"] * (xr - o["x1"])
            dash = "solid" if o["etat"] == "respectée" else "dashdot"
            _ligne(fig, [o["x1"], xr], [o["y1"], y_end], col, 2, dash,
                   f"{o['nom']} ({o['etat']})",
                   "bottom left" if o["biais"] == HAUSSIER else "top left")
            fig.add_trace(go.Scatter(x=[o["x1"], o["x2"]], y=[o["y1"], o["y2"]], mode="markers",
                                     marker=dict(size=8, color=col, symbol="circle-open"),
                                     hoverinfo="skip", showlegend=False))

    # Figures chartistes
    if "figures" in dessins:
        a_dessiner = figures_a_dessiner(res, montrer_hist)
        for k_fig, p in enumerate(a_dessiner):
            if p["fin"] < w0 and not (p["cassure"] and p["cassure"] >= w0):
                continue
            col = (VERT if p["biais"] == HAUSSIER else ROUGE) if p["active"] else GRIS
            dash = "solid" if p["active"] else "dot"
            for (x1, y1), (x2, y2), nom_l in p["lignes"]:
                _ligne(fig, [x1, x2], [y1, y2], col, 2, dash)
            fig.add_trace(go.Scatter(
                x=[pt[0] for pt in p["points"]], y=[pt[1] for pt in p["points"]],
                mode="markers+text", text=[pt[2] for pt in p["points"]],
                textposition=["top center" if (pt[1] >= np.median([q[1] for q in p["points"]]))
                              else "bottom center" for pt in p["points"]],
                textfont=dict(size=10, color=col), marker=dict(size=6, color=col),
                hovertext=f"{p['figure']} – {p['statut']}", hoverinfo="text", showlegend=False))
            # nom de la figure au-dessus (ou au-dessous) de la figure
            ys = [pt[1] for pt in p["points"]]
            haut_fig = p["biais"] != HAUSSIER
            fig.add_annotation(x=(p["debut"] + p["fin"]) / 2,
                               y=max(ys) if haut_fig else min(ys),
                               text=f"<b>{p['figure']}</b><br>{p['statut']}",
                               showarrow=False, yshift=(28 + 26 * k_fig) * (1 if haut_fig else -1),
                               font=dict(size=11, color=col),
                               bgcolor="rgba(255,255,255,0.7)")
            if p["active"] and "objectifs" in dessins:
                x_s = p["cassure"] if p["cassure"] is not None else n - 1
                # flèche de projection vers l'objectif
                fig.add_annotation(x=x_s + 3, y=p["objectif"], ax=x_s + 3, ay=p["niveau_cassure"],
                                   xref="x", yref="y", axref="x", ayref="y",
                                   showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=col, text="")
                for y, nom_l, cc in ((p["objectif"], "Objectif", col),
                                     (p["invalidation"], "Invalidation", GRIS)):
                    if y is None:
                        continue
                    _ligne(fig, [x_s, xr], [y, y], cc, 1, "dash", f"{nom_l} {fmt(y)}", "top left")
                    ylo, yhi = min(ylo, y), max(yhi, y)

    pad = (yhi - ylo) * 0.06
    step = max(1, n_aff // 6)
    ticks = list(range(w0, n, step))
    fig.update_layout(
        height=600, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[w0 - 1, xr + 2], tickvals=ticks, ticktext=[lab[i] for i in ticks],
                   rangeslider=dict(visible=False)),
        yaxis=dict(range=[ylo - pad, yhi + pad], side="right"),
        legend=dict(orientation="h", y=1.02, x=0),
        dragmode="pan",
    )
    return fig


# ---------------------------------------------------------------------------
# Tableaux
# ---------------------------------------------------------------------------

def tableau_figures(res, libelle, actives=True):
    df = res["data"]
    last = float(df["Close"].iloc[-1])
    rows = []
    for p in res["figures"]:
        if p["active"] != actives:
            continue
        rr = ratio_rr(p, last) if actives else None
        rows.append({
            "Figure": p["figure"],
            "Biais": p["biais"],
            "Statut": p["statut"],
            "Fin de figure": df.index[p["fin"]].strftime("%d/%m/%y %H:%M"),
            "Cassure": fmt(p["niveau_cassure"]),
            "Objectif": fmt(p["objectif"]),
            "Invalidation": fmt(p["invalidation"]),
            "Gain/risque": fmt(rr, 1) if rr else "—",
            "Score": p["score"],
        })
    return pd.DataFrame(rows)


def stats_historiques(res):
    hist = [p for p in res["figures"] if p["statut"] in
            ("objectif atteint", "échec après cassure", "confirmée")]
    if not hist:
        return pd.DataFrame()
    df = pd.DataFrame([{"Figure": p["figure"].split(" (")[0], "Statut": p["statut"]} for p in hist])
    g = df.groupby("Figure")["Statut"]
    out = pd.DataFrame({
        "Cassures": g.size(),
        "Objectif atteint": g.apply(lambda s: (s == "objectif atteint").sum()),
        "Échec": g.apply(lambda s: (s == "échec après cassure").sum()),
    })
    termine = out["Objectif atteint"] + out["Échec"]
    out["Taux de réussite"] = (out["Objectif atteint"] / termine.replace(0, np.nan) * 100).round(0)
    return out.sort_values("Cassures", ascending=False)


# ---------------------------------------------------------------------------
# Analyse rédigée automatiquement
# ---------------------------------------------------------------------------

def pct(a, b):
    return (a / b - 1) * 100


def niveaux_cles(res):
    sup = [z for z in res.get("niveaux", []) if z["type"] == "support"]
    resi = [z for z in res.get("niveaux", []) if z["type"] == "résistance"]
    return (sup[0] if sup else None), (resi[0] if resi else None)


def commentaire(res, nom, libelle):
    df = res["data"]
    last = float(df["Close"].iloc[-1])
    e50, e200 = float(res["ema50"].iloc[-1]), float(res["ema200"].iloc[-1])
    lignes = []

    # 1. Tendance de fond
    if len(df) >= 200:
        if last > e50 > e200:
            t = "la tendance est **haussière** : le cours est au-dessus de ses moyennes 50 et 200, elles-mêmes bien ordonnées"
        elif last < e50 < e200:
            t = "la tendance est **baissière** : le cours est sous ses moyennes 50 et 200, elles-mêmes orientées dans ce sens"
        elif last > e200:
            t = "le cours reste au-dessus de sa moyenne 200 (fond plutôt porteur) mais la dynamique récente est hésitante"
        else:
            t = "le cours est sous sa moyenne 200 (fond plutôt fragile) mais tente de se reprendre à court terme"
    else:
        t = "l'historique est trop court pour juger la tendance de fond"
    lignes.append(f"**Tendance** — En {libelle.lower()}, {t} "
                  f"(MM50 {fmt(e50)}, MM200 {fmt(e200)}).")

    # 2. Supports / résistances
    sup, resi = niveaux_cles(res)
    morceaux = []
    if resi:
        morceaux.append(f"résistance la plus proche **{fmt(resi['niveau'])}** "
                        f"({pct(resi['niveau'], last):+.1f} %, {resi['contacts']} contacts)")
    if sup:
        morceaux.append(f"support le plus proche **{fmt(sup['niveau'])}** "
                        f"({pct(sup['niveau'], last):+.1f} %, {sup['contacts']} contacts)")
    if morceaux:
        lignes.append("**Niveaux clés** — " + ", ".join(morceaux) + ".")
    else:
        lignes.append("**Niveaux clés** — Pas de support ni de résistance horizontal touché au moins deux fois récemment.")

    c = res.get("canal")
    if c:
        if c["etat"] == "en cours":
            pos = (last - c["bas_actuel"]) / c["largeur"] * 100
            ou = "près du bas" if pos < 33 else "près du haut" if pos > 67 else "au milieu"
            lignes.append(f"**Canal** — Le cours évolue dans un **{c['nom'].lower()}**, actuellement {ou} "
                          f"(borne basse {fmt(c['bas_actuel'])}, borne haute {fmt(c['haut_actuel'])}).")
        else:
            lignes.append(f"**Canal** — Le {c['nom'].lower()} précédent a été **{c['etat']}**.")
    for o in res.get("obliques", []):
        if o["etat"] == "respectée":
            role = "soutient le cours" if o["biais"] == HAUSSIER else "freine le cours"
            lignes.append(f"**{o['nom']}** — Elle {role} ({o['contacts']} contacts), "
                          f"passant aujourd'hui vers {fmt(o['niveau_actuel'])}.")
        else:
            sens_o = "signal de faiblesse" if o["biais"] == HAUSSIER else "signal de reprise"
            lignes.append(f"**{o['nom']}** — Elle vient d'être cassée ({sens_o}), "
                          f"niveau {fmt(o['niveau_actuel'])}.")

    # 3. Figures actives
    stats = stats_historiques(res)
    actives = sorted([p for p in res["figures"] if p["active"]], key=lambda p: -p["score"])[:4]
    if not actives:
        lignes.append("**Figures** — Aucune figure chartiste récente n'est en cours sur cette unité de temps.")
    for p in actives:
        date = df.index[p["fin"]].strftime("%d/%m")
        if p["statut"] == "confirmée":
            debut = (f"**{p['figure']}** — figure confirmée (cassure à {fmt(p['niveau_cassure'])} "
                     f"le {df.index[p['cassure']].strftime('%d/%m')})")
        else:
            sens_c = "au-dessus de" if p["biais"] == HAUSSIER else "sous"
            debut = (f"**{p['figure']}** — figure en formation depuis le {date} : la cassure est à surveiller "
                     f"{sens_c} {fmt(p['niveau_cassure'])}")
        txt = (f"{debut}. Biais {p['biais']}, objectif théorique **{fmt(p['objectif'])}** "
               f"({pct(p['objectif'], last):+.1f} %), invalidation à {fmt(p['invalidation'])} "
               f"({pct(p['invalidation'], last):+.1f} %).")
        rr = ratio_rr(p, last)
        if rr:
            txt += f" Rapport gain/risque depuis le cours actuel : {rr:.1f}."
        else:
            txt += " L'objectif est déjà presque atteint ou le cours est au-delà de l'invalidation : peu d'intérêt à ce stade."
        base = p["figure"].split(" (")[0]
        if not stats.empty and base in stats.index:
            s = stats.loc[base]
            fini = int(s["Objectif atteint"] + s["Échec"])
            if fini >= 2:
                txt += f" Sur ce titre, cette figure a atteint son objectif {int(s['Objectif atteint'])} fois sur {fini}."
        lignes.append(txt)

    # 4. Imbalances
    act = [f for f in res["fvg"] if f["actif"]]
    dessous = [f for f in act if f["haut"] < last]
    dessus = [f for f in act if f["bas"] > last]
    m = []
    if dessous:
        f = max(dessous, key=lambda f: f["haut"])
        m.append(f"une zone non comblée sous le cours entre {fmt(f['bas'])} et {fmt(f['haut'])} "
                 f"({pct(f['haut'], last):+.1f} %), qui peut servir de support ou attirer le prix")
    if dessus:
        f = min(dessus, key=lambda f: f["bas"])
        m.append(f"une zone au-dessus entre {fmt(f['bas'])} et {fmt(f['haut'])} "
                 f"({pct(f['bas'], last):+.1f} %), qui peut freiner une hausse ou attirer le prix")
    if m:
        lignes.append("**Imbalances** — " + " ; ".join(m) + ".")

    # 5. Conclusion
    v = res["synthese"]["verdict"]
    lignes.append(f"**En résumé** — Bilan graphique pour {nom} en {libelle.lower()} : **{v.lower()}**. "
                  "Ce n'est qu'une lecture des configurations : à croiser avec l'actualité "
                  "(résultats, annonces) et une autre unité de temps.")
    return lignes


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
NOM_DE = {v: k for k, v in VALEURS.items()}
LISTE_DEFAUT = ["^FCHI", "MC.PA", "TTE.PA", "AIR.PA", "BNP.PA", "SAF.PA"]


def nom(t):
    return NOM_DE.get(t, t)


st.title("📈 Détecteur de figures chartistes")
st.caption("Repère des configurations graphiques connues et leur objectif théorique. "
           "Ce n'est ni une prédiction ni un conseil en investissement.")

# Liste mémorisée dans l'adresse de la page : mettez la page en favori pour la retrouver.
qp = st.query_params.get("liste")
liste_init = [t for t in qp.split(",") if t] if qp else LISTE_DEFAUT
connus = [t for t in liste_init if t in NOM_DE]
perso_init = ", ".join(t for t in liste_init if t not in NOM_DE)

with st.sidebar:
    st.header("Ma liste d'actions")
    choix_noms = st.multiselect("Valeurs suivies", list(VALEURS), default=[nom(t) for t in connus])
    perso = st.text_input("Autres codes Yahoo (séparés par des virgules)", perso_init,
                          help="Ex. : AAPL, NVDA, ASML.AS, ^GSPC")
    liste = [VALEURS[n] for n in choix_noms] + \
            [c.strip().upper() for c in perso.split(",") if c.strip()]
    nouvelle = ",".join(liste)
    if nouvelle != (qp or ""):
        st.query_params["liste"] = nouvelle
    st.caption("La liste est enregistrée dans l'adresse de la page : ajoutez-la en favori "
               "ou à l'écran d'accueil pour la retrouver.")

    st.header("Réglages")
    libelle = st.selectbox("Unité de temps", list(INTERVALLES), index=2)
    sens = st.slider("Sensibilité des pivots (× ATR)", 1.0, 4.0, 2.0, 0.25,
                     help="Plus petit : plus de figures, plus de bruit. Plus grand : seulement les grandes figures.")
    figs = st.multiselect("Figures à détecter", list(FIGURES), default=list(FIGURES))
    n_aff = st.slider("Nombre de bougies affichées", 60, 1000, 250, 10)
    recent = st.slider("Une figure est « récente » si elle date de moins de … bougies", 20, 300, 80, 10)
    dessins_noms = st.multiselect("Tracés sur le graphique", list(DESSINS), default=list(DESSINS))
    montrer_hist = st.checkbox("Afficher aussi les figures passées (en gris)", False)
    fvg_min = st.slider("Taille minimale d'une imbalance (× ATR)", 0.0, 1.5, 0.3, 0.1)

codes_fig = {FIGURES[f] for f in figs}


def analyser(ticker):
    """Téléchargement + analyse ; renvoie (res, erreur)."""
    try:
        d = charger(ticker, libelle)
    except Exception as e:  # noqa: BLE001
        return None, f"téléchargement impossible ({e})"
    if d.empty or len(d) < 60:
        return None, "pas assez de données (code erroné ou limite Yahoo)"
    return analyze(d, sensibilite=sens, recent_bars=recent, fvg_min_atr=fvg_min,
                   figures=codes_fig), None


onglet_liste, onglet_scan, onglet_aide = st.tabs(["Ma liste", "Scanner", "Mode d'emploi"])

with onglet_liste:
    if not liste:
        st.info("Ajoutez des valeurs dans « Ma liste d'actions » (menu ☰ / barre latérale).")
    else:
        # --- Vue d'ensemble -------------------------------------------------
        resultats, lignes = {}, []
        with st.spinner("Analyse de la liste…"):
            for t in liste:
                r, err = analyser(t)
                resultats[t] = (r, err)
                if r is None:
                    lignes.append({"Valeur": nom(t), "Cours": "—", "Var.": "—",
                                   "Lecture": f"⚠️ {err}", "Figures actives": 0,
                                   "Figure principale": "—"})
                    continue
                d = r["data"]
                last = float(d["Close"].iloc[-1])
                ref = float(d["Close"].iloc[max(0, len(d) - 21)])
                act = sorted([p for p in r["figures"] if p["active"]], key=lambda p: -p["score"])
                v = r["synthese"]["verdict"]
                icone = "🟢" if "haussier" in v else "🔴" if "baissier" in v else "⚪"
                lignes.append({
                    "Valeur": nom(t), "Cours": fmt(last), "Var.": f"{pct(last, ref):+.1f} %",
                    "Lecture": f"{icone} {v}", "Figures actives": len(act),
                    "Figure principale": f"{act[0]['figure']} ({act[0]['statut']})" if act else "—",
                })
        st.dataframe(pd.DataFrame(lignes), use_container_width=True, hide_index=True)
        st.caption(f"Var. = variation sur les 20 dernières bougies ({libelle.lower()}).")

        # --- Sélection d'une action ----------------------------------------
        choix = st.pills("Choisir une action à afficher", [nom(t) for t in liste],
                         selection_mode="single", default=nom(liste[0]))
        if choix is None:
            choix = nom(liste[0])
        ticker = next(t for t in liste if nom(t) == choix)
        res, err = resultats[ticker]

        if res is None:
            st.warning(f"{choix} : {err}")
        else:
            data = res["data"]
            last = float(data["Close"].iloc[-1])
            syn = res["synthese"]
            couleur = VERT if "haussier" in syn["verdict"] else ROUGE if "baissier" in syn["verdict"] else GRIS

            st.subheader(f"{choix} · {libelle}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Dernier cours", fmt(last))
            c2.metric("Dernière bougie", data.index[-1].strftime("%d/%m %H:%M"))
            c3.markdown(f"<div style='padding-top:6px'>Lecture graphique<br>"
                        f"<span style='font-size:1.5em;font-weight:700;color:{couleur}'>"
                        f"{syn['verdict']}</span></div>", unsafe_allow_html=True)

            st.plotly_chart(graphique(res, libelle, n_aff, montrer_hist,
                                      {DESSINS[d] for d in dessins_noms} & (codes_fig | {"objectifs", "sr", "canal", "obliques", "mm", "figures"})),
                            use_container_width=True,
                            config={"scrollZoom": True, "displaylogo": False})

            st.markdown("#### Analyse automatique")
            for l in commentaire(res, choix, libelle):
                st.markdown(l)

            with st.expander("Détail des figures actives"):
                t_act = tableau_figures(res, libelle, True)
                if t_act.empty:
                    st.write("Aucune.")
                else:
                    st.dataframe(t_act, use_container_width=True, hide_index=True)

            if "fvg" in codes_fig:
                with st.expander("Imbalances (FVG) non comblées les plus proches"):
                    act = sorted([f for f in res["fvg"] if f["actif"]],
                                 key=lambda f: abs((f["haut"] + f["bas"]) / 2 - last))[:10]
                    if act:
                        st.dataframe(pd.DataFrame([{
                            "Date": data.index[f["idx"]].strftime("%d/%m/%y %H:%M"),
                            "Type": f["biais"], "Bas": fmt(f["bas"]), "Haut": fmt(f["haut"]),
                            "Position": "sous le cours" if f["haut"] < last
                            else "au-dessus" if f["bas"] > last else "cours dedans",
                            "Comblée à": f"{f['comblement'] * 100:.0f} %",
                        } for f in act]), use_container_width=True, hide_index=True)
                    else:
                        st.write("Aucune.")

            with st.expander("Historique : les figures ont-elles fonctionné sur ce titre ?"):
                s = stats_historiques(res)
                if s.empty:
                    st.write("Pas assez de figures passées.")
                else:
                    st.dataframe(s, use_container_width=True)
                    st.caption("Sur l'historique téléchargé, pour cette unité de temps et cette sensibilité. "
                               "Un taux proche de 50 % ou moins : la figure n'apporte pas grand-chose ici.")
                th = tableau_figures(res, libelle, False)
                if not th.empty:
                    st.dataframe(th, use_container_width=True, hide_index=True)

with onglet_scan:
    st.write("Analyse toutes les valeurs proposées (pas seulement votre liste) "
             "sur l'unité de temps choisie et liste les figures actives.")
    if st.button("Lancer le scan", type="primary"):
        lignes_s = []
        barre = st.progress(0.0)
        noms = list(VALEURS)
        for k, n_ in enumerate(noms):
            barre.progress((k + 1) / len(noms), text=n_)
            r, err = analyser(VALEURS[n_])
            if r is None:
                continue
            lc = float(r["data"]["Close"].iloc[-1])
            for p in r["figures"]:
                if p["active"]:
                    rr = ratio_rr(p, lc)
                    lignes_s.append({"Valeur": n_, "Figure": p["figure"], "Biais": p["biais"],
                                     "Statut": p["statut"], "Cours": fmt(lc),
                                     "Objectif": fmt(p["objectif"]),
                                     "Invalidation": fmt(p["invalidation"]),
                                     "Gain/risque": round(rr, 1) if rr else None,
                                     "Score": p["score"],
                                     "Lecture globale": r["synthese"]["verdict"]})
        barre.empty()
        if lignes_s:
            st.dataframe(pd.DataFrame(lignes_s).sort_values(["Statut", "Score"], ascending=[True, False]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Aucune figure active trouvée.")

with onglet_aide:
    st.markdown("""
**Utilisation**
1. Dans la barre latérale (☰ sur mobile), choisissez vos valeurs dans **Ma liste d'actions**.
2. L'onglet **Ma liste** affiche un tableau récapitulatif, puis touchez le nom d'une action :
   son graphique et son **analyse automatique** s'affichent.
3. La liste est mémorisée dans l'adresse de la page : mettez-la en favori.

**Ce que fait l'outil**
- Repère les sommets et creux significatifs (pivots), d'une amplitude minimale de *sensibilité × ATR*.
- Cherche les figures dans ces pivots, puis suit ce qui s'est passé ensuite.
- **Objectif** = règle classique de projection : hauteur de la figure (ou du mât pour un drapeau) reportée depuis la cassure.
- **Invalidation** = niveau au-delà duquel la figure n'a plus de sens (référence possible pour un stop).

**Statuts**
- *En formation* : la figure est dessinée mais la cassure n'a pas eu lieu.
- *Confirmée* : cassure en clôture, objectif pas encore atteint.
- *Objectif atteint*, *échec après cassure*, *invalidée*, *expirée* : figures terminées (visibles dans l'historique).

**Imbalances (FVG)** : écart laissé entre la bougie 1 et la bougie 3 lors d'un mouvement rapide.
Zone verte = sous un mouvement haussier, rouge = sous un mouvement baissier. Elles disparaissent une fois comblées.

**À garder en tête**
- Des figures apparaissent aussi dans des cours purement aléatoires : regardez l'historique pour voir si elles ont fonctionné sur ce titre.
- Les résultats d'entreprise et l'actualité peuvent invalider une figure instantanément.
- Données Yahoo Finance, gratuites et non officielles, parfois retardées de 15 min. Historique limité à 60 jours en 15/30 min et 2 ans en 1 h.
""")
