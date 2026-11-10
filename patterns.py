"""
Moteur de détection de figures chartistes.

Aucune prédiction : l'outil repère des configurations géométriques connues,
indique le biais qu'on leur attribue classiquement, l'objectif théorique
(règle de projection de la hauteur) et le niveau d'invalidation.

Entrée : DataFrame avec colonnes Open, High, Low, Close (Volume optionnel),
index temporel trié.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HAUSSIER = "haussier"
BAISSIER = "baissier"

# ---------------------------------------------------------------------------
# Outils de base
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range (lissage de Wilder)."""
    pc = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def zigzag(df: pd.DataFrame, atr_s: pd.Series, mult: float):
    """Pivots hauts/bas alternés. Un pivot est validé quand le prix
    repart en sens inverse d'au moins `mult` x ATR.
    Renvoie (pivots confirmés, dernier extrême provisoire)."""
    H, L, A = df["High"].values, df["Low"].values, atr_s.values
    n = len(df)
    pivots = []
    trend, ext_i, hi_i, lo_i = 0, 0, 0, 0
    for i in range(1, n):
        thr = mult * A[i]
        if trend == 0:
            if H[i] > H[hi_i]:
                hi_i = i
            if L[i] < L[lo_i]:
                lo_i = i
            if H[hi_i] - L[lo_i] >= thr:
                if hi_i > lo_i:
                    pivots.append((lo_i, L[lo_i], "L"))
                    trend, ext_i = 1, hi_i
                else:
                    pivots.append((hi_i, H[hi_i], "H"))
                    trend, ext_i = -1, lo_i
        elif trend == 1:
            if H[i] >= H[ext_i]:
                ext_i = i
            elif H[ext_i] - L[i] >= thr:
                pivots.append((ext_i, H[ext_i], "H"))
                trend, ext_i = -1, i
        else:
            if L[i] <= L[ext_i]:
                ext_i = i
            elif H[i] - L[ext_i] >= thr:
                pivots.append((ext_i, L[ext_i], "L"))
                trend, ext_i = 1, i
    prov = None
    if trend == 1:
        prov = (ext_i, H[ext_i], "H")
    elif trend == -1:
        prov = (ext_i, L[ext_i], "L")
    return pivots, prov


class View:
    """Vue « miroir » des prix : s=+1 prix normaux, s=-1 prix inversés.
    Permet d'écrire une seule fois la logique d'une figure (ex. tête-épaules)
    et d'obtenir sa version inversée gratuitement."""

    def __init__(self, df: pd.DataFrame, s: int):
        self.s = s
        if s == 1:
            self.hi, self.lo = df["High"].values, df["Low"].values
        else:
            self.hi, self.lo = -df["Low"].values, -df["High"].values
        self.close = s * df["Close"].values
        self.n = len(df)

    def piv(self, pivots):
        flip = {"H": "L", "L": "H"}
        return [(i, self.s * p, k if self.s == 1 else flip[k]) for i, p, k in pivots]

    def real(self, p):
        return None if p is None else float(self.s * p)


def line_through(i1, p1, i2, p2):
    slope = (p2 - p1) / (i2 - i1) if i2 != i1 else 0.0
    return lambda i: p1 + slope * (i - i1), slope


def fit_line(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 2:
        return None, None
    if len(xs) == 2:
        f, s = line_through(xs[0], ys[0], xs[1], ys[1])
        return f, s
    s, b = np.polyfit(xs, ys, 1)
    return (lambda i: s * i + b), s


def evaluate(v: View, start, line, target, invalid, d, max_wait):
    """Suivi après la fin de la figure (espace transformé).
    d=+1 : cassure attendue vers le haut, d=-1 : vers le bas."""
    brk = None
    for i in range(start, v.n):
        c = v.close[i]
        if brk is None:
            if d * (c - invalid) < 0:
                return "invalidée", None, i
            if d * (c - line(i)) > 0:
                brk = i
                continue
            if i - start > max_wait:
                return "expirée", None, i
        else:
            reached = v.hi[i] >= target if d == 1 else v.lo[i] <= target
            if reached:
                return "objectif atteint", brk, i
            if d * (c - invalid) < 0:
                return "échec après cassure", brk, i
    return ("confirmée" if brk is not None else "en formation"), brk, v.n - 1


def volume_ok(df, i, n=20, k=1.2):
    if "Volume" not in df or i is None or i < n:
        return None
    vol = df["Volume"].values
    avg = vol[i - n:i].mean()
    if avg <= 0:
        return None
    return bool(vol[i] >= k * avg)


def _pattern(kind, bias, status, start, end, points, lines, level, target,
             invalid, score, brk, note="", end_eval=None):
    return dict(
        figure=kind, biais=bias, statut=status, debut=int(start), fin=int(end),
        points=points, lignes=lines, niveau_cassure=level, objectif=target,
        invalidation=invalid, score=int(max(0, min(100, round(score)))),
        cassure=None if brk is None else int(brk), note=note,
        fin_suivi=None if end_eval is None else int(end_eval),
    )


# ---------------------------------------------------------------------------
# Tête-épaules / tête-épaules inversée
# ---------------------------------------------------------------------------

def detect_head_shoulders(df, pivots, A, s):
    v = View(df, s)
    P = v.piv(pivots)
    out = []
    name = "Tête-épaules" if s == 1 else "Tête-épaules inversée"
    bias = BAISSIER if s == 1 else HAUSSIER
    for j in range(len(P) - 4):
        LS, T1, HD, T2, RS = P[j:j + 5]
        if [x[2] for x in (LS, T1, HD, T2, RS)] != ["H", "L", "H", "L", "H"]:
            continue
        if not (HD[1] > LS[1] and HD[1] > RS[1]):
            continue
        nl, slope = line_through(T1[0], T1[1], T2[0], T2[1])
        height = HD[1] - nl(HD[0])
        if height <= 0:
            continue
        if LS[1] - nl(LS[0]) < 0.25 * height or RS[1] - nl(RS[0]) < 0.25 * height:
            continue
        if HD[1] - max(LS[1], RS[1]) < 0.12 * height:
            continue
        asym = abs(LS[1] - RS[1]) / height
        if asym > 0.5:
            continue
        neck_tilt = abs(slope * (T2[0] - T1[0])) / height
        if neck_tilt > 0.5:
            continue
        d1, d2 = HD[0] - LS[0], RS[0] - HD[0]
        ratio = max(d1, d2) / max(1, min(d1, d2))
        if ratio > 2.5:
            continue
        score = 100 - 60 * asym - 12 * (ratio - 1) - 30 * neck_tilt
        prior = P[j - 1] if j >= 1 else None
        if prior is not None and prior[1] < min(T1[1], T2[1]):
            score += 5   # tendance préalable cohérente
        else:
            score -= 15
        invalid = RS[1]
        width = RS[0] - LS[0]
        # objectif = cou au moment de la cassure - hauteur (calculé après)
        status, brk, end_e = evaluate(
            v, RS[0] + 1, nl, nl(min(v.n - 1, RS[0] + 1)) - height,
            invalid, -1, width)
        ref = brk if brk is not None else v.n - 1
        target = nl(ref) - height
        if brk is not None:
            # réévaluer avec l'objectif exact au point de cassure
            status, brk, end_e = evaluate(v, RS[0] + 1, nl, target, invalid, -1, width)
            vk = volume_ok(df, brk)
            if vk:
                score += 5
        x_end = brk + 2 if brk is not None else (end_e if status != "en formation" else v.n - 1)
        pts = [(LS[0], v.real(LS[1]), "Épaule G"), (T1[0], v.real(T1[1]), ""),
               (HD[0], v.real(HD[1]), "Tête"), (T2[0], v.real(T2[1]), ""),
               (RS[0], v.real(RS[1]), "Épaule D")]
        lines = [((T1[0], v.real(nl(T1[0]))), (x_end, v.real(nl(x_end))), "Ligne de cou")]
        out.append(_pattern(name, bias, status, LS[0], RS[0], pts, lines,
                            v.real(nl(ref)), v.real(target), v.real(invalid),
                            score, brk, end_eval=end_e))
    return out


# ---------------------------------------------------------------------------
# Double sommet / double creux
# ---------------------------------------------------------------------------

def detect_double(df, pivots, A, s, tol=0.2, min_bars=5):
    v = View(df, s)
    P = v.piv(pivots)
    out = []
    name = "Double sommet" if s == 1 else "Double creux"
    bias = BAISSIER if s == 1 else HAUSSIER
    for j in range(1, len(P) - 2):
        P0, H1, T, H2 = P[j - 1], P[j], P[j + 1], P[j + 2]
        if [x[2] for x in (P0, H1, T, H2)] != ["L", "H", "L", "H"]:
            continue
        top = max(H1[1], H2[1])
        height = top - T[1]
        if height <= 0 or H2[0] - H1[0] < min_bars:
            continue
        diff = abs(H1[1] - H2[1]) / height
        if diff > tol:
            continue
        if P0[1] >= T[1]:      # il faut une tendance préalable
            continue
        neck = T[1]
        line = lambda i, neck=neck: neck
        target = neck - height
        invalid = top
        status, brk, end_e = evaluate(v, H2[0] + 1, line, target, invalid, -1,
                                      H2[0] - H1[0])
        score = 90 - 150 * diff
        if brk is not None and volume_ok(df, brk):
            score += 5
        x_end = brk + 2 if brk is not None else (end_e if status != "en formation" else v.n - 1)
        pts = [(H1[0], v.real(H1[1]), "Sommet 1" if s == 1 else "Creux 1"),
               (T[0], v.real(T[1]), ""),
               (H2[0], v.real(H2[1]), "Sommet 2" if s == 1 else "Creux 2")]
        lines = [((T[0], v.real(neck)), (x_end, v.real(neck)), "Ligne de cou")]
        out.append(_pattern(name, bias, status, H1[0], H2[0], pts, lines,
                            v.real(neck), v.real(target), v.real(invalid),
                            score, brk, end_eval=end_e))
    return out


# ---------------------------------------------------------------------------
# Triangles (symétrique, ascendant, descendant)
# ---------------------------------------------------------------------------

def detect_triangles(df, pivots, A):
    C, Hh, Ll = df["Close"].values, df["High"].values, df["Low"].values
    n = len(df)
    out = []
    for m in (5, 6, 7):
        for j in range(len(pivots) - m + 1):
            W = pivots[j:j + m]
            hs = [(i, p) for i, p, k in W if k == "H"]
            ls = [(i, p) for i, p, k in W if k == "L"]
            if len(hs) < 2 or len(ls) < 2:
                continue
            up, su = fit_line([x[0] for x in hs], [x[1] for x in hs])
            lo, sl = fit_line([x[0] for x in ls], [x[1] for x in ls])
            s0, e0 = W[0][0], W[-1][0]
            ws, we = up(s0) - lo(s0), up(e0) - lo(e0)
            if ws <= 0 or we <= 0 or we > 0.75 * ws:
                continue
            resid = max([abs(p - up(i)) for i, p in hs] + [abs(p - lo(i)) for i, p in ls])
            if resid > 0.2 * ws:
                continue
            du, dl = up(e0) - up(s0), lo(e0) - lo(s0)
            flat = 0.15 * ws
            if abs(du) <= flat and dl > flat:
                kind, bias = "Triangle ascendant", HAUSSIER
            elif abs(dl) <= flat and du < -flat:
                kind, bias = "Triangle descendant", BAISSIER
            elif du < -flat and dl > flat:
                kind = "Triangle symétrique"
                back = max(0, s0 - (e0 - s0))
                bias = HAUSSIER if C[s0] > C[back] else BAISSIER
            else:
                continue
            # apex
            denom = su - sl
            apex = s0 + ws / (-denom) if denom < 0 else e0 + 3 * (e0 - s0)
            max_wait = int(max(5, min(apex - e0, e0 - s0)))
            # suivi de cassure dans les deux sens
            status, brk, d, end_e = "en formation", None, 0, n - 1
            for i in range(e0 + 1, n):
                if brk is None:
                    if C[i] > up(i):
                        brk, d = i, 1
                    elif C[i] < lo(i):
                        brk, d = i, -1
                    elif i - e0 > max_wait:
                        status, end_e = "expirée", i
                        break
                else:
                    tgt = up(brk) + ws if d == 1 else lo(brk) - ws
                    inv = ls[-1][1] if d == 1 else hs[-1][1]
                    if (d == 1 and Hh[i] >= tgt) or (d == -1 and Ll[i] <= tgt):
                        status, end_e = "objectif atteint", i
                        break
                    if (d == 1 and C[i] < inv) or (d == -1 and C[i] > inv):
                        status, end_e = "échec après cassure", i
                        break
            if brk is not None:
                if status == "en formation":
                    status = "confirmée"
                bias_eff = HAUSSIER if d == 1 else BAISSIER
                if bias_eff != bias:
                    kind += " (cassure contraire)"
                bias = bias_eff
            if bias == HAUSSIER:
                ref = brk if brk is not None else n - 1
                level, target, invalid = up(ref), up(ref) + ws, ls[-1][1]
            else:
                ref = brk if brk is not None else n - 1
                level, target, invalid = lo(ref), lo(ref) - ws, hs[-1][1]
            score = 85 - 100 * resid / ws + 5 * (m - 5)
            if brk is not None and volume_ok(df, brk):
                score += 5
            x_end = brk + 2 if brk is not None else (end_e if status != "en formation" else n - 1)
            lines = [((s0, float(up(s0))), (x_end, float(up(x_end))), "Résistance"),
                     ((s0, float(lo(s0))), (x_end, float(lo(x_end))), "Support")]
            pts = [(i, float(p), "") for i, p, k in W]
            out.append(_pattern(kind, bias, status, s0, e0, pts, lines,
                                float(level), float(target), float(invalid),
                                score, brk, end_eval=end_e))
    return _dedupe(out)


# ---------------------------------------------------------------------------
# Drapeaux et fanions
# ---------------------------------------------------------------------------

def detect_flags(df, pivots, A, s, pole_atr=3.0, max_pole_bars=25,
                 min_cons=5, max_cons=40):
    v = View(df, s)
    P = v.piv(pivots)
    out = []
    bias = HAUSSIER if s == 1 else BAISSIER
    for j in range(len(P) - 1):
        (a, pa, ka), (b, pb, kb) = P[j], P[j + 1]
        if (ka, kb) != ("L", "H"):
            continue
        pole = pb - pa
        bars = b - a
        if bars < 2 or bars > max_pole_bars or pole < pole_atr * A[b]:
            continue
        if pole / bars < 0.3 * A[b]:
            continue
        limit = min(v.n - 1, b + max(max_cons, 2 * bars))
        status, brk, res, end_e = None, None, None, None
        for i in range(b + 1, limit + 1):
            # échec : retracement > 50 % du mât ou nouveau plus haut avant drapeau
            if v.lo[i] < pb - 0.5 * pole:
                status, end_e = "invalidée", i
                break
            L = i - (b + 1)
            if L >= min_cons:
                xs = np.arange(b + 1, i)
                up, su = fit_line(xs, v.hi[b + 1:i])
                lo, sl = fit_line(xs, v.lo[b + 1:i])
                if v.close[i] > up(i) + 0.25 * A[i] and v.close[i] > v.hi[b + 1:i].max() - 0.25 * (v.hi[b + 1:i].max() - v.lo[b + 1:i].min()):
                    brk, res, end_e = i, (up, lo, su, sl, xs), i
                    break
            if v.hi[i] > pb and L < min_cons:
                break   # le mât continue : ce n'est pas encore une consolidation
        else:
            i = limit
            if limit - b - 1 >= min_cons:
                xs = np.arange(b + 1, limit + 1)
                up, su = fit_line(xs, v.hi[b + 1:limit + 1])
                lo, sl = fit_line(xs, v.lo[b + 1:limit + 1])
                res = (up, lo, su, sl, xs)
                status = "en formation" if limit == v.n - 1 and limit - b < max(max_cons, 2 * bars) else "expirée"
                end_e = limit
        if res is None or status == "invalidée":
            continue
        up, lo, su, sl, xs = res
        Lc = len(xs)
        du, dl = su * Lc, sl * Lc
        chan = float(np.mean(v.hi[xs] - v.lo[xs]))
        if chan > 0.5 * pole:
            continue
        if du < -0.05 * pole and dl > 0.05 * pole:
            kind = "Fanion"
        elif du <= 0.1 * pole and dl <= 0.1 * pole and abs(du - dl) <= 0.35 * pole:
            kind = "Drapeau"
        else:
            continue
        kind += " haussier" if s == 1 else " baissier"
        cons_low = float(v.lo[xs].min())
        ref = brk if brk is not None else v.n - 1
        level = up(ref)
        target = level + pole
        if brk is not None:
            st, _, end_e = evaluate(v, brk + 1, lambda i: -np.inf, target, cons_low, 1, 10**9)
            status = "confirmée" if st == "en formation" else st
        score = 80 - 40 * abs(du - dl) / pole - 30 * chan / pole + min(15, 3 * pole / A[b])
        if brk is not None and volume_ok(df, brk):
            score += 5
        x_end = brk + 2 if brk is not None else (end_e if status != "en formation" else v.n - 1)
        x0 = xs[0]
        pts = [(a, v.real(pa), "Début mât"), (b, v.real(pb), "Haut mât" if s == 1 else "Bas mât")]
        lines = [((a, v.real(pa)), (b, v.real(pb)), "Mât"),
                 ((x0, v.real(up(x0))), (x_end, v.real(up(x_end))), "Canal"),
                 ((x0, v.real(lo(x0))), (x_end, v.real(lo(x_end))), "Canal")]
        out.append(_pattern(kind, bias, status, a, xs[-1], pts, lines,
                            v.real(level), v.real(target), v.real(cons_low),
                            score, brk, end_eval=end_e))
    return out


# ---------------------------------------------------------------------------
# Fair Value Gaps (imbalances)
# ---------------------------------------------------------------------------

def detect_fvg(df, A, min_atr=0.3):
    """FVG sur 3 bougies. Haussier : bas(i) > haut(i-2). Baissier : haut(i) < bas(i-2).
    Renvoie toutes les zones avec leur taux de comblement."""
    H, L, Av = df["High"].values, df["Low"].values, A.values
    n = len(df)
    out = []
    for i in range(2, n):
        if L[i] > H[i - 2] and L[i] - H[i - 2] >= min_atr * Av[i]:
            bot, top, bias = H[i - 2], L[i], HAUSSIER
        elif H[i] < L[i - 2] and L[i - 2] - H[i] >= min_atr * Av[i]:
            bot, top, bias = H[i], L[i - 2], BAISSIER
        else:
            continue
        size = top - bot
        filled, fill_idx = 0.0, None
        if i + 1 < n:
            if bias == HAUSSIER:
                m = np.minimum.accumulate(L[i + 1:])
                filled = float(np.clip((top - m[-1]) / size, 0, 1))
                full = np.where(m <= bot)[0]
            else:
                m = np.maximum.accumulate(H[i + 1:])
                filled = float(np.clip((m[-1] - bot) / size, 0, 1))
                full = np.where(m >= top)[0]
            if len(full):
                fill_idx = int(i + 1 + full[0])
        out.append(dict(idx=i - 1, bas=float(bot), haut=float(top), biais=bias,
                        comblement=filled, comble_idx=fill_idx,
                        actif=fill_idx is None))
    return out


# ---------------------------------------------------------------------------
# Supports / résistances horizontaux, canaux, obliques
# ---------------------------------------------------------------------------

def supports_resistances(df, pivots, A, fenetre=400, tol_atr=0.6, min_touches=2, par_cote=3):
    """Regroupe les pivots proches en zones. Une zone touchée au moins 2 fois
    = support (sous le cours) ou résistance (au-dessus)."""
    n = len(df)
    last = float(df["Close"].iloc[-1])
    tol = tol_atr * float(A[-1])
    pts = sorted([(p, i) for i, p, k in pivots if i >= n - fenetre])
    zones, cur = [], []
    for p, i in pts:
        if cur and p - np.mean([c[0] for c in cur]) > tol:
            zones.append(cur)
            cur = []
        cur.append((p, i))
    if cur:
        zones.append(cur)
    out = []
    for z in zones:
        if len(z) < min_touches:
            continue
        prix = [c[0] for c in z]
        niveau = float(np.mean(prix))
        out.append(dict(niveau=niveau, bas=float(min(prix)), haut=float(max(prix)),
                        contacts=len(z), debut=int(min(c[1] for c in z)),
                        dernier=int(max(c[1] for c in z)),
                        type="résistance" if niveau > last else "support"))
    sup = sorted([z for z in out if z["type"] == "support"], key=lambda z: -z["niveau"])[:par_cote]
    res = sorted([z for z in out if z["type"] == "résistance"], key=lambda z: z["niveau"])[:par_cote]
    return sup + res


def detect_canal(df, pivots, A):
    """Canal en cours : lignes parallèles passant par les derniers sommets et creux.
    Cherche la plus longue série récente de pivots compatible."""
    C = df["Close"].values
    n = len(df)
    best = None
    for m in range(8, 3, -1):
        for fin in range(len(pivots), max(len(pivots) - 4, 0), -1):
            W = pivots[max(0, fin - m):fin]
            if len(W) < m:
                continue
            hs = [(i, p) for i, p, k in W if k == "H"]
            ls = [(i, p) for i, p, k in W if k == "L"]
            if len(hs) < 2 or len(ls) < 2:
                continue
            _, su = fit_line([x[0] for x in hs], [x[1] for x in hs])
            _, sl = fit_line([x[0] for x in ls], [x[1] for x in ls])
            s = (su + sl) / 2
            b_up = float(np.mean([p - s * i for i, p in hs]))
            b_lo = float(np.mean([p - s * i for i, p in ls]))
            w = b_up - b_lo
            s0, e0 = W[0][0], W[-1][0]
            span = e0 - s0
            if w < 1.5 * A[-1] or span < 10:
                continue
            if abs(su - sl) * span > 0.3 * w:
                continue
            resid = max([abs(p - (s * i + b_up)) for i, p in hs] +
                        [abs(p - (s * i + b_lo)) for i, p in ls])
            if resid > 0.25 * w:
                continue
            up_last, lo_last = s * (n - 1) + b_up, s * (n - 1) + b_lo
            if C[-1] > up_last + 0.5 * w:
                etat = "cassé par le haut"
            elif C[-1] < lo_last - 0.5 * w:
                etat = "cassé par le bas"
            else:
                etat = "en cours"
            pente = s * span
            if pente > 0.5 * w:
                nom, biais = "Canal haussier", HAUSSIER
            elif pente < -0.5 * w:
                nom, biais = "Canal baissier", BAISSIER
            else:
                nom, biais = "Canal horizontal (range)", None
            score = 100 - 200 * resid / w + 3 * m
            cand = dict(nom=nom, biais=biais, etat=etat, debut=int(s0), pente=float(s),
                        b_haut=b_up, b_bas=b_lo, largeur=float(w), score=score,
                        haut_actuel=float(up_last), bas_actuel=float(lo_last))
            if best is None or m > best["_m"] or (m == best["_m"] and score > best["score"]):
                cand["_m"] = m
                best = cand
        if best is not None:
            break
    if best:
        best.pop("_m", None)
    return best


def detect_obliques(df, pivots, A, nb_pivots=14, tol_atr=0.35):
    """Oblique haussière : relie deux creux montants non cassés depuis.
    Oblique baissière : relie deux sommets descendants non cassés depuis."""
    C = df["Close"].values
    n = len(df)
    tol = tol_atr * float(A[-1])
    piv = pivots[-nb_pivots:]
    out = []
    for kind, nom, biais in (("L", "Oblique haussière", HAUSSIER), ("H", "Oblique baissière", BAISSIER)):
        pts = [(i, p) for i, p, k in piv if k == kind]
        best = None
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                (i1, p1), (i2, p2) = pts[a], pts[b]
                if (kind == "L" and p2 <= p1) or (kind == "H" and p2 >= p1) or i2 - i1 < 5:
                    continue
                f, _ = line_through(i1, p1, i2, p2)
                xs = np.arange(i1, n)
                ligne = f(xs)
                # aucune clôture nettement de l'autre côté entre le 1er point et le 2e
                seg = slice(0, i2 - i1)
                if kind == "L":
                    if np.any(C[i1:i2] < ligne[seg] - tol):
                        continue
                    casse = bool(C[-1] < f(n - 1) - tol)
                else:
                    if np.any(C[i1:i2] > ligne[seg] + tol):
                        continue
                    casse = bool(C[-1] > f(n - 1) + tol)
                contacts = sum(1 for i, p in pts if i >= i1 and abs(p - f(i)) <= tol)
                cand = dict(nom=nom, biais=biais, x1=int(i1), y1=float(p1), x2=int(i2), y2=float(p2),
                            pente=float((p2 - p1) / (i2 - i1)), contacts=contacts,
                            etat="cassée" if casse else "respectée",
                            niveau_actuel=float(f(n - 1)))
                key = (not casse, contacts, i2)
                if best is None or key > best[0]:
                    best = (key, cand)
        if best:
            out.append(best[1])
    return out


# ---------------------------------------------------------------------------
# Dédoublonnage, contexte de tendance, synthèse
# ---------------------------------------------------------------------------

def _overlap(a, b):
    lo, hi = max(a["debut"], b["debut"]), min(a["fin"], b["fin"])
    if hi <= lo:
        return 0.0
    return (hi - lo) / max(1, min(a["fin"] - a["debut"], b["fin"] - b["debut"]))


def _dedupe(pats, thr=0.5):
    pats = sorted(pats, key=lambda p: (-p["score"], -(p["fin"] - p["debut"])))
    keep = []
    for p in pats:
        family = p["figure"].split(" ")[0]
        if all(not (k["figure"].split(" ")[0] == family and _overlap(p, k) > thr) for k in keep):
            keep.append(p)
    return keep


def trend_context(df):
    c = df["Close"]
    e50 = c.ewm(span=50, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    last = float(c.iloc[-1])
    slope50 = float(e50.iloc[-1] - e50.iloc[-11]) if len(c) > 11 else 0.0
    vote, why = 0.0, []
    if len(c) >= 200:
        if last > e200.iloc[-1] and e50.iloc[-1] > e200.iloc[-1]:
            vote, why = 1.0, ["Cours au-dessus des moyennes 50 et 200, MM50 > MM200"]
        elif last < e200.iloc[-1] and e50.iloc[-1] < e200.iloc[-1]:
            vote, why = -1.0, ["Cours sous les moyennes 50 et 200, MM50 < MM200"]
        else:
            why = ["Moyennes 50/200 sans alignement net"]
    if slope50 > 0:
        why.append("MM50 orientée à la hausse")
        vote += 0.25
    elif slope50 < 0:
        why.append("MM50 orientée à la baisse")
        vote -= 0.25
    return dict(ema50=e50, ema200=e200, vote=vote, raisons=why)


def analyze(df: pd.DataFrame, sensibilite: float = 2.0, recent_bars: int = 120,
            fvg_min_atr: float = 0.3, figures=None):
    """Point d'entrée. `sensibilite` = multiple d'ATR pour valider un pivot
    (plus petit = plus de figures, plus de bruit)."""
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    figures = figures or {"ete", "double", "triangle", "drapeau", "fvg"}
    A = atr(df)
    piv, prov = zigzag(df, A, sensibilite)
    piv_all = piv + ([prov] if prov else [])
    A_v = A.values
    pats = []
    if "ete" in figures:
        pats += detect_head_shoulders(df, piv_all, A_v, 1)
        pats += detect_head_shoulders(df, piv_all, A_v, -1)
    if "double" in figures:
        pats += detect_double(df, piv_all, A_v, 1)
        pats += detect_double(df, piv_all, A_v, -1)
    if "triangle" in figures:
        pats += detect_triangles(df, piv_all, A_v)
    if "drapeau" in figures:
        pats += detect_flags(df, piv_all, A_v, 1)
        pats += detect_flags(df, piv_all, A_v, -1)
    fvgs = detect_fvg(df, A, fvg_min_atr) if "fvg" in figures else []
    niveaux = supports_resistances(df, piv_all, A_v)
    canal = detect_canal(df, piv_all, A_v)
    obliques = detect_obliques(df, piv_all, A_v)

    n = len(df)
    for p in pats:
        p["recente"] = (n - 1 - p["fin"]) <= recent_bars or (
            p["cassure"] is not None and n - 1 - p["cassure"] <= recent_bars)
        p["active"] = p["statut"] in ("en formation", "confirmée") and p["recente"]
    pats.sort(key=lambda p: (not p["active"], -p["fin"]))

    # Synthèse
    ctx = trend_context(df)
    total, raisons = ctx["vote"], list(ctx["raisons"])
    for p in pats:
        if not p["active"]:
            continue
        w = p["score"] / 100 * (1.0 if p["statut"] == "confirmée" else 0.5)
        sgn = 1 if p["biais"] == HAUSSIER else -1
        total += sgn * w
        raisons.append(f"{p['figure']} ({p['statut']}, score {p['score']}) → {p['biais']}")
    last = float(df["Close"].iloc[-1])
    if canal and canal["biais"] and canal["etat"] == "en cours":
        total += 0.5 if canal["biais"] == HAUSSIER else -0.5
        raisons.append(f"{canal['nom']} en cours")
    for o in obliques:
        if o["etat"] == "respectée" and o["contacts"] >= 2:
            total += 0.25 if o["biais"] == HAUSSIER else -0.25
            raisons.append(f"{o['nom']} respectée ({o['contacts']} contacts)")
        elif o["etat"] == "cassée":
            total += -0.25 if o["biais"] == HAUSSIER else 0.25
            raisons.append(f"{o['nom']} cassée")
    act = [f for f in fvgs if f["actif"]]
    sup = [f for f in act if f["haut"] < last]
    res = [f for f in act if f["bas"] > last]
    if sup:
        f = max(sup, key=lambda f: f["haut"])
        raisons.append(f"Imbalance non comblée sous le cours : {f['bas']:.2f}–{f['haut']:.2f} (support potentiel)")
    if res:
        f = min(res, key=lambda f: f["bas"])
        raisons.append(f"Imbalance non comblée au-dessus : {f['bas']:.2f}–{f['haut']:.2f} (résistance / aimant potentiel)")
    if total > 0.5:
        verdict = "Plutôt haussier"
    elif total < -0.5:
        verdict = "Plutôt baissier"
    else:
        verdict = "Neutre / indécis"
    return dict(
        figures=pats, fvg=fvgs, pivots=piv, pivot_provisoire=prov,
        niveaux=niveaux, canal=canal, obliques=obliques,
        ema50=ctx["ema50"], ema200=ctx["ema200"], atr=A,
        synthese=dict(verdict=verdict, score=round(total, 2), raisons=raisons),
        data=df,
    )
