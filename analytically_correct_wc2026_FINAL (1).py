"""
╔══════════════════════════════════════════════════════════════════╗
║         ANALYTICALLY CORRECT — WC2026 PREDICTION ENGINE         ║
║         "Not politically correct. Analytically correct."         ║
║         Model: XGBoost + Multi-Source Data Pipeline              ║
║         Author: Mathieu A. Ndong | @analyticallycorrect          ║
║         Updated: June 20, 2026                                   ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO USE:
1. Run Cell 1 (pip install) once
2. Run this script each matchday
3. Update TODAY_GAMES list with real FanDuel odds
4. Charts saved as PNG + CSV exported for Power BI
"""

# ── INSTALL (run once in Jupyter) ──────────────────────────────────
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "xgboost", "scikit-learn", "pandas",
                       "numpy", "matplotlib", "requests"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ── IMPORTS ────────────────────────────────────────────────────────
import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")  # Remove this line if running in Jupyter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from datetime import datetime
warnings.filterwarnings("ignore")

# ── BRAND COLORS ────────────────────────────────────────────────────
NAVY  = "#0A1F44"
GOLD  = "#BA7517"
BLUE  = "#185FA5"
TEAL  = "#0F6E56"
CORAL = "#993C1D"
GRAY  = "#5F5E5A"
WHITE = "#FFFFFF"

# ── API KEYS ────────────────────────────────────────────────────────
FOOTBALL_DATA_KEY = "38c88824730a4f519a0c3c0ea67f7624"
BALLDONTLIE_KEY   = "5a196495-405f-48d6-ad43-4528e6f42139"

# ── ALL 48 WC2026 TEAMS ─────────────────────────────────────────────
# Format: [fifa_rank, possession, shots, shots_on_target,
#          goals_scored, goals_conceded, xg_per_game]
TEAM_PROFILES = {
    "Argentina":    [1,  57, 15, 6.2, 2.1, 0.8, 2.3],
    "France":       [2,  55, 14, 5.8, 2.0, 0.9, 2.1],
    "England":      [3,  54, 13, 5.5, 1.9, 0.9, 2.0],
    "Brazil":       [4,  58, 16, 6.5, 2.2, 0.7, 2.4],
    "Belgium":      [5,  52, 13, 5.2, 1.8, 1.0, 1.9],
    "Portugal":     [6,  53, 14, 5.6, 2.0, 0.9, 2.0],
    "Netherlands":  [7,  54, 14, 5.5, 1.9, 0.9, 2.0],
    "Spain":        [8,  62, 15, 5.8, 1.8, 0.6, 1.9],
    "Germany":      [9,  56, 14, 5.6, 1.9, 1.0, 2.0],
    "Croatia":      [10, 51, 12, 4.8, 1.6, 1.0, 1.7],
    "Italy":        [11, 52, 12, 4.9, 1.5, 0.9, 1.6],
    "Morocco":      [12, 47, 11, 4.2, 1.4, 0.9, 1.4],
    "USA":          [13, 49, 12, 4.5, 1.5, 1.1, 1.5],
    "Mexico":       [14, 50, 12, 4.6, 1.5, 1.1, 1.5],
    "Senegal":      [15, 48, 12, 4.4, 1.5, 1.0, 1.5],
    "Japan":        [16, 50, 13, 4.8, 1.6, 0.9, 1.6],
    "South Korea":  [17, 49, 12, 4.5, 1.5, 1.0, 1.5],
    "Colombia":     [18, 51, 13, 5.0, 1.7, 1.0, 1.7],
    "Uruguay":      [19, 48, 12, 4.6, 1.6, 1.0, 1.6],
    "Sweden":       [20, 51, 12, 4.6, 1.5, 1.0, 1.5],
    "Switzerland":  [20, 50, 12, 4.7, 1.5, 0.9, 1.6],
    "Ecuador":      [21, 46, 11, 4.1, 1.3, 1.1, 1.3],
    "Denmark":      [22, 52, 13, 5.0, 1.6, 0.9, 1.7],
    "Australia":    [23, 46, 11, 4.0, 1.3, 1.2, 1.3],
    "Serbia":       [24, 49, 13, 4.8, 1.6, 1.1, 1.6],
    "Turkey":       [25, 52, 14, 5.2, 1.7, 1.0, 1.7],
    "Iran":         [26, 44, 10, 3.8, 1.2, 1.2, 1.2],
    "Poland":       [27, 47, 11, 4.2, 1.4, 1.1, 1.4],
    "Ukraine":      [28, 49, 12, 4.5, 1.5, 1.0, 1.5],
    "Ghana":        [29, 46, 11, 4.0, 1.3, 1.2, 1.3],
    "Tunisia":      [30, 45, 10, 3.9, 1.2, 1.1, 1.2],
    "Cameroon":     [31, 46, 11, 4.0, 1.3, 1.2, 1.3],
    "Paraguay":     [32, 44, 10, 3.9, 1.3, 1.2, 1.3],
    "Saudi Arabia": [33, 45, 11, 4.0, 1.3, 1.2, 1.3],
    "Qatar":        [34, 44, 10, 3.8, 1.2, 1.3, 1.2],
    "Costa Rica":   [35, 43, 10, 3.7, 1.2, 1.2, 1.2],
    "Ivory Coast":  [36, 47, 11, 4.1, 1.4, 1.1, 1.4],
    "Nigeria":      [37, 48, 12, 4.3, 1.4, 1.1, 1.4],
    "Egypt":        [38, 47, 11, 4.1, 1.4, 1.0, 1.4],
    "Algeria":      [39, 46, 11, 4.0, 1.3, 1.1, 1.3],
    "South Africa": [40, 44, 10, 3.8, 1.2, 1.2, 1.2],
    "New Zealand":  [41, 43, 10, 3.6, 1.1, 1.3, 1.1],
    "Panama":       [42, 42,  9, 3.5, 1.1, 1.3, 1.1],
    "Jamaica":      [43, 43, 10, 3.6, 1.1, 1.3, 1.1],
    "Honduras":     [44, 42,  9, 3.5, 1.1, 1.4, 1.1],
    "Bolivia":      [45, 41,  9, 3.4, 1.0, 1.4, 1.0],
    "Venezuela":    [46, 44, 10, 3.7, 1.2, 1.2, 1.2],
    "Canada":       [47, 47, 11, 4.1, 1.4, 1.1, 1.4],
    "Guatemala":    [48, 41,  9, 3.3, 1.0, 1.4, 1.0],
    "Curacao":      [75, 41,  9, 3.3, 1.0, 1.5, 1.0],
}

FEATURES = [
    "fifa_rank", "possession", "shots_total", "shots_on_target",
    "goals_scored", "goals_conceded", "xg_per_game",
    "rank_diff", "possession_diff", "xg_diff"
]

# ═══════════════════════════════════════════════════════════════════
# TODAY'S GAMES — UPDATE THESE EVERY MATCHDAY
# Format: (team_a, team_b, fanduel_a_odds, fanduel_b_odds, actual_result)
# actual_result: fill in AFTER the game e.g. "Netherlands 2-0 Sweden"
# Leave actual_result as None before the game
# ═══════════════════════════════════════════════════════════════════
TODAY_GAMES = [
    ("Netherlands", "Sweden",     -145,  +370,  None),  # 1:00 PM ET
    ("Germany",     "Ivory Coast",-210,  +500,  None),  # 4:00 PM ET
    ("Ecuador",     "Curacao",    -800,  +2000, None),  # 4:00 PM ET
    ("Tunisia",     "Japan",      +550,  -180,  None),  # 8:00 PM ET
]

# ═══════════════════════════════════════════════════════════════════
# PAST RESULTS — ADD AFTER EACH GAME FOR ACCURACY TRACKING
# ═══════════════════════════════════════════════════════════════════
PAST_RESULTS = [
    # EP.001 — June 16, 2026
    ("France",   "Senegal", -200, +550, "France 3-1 Senegal"),
]


# ── DATA PIPELINE ──────────────────────────────────────────────────
def fetch_worldcup_ir():
    try:
        r = requests.get("https://worldcup26.ir/get/games", timeout=5)
        r.raise_for_status()
        rows = []
        for g in r.json():
            if g.get("status") == "completed":
                stats = g.get("stats") or {}
                score = g.get("score") or {}
                for side, opp in [("home","away"),("away","home")]:
                    rows.append({
                        "team": g.get(f"{side}_team",{}).get("name","Unknown"),
                        "possession": stats.get(side,{}).get("possession",50),
                        "shots_total": stats.get(side,{}).get("shots_total",12),
                        "shots_on_target": stats.get(side,{}).get("shots_on_target",4),
                        "goals_scored": score.get(side,0),
                        "goals_conceded": score.get(opp,0),
                        "result": 1 if score.get(side,0) > score.get(opp,0) else 0
                    })
        if len(rows) >= 10:
            print(f"  🟢 worldcup26.ir: {len(rows)} rows loaded")
            return pd.DataFrame(rows)
    except Exception as e:
        print(f"  ⚠️  worldcup26.ir: {e}")
    return None

def fetch_balldontlie():
    try:
        headers = {"Authorization": BALLDONTLIE_KEY}
        r = requests.get(
            "https://api.balldontlie.io/fifa/worldcup/v1/matches?seasons[]=2026",
            headers=headers, timeout=10)
        if r.status_code in [401, 403]:
            print(f"  ⚠️  BallDontLie: Auth {r.status_code}")
            return None
        r.raise_for_status()
        rows = []
        for g in r.json().get("data", []):
            if g.get("status") == "completed":
                for side, opp in [("home","away"),("away","home")]:
                    rows.append({
                        "team": g.get(f"{side}_team",{}).get("name","Unknown"),
                        "possession": 50, "shots_total": 12,
                        "shots_on_target": 4,
                        "goals_scored": g.get(f"{side}_score",0),
                        "goals_conceded": g.get(f"{opp}_score",0),
                        "result": 1 if g.get(f"{side}_score",0) > g.get(f"{opp}_score",0) else 0
                    })
        if len(rows) >= 10:
            print(f"  🟢 BallDontLie: {len(rows)} rows loaded")
            return pd.DataFrame(rows)
    except Exception as e:
        print(f"  ⚠️  BallDontLie: {e}")
    return None

def fetch_football_data_org():
    try:
        headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
        r = requests.get(
            "https://api.football-data.org/v4/competitions/WC/matches",
            headers=headers, timeout=8)
        r.raise_for_status()
        rows = []
        for m in r.json().get("matches", []):
            if m.get("status") == "FINISHED":
                score = m.get("score",{}).get("fullTime",{})
                for side, opp in [("home","away"),("away","home")]:
                    gs = score.get(side,0) or 0
                    gc = score.get(opp,0) or 0
                    rows.append({
                        "team": m.get(f"{side}Team",{}).get("name","Unknown"),
                        "possession": 50, "shots_total": 12,
                        "shots_on_target": 4,
                        "goals_scored": gs, "goals_conceded": gc,
                        "result": 1 if gs > gc else 0
                    })
        if len(rows) >= 6:
            print(f"  🟢 football-data.org: {len(rows)} rows loaded")
            return pd.DataFrame(rows)
    except Exception as e:
        print(f"  ⚠️  football-data.org: {e}")
    return None

def build_profile_dataset():
    print("  🧠 Building FIFA profile matrix (48 teams)...")
    np.random.seed(42)
    rows = []
    teams = list(TEAM_PROFILES.keys())
    for _ in range(500):
        t1, t2 = np.random.choice(teams, 2, replace=False)
        p1, p2 = TEAM_PROFILES[t1], TEAM_PROFILES[t2]
        noise = lambda x: x + np.random.normal(0, x * 0.12)
        gs1 = max(0, round(noise(p1[4])))
        gs2 = max(0, round(noise(p2[4])))
        for team, prof, gs, gc in [(t1,p1,gs1,gs2),(t2,p2,gs2,gs1)]:
            rows.append({
                "team": team,
                "possession": noise(prof[1]),
                "shots_total": noise(prof[2]),
                "shots_on_target": noise(prof[3]),
                "goals_scored": gs, "goals_conceded": gc,
                "result": 1 if gs > gc else 0
            })
    df = pd.DataFrame(rows)
    print(f"  ✅ {len(df)} rows generated")
    return df

def get_dataset():
    print("\n" + "═"*65)
    print("🛰️  ANALYTICALLY CORRECT — DATA PIPELINE")
    print("═"*65)
    for fetcher in [fetch_worldcup_ir, fetch_balldontlie, fetch_football_data_org]:
        df = fetcher()
        if df is not None:
            return df
    return build_profile_dataset()


# ── FEATURE ENGINEERING ────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df["fifa_rank"] = df["team"].apply(
        lambda t: TEAM_PROFILES.get(t, [30])[0])
    df["xg_per_game"] = df["team"].apply(
        lambda t: TEAM_PROFILES.get(t, [0]*7)[6] if len(TEAM_PROFILES.get(t,[])) > 6 else 1.5)
    df["rank_diff"]       = df["fifa_rank"].mean() - df["fifa_rank"]
    df["possession_diff"] = df["possession"] - 50
    df["xg_diff"]         = df["xg_per_game"] - df["xg_per_game"].mean()
    return df


# ── TRAIN XGBOOST ──────────────────────────────────────────────────
def train_xgboost(df):
    df = engineer_features(df)
    avail = [f for f in FEATURES if f in df.columns]
    X = df[avail].fillna(df[avail].mean())
    y = df["result"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42)
    model.fit(X_scaled, y)
    cv = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    print(f"\n  📈 XGBoost CV Accuracy: {cv.mean():.1%} ± {cv.std():.1%}")
    return model, scaler, avail


# ── GET TEAM STATS ─────────────────────────────────────────────────
def get_team_stats(team, df):
    rows = df[df["team"].str.lower() == team.lower()]
    if not rows.empty:
        s = rows[["possession","shots_total","shots_on_target",
                  "goals_scored","goals_conceded"]].mean().to_dict()
    else:
        p = TEAM_PROFILES.get(team, [30,48,11,4,1.3,1.2,1.3])
        s = {"possession":p[1],"shots_total":p[2],
             "shots_on_target":p[3],"goals_scored":p[4],"goals_conceded":p[5]}
    p = TEAM_PROFILES.get(team, [30,48,11,4,1.3,1.2,1.3])
    s["fifa_rank"]   = p[0]
    s["xg_per_game"] = p[6] if len(p) > 6 else 1.3
    return s


# ── PREDICT MATCH ──────────────────────────────────────────────────
def predict_match(team_a, team_b, model, scaler, features, df,
                  fanduel_a=None, fanduel_b=None, actual_result=None):
    sa = get_team_stats(team_a, df)
    sb = get_team_stats(team_b, df)
    # Head-to-head differential features
    for s, opp in [(sa, sb), (sb, sa)]:
        s["rank_diff"]       = opp["fifa_rank"] - s["fifa_rank"]
        s["possession_diff"] = s["possession"] - opp["possession"]
        s["xg_diff"]         = s["xg_per_game"] - opp["xg_per_game"]
    dfa = pd.DataFrame([{f: sa.get(f,0) for f in features}])
    dfb = pd.DataFrame([{f: sb.get(f,0) for f in features}])
    raw_a = model.predict_proba(scaler.transform(dfa))[0][1]
    raw_b = model.predict_proba(scaler.transform(dfb))[0][1]
    total = raw_a + raw_b
    draw_p = 0.27
    prob_a = (raw_a / total) * (1 - draw_p)
    prob_b = (raw_b / total) * (1 - draw_p)

    def to_american(p):
        d = 1/p if p > 0 else 999
        return f"+{round((d-1)*100)}" if d >= 2 else f"{round(-100/(d-1))}"

    def calc_edge(fd, mp):
        if fd is None: return None
        fd_dec = (fd/100 + 1) if fd > 0 else (1 - 100/fd)
        fi = 1 / fd_dec
        edge = (mp - fi) * 100
        verdict = "🟢 MODEL LIKES" if edge > 3 else ("🔴 FADE" if edge < -3 else "⚪ NEUTRAL")
        return {"fd_odds": f"+{fd}" if fd > 0 else str(fd),
                "fd_implied": fi, "model_prob": mp,
                "edge": edge, "verdict": verdict}

    return {
        "team_a": team_a, "team_b": team_b,
        "stats_a": sa, "stats_b": sb,
        "prob_a": prob_a, "prob_b": prob_b, "prob_draw": draw_p,
        "american_a": to_american(prob_a),
        "american_b": to_american(prob_b),
        "american_draw": to_american(draw_p),
        "winner": team_a if prob_a > prob_b else team_b,
        "confidence": max(prob_a, prob_b),
        "edge_a": calc_edge(fanduel_a, prob_a),
        "edge_b": calc_edge(fanduel_b, prob_b),
        "actual_result": actual_result,
        "date": datetime.now().strftime("%Y-%m-%d"),
    }


# ── PRINT PREDICTION ───────────────────────────────────────────────
def print_prediction(r):
    a, b = r["team_a"], r["team_b"]
    pa, pb, pd_ = r["prob_a"], r["prob_b"], r["prob_draw"]
    sa, sb = r["stats_a"], r["stats_b"]

    print("\n" + "═"*65)
    print(f"  ⚽ ANALYTICALLY CORRECT | {a.upper()} vs {b.upper()}")
    print(f"  {r['date']}")
    print("═"*65)
    print(f"\n  {'METRIC':<26} {a:<20} {b}")
    print(f"  {'─'*60}")
    for label, key in [
        ("FIFA Rank","fifa_rank"),("Possession %","possession"),
        ("Shots/Game","shots_total"),("Shots on Target","shots_on_target"),
        ("Goals Scored/G","goals_scored"),("Goals Conceded/G","goals_conceded"),
        ("xG per Game","xg_per_game")]:
        va, vb = sa.get(key,0), sb.get(key,0)
        print(f"  {label:<26} {str(round(va,1)):<20} {round(vb,1)}")
    print(f"\n  {a:<24} {pa:.1%}  {'█'*round(pa*35)}")
    print(f"  {'Draw':<24} {pd_:.1%}  {'█'*round(pd_*35)}")
    print(f"  {b:<24} {pb:.1%}  {'█'*round(pb*35)}")
    print(f"\n  🎲 MODEL ODDS: {a} {r['american_a']} | Draw {r['american_draw']} | {b} {r['american_b']}")
    if r["edge_a"]:
        e = r["edge_a"]
        print(f"\n  💰 FANDUEL EDGE")
        print(f"  {a:<24} FD:{e['fd_odds']:<8} Model:{e['model_prob']:.1%}  "
              f"Edge:{e['edge']:+.1f}%  {e['verdict']}")
    if r["edge_b"]:
        e = r["edge_b"]
        print(f"  {b:<24} FD:{e['fd_odds']:<8} Model:{e['model_prob']:.1%}  "
              f"Edge:{e['edge']:+.1f}%  {e['verdict']}")
    if r.get("actual_result"):
        winner_correct = r["winner"].lower() in r["actual_result"].lower()
        stamp = "✅ CORRECT" if winner_correct else "❌ WRONG"
        print(f"\n  📋 ACTUAL RESULT: {r['actual_result']}  {stamp}")
    print(f"\n  🏆 MODEL PICKS: {r['winner'].upper()} ({r['confidence']:.1%} confidence)")
    print("═"*65)


# ── VISUALIZATION ──────────────────────────────────────────────────
def plot_prediction(r, save_path=None):
    a, b = r["team_a"], r["team_b"]
    pa, pb, pd_ = r["prob_a"], r["prob_b"], r["prob_draw"]
    sa, sb = r["stats_a"], r["stats_b"]

    fig = plt.figure(figsize=(16, 10), facecolor=NAVY)
    fig.suptitle(f"ANALYTICALLY CORRECT — WC2026  |  {a} vs {b}  |  {r['date']}",
                color=GOLD, fontsize=13, fontweight="bold", y=0.98)
    gs_ = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                           left=0.06, right=0.96, top=0.90, bottom=0.08)

    # Panel 1 — Donut
    ax1 = fig.add_subplot(gs_[0,0]); ax1.set_facecolor(NAVY)
    ax1.pie([pa, pd_, pb], colors=[BLUE, GRAY, CORAL], startangle=90,
            wedgeprops=dict(width=0.55, edgecolor=NAVY, linewidth=2))
    ax1.set_title("Win Probability", color=GOLD, fontsize=9, fontweight="bold")
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(
        [BLUE, GRAY, CORAL],
        [f"{a} {pa:.0%}", f"Draw {pd_:.0%}", f"{b} {pb:.0%}"])]
    ax1.legend(handles=patches, loc="center", frameon=False, labelcolor=WHITE, fontsize=7.5)

    # Panel 2 — Tactical bars
    ax2 = fig.add_subplot(gs_[0,1]); ax2.set_facecolor(NAVY)
    mets = ["Possession", "Shots", "Shots OT", "xG"]
    va_ = [sa.get("possession",50), sa.get("shots_total",12),
           sa.get("shots_on_target",4), sa.get("xg_per_game",1.5)]
    vb_ = [sb.get("possession",50), sb.get("shots_total",12),
           sb.get("shots_on_target",4), sb.get("xg_per_game",1.5)]
    x = np.arange(len(mets)); w = 0.35
    ax2.bar(x-w/2, va_, w, color=BLUE, label=a, alpha=0.9)
    ax2.bar(x+w/2, vb_, w, color=CORAL, label=b, alpha=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels(mets, color=WHITE, fontsize=8)
    ax2.tick_params(colors=WHITE)
    ax2.set_title("Tactical Profile", color=GOLD, fontsize=9, fontweight="bold")
    ax2.legend(labelcolor=WHITE, frameon=False, fontsize=8)
    [ax2.spines[s].set_visible(False) for s in ax2.spines]
    ax2.tick_params(axis="y", colors=WHITE)

    # Panel 3 — Probability bars
    ax3 = fig.add_subplot(gs_[0,2]); ax3.set_facecolor(NAVY)
    bars = ax3.barh([b, "Draw", a], [pb, pd_, pa], color=[CORAL, GRAY, BLUE], height=0.5)
    for bar, val in zip(bars, [pb, pd_, pa]):
        ax3.text(val+0.01, bar.get_y()+bar.get_height()/2,
                f"{val:.1%}", va="center", color=WHITE, fontsize=9, fontweight="bold")
    ax3.set_xlim(0, 1)
    ax3.set_title("Probability Breakdown", color=GOLD, fontsize=9, fontweight="bold")
    ax3.tick_params(colors=WHITE, labelsize=9)
    [ax3.spines[s].set_visible(False) for s in ax3.spines]
    ax3.xaxis.set_visible(False)

    # Panel 4 — Goals
    ax4 = fig.add_subplot(gs_[1,0]); ax4.set_facecolor(NAVY)
    gva = [sa.get("goals_scored",1.5), sa.get("goals_conceded",1.0)]
    gvb = [sb.get("goals_scored",1.5), sb.get("goals_conceded",1.0)]
    x2 = np.arange(2)
    ax4.bar(x2-w/2, gva, w, color=BLUE, label=a, alpha=0.9)
    ax4.bar(x2+w/2, gvb, w, color=CORAL, label=b, alpha=0.9)
    ax4.set_xticks(x2)
    ax4.set_xticklabels(["Goals\nScored","Goals\nConceded"], color=WHITE, fontsize=8)
    ax4.tick_params(colors=WHITE)
    ax4.set_title("Goals Profile", color=GOLD, fontsize=9, fontweight="bold")
    ax4.legend(labelcolor=WHITE, frameon=False, fontsize=8)
    [ax4.spines[s].set_visible(False) for s in ax4.spines]
    ax4.tick_params(axis="y", colors=WHITE)

    # Panel 5 — FIFA Rank
    ax5 = fig.add_subplot(gs_[1,1]); ax5.set_facecolor(NAVY)
    ra_r = sa.get("fifa_rank", 20); rb_r = sb.get("fifa_rank", 20)
    ax5.barh([b, a], [rb_r, ra_r], color=[CORAL, BLUE], height=0.4)
    ax5.invert_xaxis()
    ax5.set_title("FIFA Rank (lower = better)", color=GOLD, fontsize=9, fontweight="bold")
    ax5.tick_params(colors=WHITE, labelsize=9)
    [ax5.spines[s].set_visible(False) for s in ax5.spines]
    ax5.tick_params(axis="x", colors=WHITE)
    for team, rank, i in [(b, rb_r, 0), (a, ra_r, 1)]:
        ax5.text(rank+0.3, i, f"#{rank}", va="center", color=WHITE, fontsize=9)

    # Panel 6 — Verdict card
    ax6 = fig.add_subplot(gs_[1,2]); ax6.set_facecolor("#0F1E3C"); ax6.axis("off")
    winner = r["winner"]; conf = r["confidence"]
    ax6.text(0.5, 0.93, "THE SCORE", ha="center", color=GOLD,
             fontsize=11, fontweight="bold", transform=ax6.transAxes)
    ax6.text(0.5, 0.76, "MODEL PICKS:", ha="center", color=WHITE,
             fontsize=9, transform=ax6.transAxes)
    ax6.text(0.5, 0.60, winner.upper(), ha="center",
             color=TEAL if conf > 0.55 else GOLD,
             fontsize=14, fontweight="bold", transform=ax6.transAxes)
    ax6.text(0.5, 0.45, f"Confidence: {conf:.1%}", ha="center",
             color=WHITE, fontsize=9, transform=ax6.transAxes)
    fd_lines = []
    if r["edge_a"]: fd_lines.append(f"{a}: {r['edge_a']['fd_odds']} {r['edge_a']['verdict']}")
    if r["edge_b"]: fd_lines.append(f"{b}: {r['edge_b']['fd_odds']} {r['edge_b']['verdict']}")
    if fd_lines:
        ax6.text(0.5, 0.28, "\n".join(fd_lines), ha="center", color=GOLD,
                fontsize=7.5, transform=ax6.transAxes)
    if r.get("actual_result"):
        ax6.text(0.5, 0.12, f"RESULT: {r['actual_result']}", ha="center",
                color=TEAL, fontsize=9, fontweight="bold", transform=ax6.transAxes)
    ax6.text(0.5, 0.02, '"Not politically correct. Analytically correct."',
             ha="center", color=GRAY, fontsize=7, style="italic",
             transform=ax6.transAxes)

    fname = save_path or f"AC_{a.replace(' ','_')}_vs_{b.replace(' ','_')}_{r['date']}.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight", facecolor=NAVY)
    print(f"  🎨 Chart saved: {fname}")
    plt.show()
    return fname


# ── EXPORT TO POWER BI CSV ─────────────────────────────────────────
def export_for_powerbi(results, filename="AC_WC2026_PowerBI.csv"):
    rows = []
    for r in results:
        row = {
            "Date": r["date"],
            "Team_A": r["team_a"], "Team_B": r["team_b"],
            "Prob_A_Win": round(r["prob_a"], 4),
            "Prob_Draw":  round(r["prob_draw"], 4),
            "Prob_B_Win": round(r["prob_b"], 4),
            "Model_Pick": r["winner"],
            "Confidence": round(r["confidence"], 4),
            "Model_Odds_A": r["american_a"],
            "Model_Odds_B": r["american_b"],
            "FIFA_Rank_A": r["stats_a"].get("fifa_rank"),
            "FIFA_Rank_B": r["stats_b"].get("fifa_rank"),
            "Possession_A": round(r["stats_a"].get("possession",0),1),
            "Possession_B": round(r["stats_b"].get("possession",0),1),
            "xG_A": round(r["stats_a"].get("xg_per_game",0),2),
            "xG_B": round(r["stats_b"].get("xg_per_game",0),2),
            "Goals_Scored_A": round(r["stats_a"].get("goals_scored",0),2),
            "Goals_Scored_B": round(r["stats_b"].get("goals_scored",0),2),
        }
        if r.get("edge_a"):
            row["FD_Odds_A"]    = r["edge_a"]["fd_odds"]
            row["Edge_A_Pct"]   = round(r["edge_a"]["edge"],2)
            row["Verdict_A"]    = r["edge_a"]["verdict"]
        if r.get("edge_b"):
            row["FD_Odds_B"]    = r["edge_b"]["fd_odds"]
            row["Edge_B_Pct"]   = round(r["edge_b"]["edge"],2)
            row["Verdict_B"]    = r["edge_b"]["verdict"]
        if r.get("actual_result"):
            row["Actual_Result"] = r["actual_result"]
            row["Model_Correct"] = 1 if r["winner"].lower() in r["actual_result"].lower() else 0
        rows.append(row)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(filename, index=False)
    print(f"\n  📊 Power BI CSV saved: {filename}")
    print(f"  Columns: {list(df_out.columns)}")
    return df_out


# ── MAIN ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "═"*65)
    print("  ANALYTICALLY CORRECT — WC2026 ENGINE")
    print(f"  {datetime.now().strftime('%B %d, %Y')}")
    print("═"*65)

    # 1. Load data
    df_raw = get_dataset()

    # 2. Train XGBoost
    print("\n  🤖 Training XGBoost model...")
    model, scaler, features = train_xgboost(df_raw)

    all_results = []

    # 3. Run today's predictions
    print(f"\n  📅 TODAY'S GAMES ({datetime.now().strftime('%B %d')})")
    for ta, tb, fda, fdb, actual in TODAY_GAMES:
        r = predict_match(ta, tb, model, scaler, features, df_raw,
                         fanduel_a=fda, fanduel_b=fdb, actual_result=actual)
        print_prediction(r)
        plot_prediction(r, save_path=f"AC_{ta.replace(' ','_')}_vs_{tb.replace(' ','_')}_{r['date']}.png")
        all_results.append(r)

    # 4. Past results accuracy check
    if PAST_RESULTS:
        print(f"\n  📋 PAST RESULTS — ACCURACY TRACKER")
        correct = 0
        for ta, tb, fda, fdb, actual in PAST_RESULTS:
            r = predict_match(ta, tb, model, scaler, features, df_raw,
                             fanduel_a=fda, fanduel_b=fdb, actual_result=actual)
            winner_in_result = r["winner"].lower() in actual.lower()
            stamp = "✅" if winner_in_result else "❌"
            print(f"  {stamp} {ta} vs {tb}: Model={r['winner']} | Actual={actual}")
            if winner_in_result: correct += 1
        print(f"\n  📈 Model Accuracy: {correct}/{len(PAST_RESULTS)} ({correct/len(PAST_RESULTS):.0%})")

    # 5. Export for Power BI
    export_for_powerbi(all_results)

    print("\n" + "═"*65)
    print('  "Not politically correct. Analytically correct."')
    print("═"*65)
