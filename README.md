# ⚽ Extended Analytics — WC2026 Prediction Engine

> *"Not politically correct. Analytically correct."*

XGBoost-powered match prediction engine for the 2026 FIFA World Cup.
Compares model probabilities vs FanDuel/DraftKings odds to find edges.
Exports branded visualizations + Power BI CSV every matchday.

Disclaimer:"This repository serves as a showcase for the Extended Analytics project. The core XGBoost predictive engine and proprietary data pipelines are not open-source and remain the exclusive intellectual property of the author."
---

## 🎙️ About the Show

**Extended Analytics** is a prediction accountability podcast.
I make explicit predictions before every game, then score who is right — my model, the media, or the betting market.

- 🎵 Spotify: @extendedanalytics
- 📱 TikTok: @extendedanalytics

---

## 📊 Model Overview

| Feature | Detail |
|---|---|
| Algorithm | XGBoost Classifier |
| Features | FIFA Rank, Possession, Shots, xG, Goals, Head-to-Head Differentials |
| Teams | All 48 WC2026 teams profiled |
| Data Sources | worldcup26.ir → BallDontLie → football-data.org → FIFA Profile Matrix |
| Output | Win probabilities + FanDuel edge analysis + PNG charts + Power BI CSV |

---

## 🚀 Quick Start

### 1. Install dependencies
```python
pip install xgboost scikit-learn pandas numpy matplotlib requests
```

### 2. Update today's games
Edit the `TODAY_GAMES` list in the script:
```python
TODAY_GAMES = [
    ("Netherlands", "Sweden",     -145,  +370,  None),  # 1:00 PM ET
    ("Germany",     "Ivory Coast",-210,  +500,  None),  # 4:00 PM ET
    ("Ecuador",     "Curacao",    -800,  +2000, None),  # 4:00 PM ET
    ("Tunisia",     "Japan",      +550,  -180,  None),  # 8:00 PM ET
]
```

### 3. Run
```bash
python analytically_correct_wc2026_FINAL.py
```

### 4. After games — add results
```python
PAST_RESULTS = [
    ("France", "Senegal", -200, +550, "France 3-1 Senegal"),
]
```

---

## 📈 Key Variables That Affect Winning

| Variable | Impact | Why |
|---|---|---|
| FIFA Rank difference | 🔴 HIGH | Best proxy for overall team quality |
| xG per game | 🔴 HIGH | Expected goals = true attacking quality |
| Shots on target | 🟡 MEDIUM | Directly correlates with scoring |
| Possession % | 🟡 MEDIUM | Controls game tempo |
| Goals conceded/game | 🟡 MEDIUM | Defensive solidity |
| Shots total | 🟢 LOW | Volume without accuracy less predictive |

---

## 💰 How to Read FanDuel Edge

```
🟢 MODEL LIKES = Model thinks team more likely to win than FD implies (value bet)
🔴 FADE        = Model thinks team less likely to win than FD implies (avoid)
⚪ NEUTRAL     = Model and FanDuel roughly agree
```

Edge formula: `(Model Win % - FanDuel Implied %) × 100`
Positive edge = model sees value vs market.

---

## 📊 Power BI Setup

After running the script, import `AC_WC2026_PowerBI.csv` into Power BI:

1. **Open Power BI Desktop**
2. **Get Data → Text/CSV** → Select `AC_WC2026_PowerBI.csv`
3. **Recommended visuals:**
   - Clustered bar chart: `Team_A` / `Team_B` vs `Prob_A_Win` / `Prob_B_Win`
   - Gauge: `Confidence` per match
   - Table: All columns with conditional formatting on `Edge_A_Pct` / `Edge_B_Pct`
   - Scatter plot: `FD_Implied` vs `Model_Prob` — dots above diagonal = value
   - Card: `Model_Correct` sum / count = accuracy tracker

---

## 🎯 Accuracy Tracker

| Episode | Match | Model Pick | Actual | Result |
|---|---|---|---|---|
| EP.001 | France vs Senegal | France (72.7%) | France 3-1 Senegal | ✅ |

*Updated after every matchday*

---

## 🗂️ File Structure

```
analytically-correct-wc2026/
├── analytically_correct_wc2026_FINAL.py  # Main prediction engine
├── AC_WC2026_PowerBI.csv                  # Power BI data export
├── charts/                                # PNG visualizations per match
│   ├── AC_France_vs_Senegal_2026-06-16.png
│   ├── AC_Netherlands_vs_Sweden_2026-06-20.png
│   └── ...
└── README.md
```

---

## 🔧 Adding a New Team

Add to `TEAM_PROFILES` dictionary:
```python
"Team Name": [fifa_rank, possession%, shots_per_game, shots_on_target,
              goals_scored_per_game, goals_conceded_per_game, xg_per_game],
```

---

*Built with XGBoost, pandas, matplotlib | Host: Mathieu A. Ndong | MSBA LeBow College of Business*
