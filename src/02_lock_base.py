import pandas as pd
from pathlib import Path

IN  = Path("data/processed/pl_matches.csv")
OUT = Path("data/processed/pl_matches_clean.csv")
REPORT_DIR = Path("reports"); REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical short names used throughout the pipeline
def canonicalize_team_names(s: pd.Series) -> pd.Series:
    canon = {
        # Man United/City
        "Man United": "Manchester United", "Man Utd": "Manchester United", "Manchester Utd": "Manchester United",
        "Man City": "Manchester City",

        # Spurs / Tottenham
        "Tottenham Hotspur": "Tottenham", "Spurs": "Tottenham",

        # Brighton
        "Brighton and Hove Albion": "Brighton", "Brighton & Hove Albion": "Brighton",

        # Wolves
        "Wolves": "Wolverhampton", "Wolverhampton Wanderers": "Wolverhampton",

        # West Ham / Newcastle / Leeds
        "West Ham United": "West Ham",
        "Newcastle United": "Newcastle",
        "Leeds United": "Leeds",

        # Forest
        "Nott'm Forest": "Nottingham Forest", "Nott Forest": "Nottingham Forest",

        # Bournemouth
        "AFC Bournemouth": "Bournemouth",

        # West Brom variations
        "West Brom": "West Bromwich Albion", "West Bromwich": "West Bromwich Albion",
    }
    return s.astype(str).replace(canon)

# Your declared 25/26 teams (normalized to our canonical short names)
TEAMS_2526 = {
    "Arsenal","Aston Villa","Bournemouth","Brentford","Brighton","Burnley","Chelsea","Crystal Palace",
    "Everton","Fulham","Leeds","Liverpool","Manchester City","Manchester United","Newcastle","Nottingham Forest",
    "Sunderland","Tottenham","West Ham","Wolverhampton"
}

def main():
    assert IN.exists(), f"Missing input {IN}"
    df = pd.read_csv(IN, parse_dates=["Date"])

    # Basic cleaning
    df = df.dropna(subset=["Date","HomeTeam","AwayTeam","FTR"]).copy()
    df["HomeTeam"] = canonicalize_team_names(df["HomeTeam"])
    df["AwayTeam"] = canonicalize_team_names(df["AwayTeam"])

    # Season label (Aug–May rule)
    df["season"] = df["Date"].dt.year + (df["Date"].dt.month >= 8)

    # Deduplicate (rare but safer)
    before = len(df)
    df = df.sort_values("Date").drop_duplicates(subset=["Date","HomeTeam","AwayTeam"])
    removed = before - len(df)

    # Save the clean base
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # === QA REPORT ===
    counts = df.groupby("season").size().sort_index()
    ftr = df["FTR"].value_counts(normalize=True).sort_index()

    with open(REPORT_DIR/"base_qa.txt","w",encoding="utf-8") as f:
        f.write(f"Rows: {len(df)}  (deduplicated: {removed})\n")
        f.write(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}\n\n")

        f.write("Matches per season (last 15):\n")
        f.write(counts.tail(15).to_string() + "\n\n")

        f.write("FTR distribution (H/D/A shares):\n")
        for k, v in (ftr*100).round(1).items():
            f.write(f"  {k}: {v:.1f}%\n")

        # Compare latest season’s team set to your 25/26 list (just a sanity view)
        latest = counts.index.max()
        teams_latest = set(pd.unique(pd.concat([
            df.loc[df["season"]==latest,"HomeTeam"],
            df.loc[df["season"]==latest,"AwayTeam"]
        ])))
        missing_from_your_list = teams_latest - TEAMS_2526
        not_present_in_latest  = TEAMS_2526 - teams_latest

        f.write("\nLatest season in data: " + str(latest) + "\n")
        f.write("Teams in latest season (canonical):\n")
        f.write(", ".join(sorted(teams_latest)) + "\n")
        f.write("\nTeams present in latest season but NOT in your 25/26 list:\n")
        f.write(", ".join(sorted(missing_from_your_list)) or "None")
        f.write("\nTeams in your 25/26 list but NOT present in latest season of the dataset:\n")
        f.write(", ".join(sorted(not_present_in_latest)) or "None")
        f.write("\n")

    print(f"Saved {len(df)} rows to {OUT}")
    print("QA -> reports/base_qa.txt")

if __name__ == "__main__":
    main()