# src/00_parse_espn_fixtures.py
from pathlib import Path
from datetime import datetime
import pandas as pd
import re

# === Paste the ESPN text between the triple quotes ===
raw_text = r"""
Friday, Aug. 15, 2025
Liverpool vs. AFC Bournemouth
Saturday, Aug. 16, 2025
Aston Villa vs. Newcastle United
Brighton & Hove Albion vs. Fulham
Sunderland vs. West Ham United
Tottenham Hotspur vs. Burnley
Wolverhampton Wanderers vs. Manchester City
Sunday, Aug. 17, 2025
Chelsea vs. Crystal Palace
Nottingham Forest vs. Brentford
Manchester United vs. Arsenal
Monday, Aug. 18, 2025
Leeds United vs. Everton

Friday, Aug. 22 2025
West Ham United vs. Chelsea
Saturday, Aug. 23, 2025
Manchester City vs. Tottenham Hotspur
AFC Bournemouth vs. Wolves
Brentford vs. Aston Villa
Burnley vs. Sunderland
Arsenal vs. Leeds United
Sunday, Aug. 24, 2025
Crystal Palace vs. Nottingham Forest
Everton vs. Brighton
Fulham vs. Manchester United
Monday, Aug. 25, 2025
Newcastle United vs. Liverpool
Saturday, Aug. 30, 2025
Chelsea vs. Fulham
Manchester United vs. Burnley
Sunderland vs. Brentford
Tottenham Hotspur vs. AFC Bournemouth
Wolves vs. Everton
Leeds United vs. Newcastle United
Sunday, Aug. 31, 2025
Brighton vs. Manchester City
Nottingham Forest vs. West Ham United
Liverpool vs. Arsenal
Aston Villa vs. Crystal Palace
Saturday, Sept. 13, 2025
Arsenal vs. Nottingham Forest
AFC Bournemouth vs. Brighton
Crystal Palace vs. Sunderland
Everton vs. Aston Villa
Fulham vs. Leeds United
Newcastle United vs. Wolves
West Ham United vs. Tottenham Hotspur
Brentford vs. Chelsea
Sunday, Sept. 14, 2025
Burnley vs. Liverpool
Manchester City vs. Manchester United
Saturday, Sept. 20, 2025
Liverpool vs. Everton
AFC Bournemouth vs. Newcastle United
Brighton vs. Tottenham Hotspur
Burnley vs. Nottingham Forest
West Ham United vs. Crystal Palace
Wolves vs. Leeds United
Manchester United vs. Chelsea
Fulham vs. Brentford
Saturday, Sept. 21, 2025
Sunderland vs. Aston Villa
Arsenal vs. Manchester City
Saturday, Sept. 27, 2025
Brentford vs. Manchester United
Aston Villa vs. Fulham
Chelsea vs. Brighton
Crystal Palace vs. Liverpool
Leeds United vs. AFC Bournemouth
Manchester City vs. Burnley
Nottingham Forest vs. Sunderland
Sunday, Sept. 27, 2025
Tottenham Hotspur vs. Wolves
Newcastle United vs. Arsenal
Monday, Sept. 28, 2025
Everton vs. West Ham United
Saturday, Oct. 4, 2025
AFC Bournemouth vs. Fulham
Arsenal vs. West Ham United
Aston Villa vs. Burnley
Brentford vs. Manchester City
Chelsea vs. Liverpool
Everton vs. Crystal Palace
Leeds United vs. Tottenham Hotspur
Manchester United vs. Sunderland
Newcastle United vs. Nottingham Forest
Wolves vs. Brighton
Saturday, Oct. 18, 2025
Brighton vs. Newcastle United
Burnley vs. Leeds United
Crystal Palace vs. AFC Bournemouth
Fulham vs. Arsenal
Liverpool vs. Manchester United
Manchester City vs. Everton
Nottingham Forest vs. Chelsea
Sunderland vs. Wolves
Tottenham Hotspur vs. Aston Villa
West Ham United vs. Brentford
Saturday, Oct. 25, 2025
AFC Bournemouth vs. Nottingham Forest
Arsenal vs. Crystal Palace
Aston Villa vs. Manchester City
Brentford vs. Liverpool
Chelsea vs. Sunderland
Everton vs. Tottenham Hotspur
Leeds United vs. West Ham United
Manchester United vs. Brighton
Newcastle United vs. Fulham
Wolves vs. Burnley
Saturday, Nov. 1, 2025
Brighton vs. Leeds United
Burnley vs. Arsenal
Crystal Palace vs. Brentford
Fulham vs. Wolves
Liverpool vs. Aston Villa
Manchester City vs. AFC Bournemouth
Nottingham Forest vs. Manchester United
Sunderland vs. Everton
Tottenham Hotspur vs. Chelsea
West Ham United vs. Newcastle United
Saturday, Nov. 8, 2025
Aston Villa vs. AFC Bournemouth
Brentford vs. Newcastle United
Chelsea vs. Wolves
Crystal Palace vs. Brighton
Everton vs. Fulham
Manchester City vs. Liverpool
Nottingham Forest vs. Leeds United
Sunderland vs. Arsenal
Tottenham Hotspur vs. Manchester United
West Ham United vs. Burnley
Saturday, Nov. 22, 2025
AFC Bournemouth vs. West Ham United
Arsenal vs. Tottenham Hotspur
Brighton vs. Brentford
Burnley vs. Chelsea
Fulham vs. Sunderland
Leeds United vs. Aston Villa
Liverpool vs. Nottingham Forest
Manchester United vs. Everton
Newcastle United vs. Manchester City
Wolves vs. Crystal Palace
Saturday, Nov. 29, 2025
Aston Villa vs. Wolves
Brentford vs. Burnley
Chelsea vs. Arsenal
Crystal Palace vs. Manchester United
Everton vs. Newcastle United
Manchester City vs. Leeds United
Nottingham Forest vs. Brighton
Sunderland vs. AFC Bournemouth
Tottenham Hotspur vs. Fulham
West Ham United vs. Liverpool
Wednesday, Dec. 3, 2025
AFC Bournemouth vs. Everton
Arsenal vs. Brentford
Brighton vs. Aston Villa
Burnley vs. Crystal Palace
Fulham vs. Manchester City
Leeds United vs. Chelsea
Liverpool vs. Sunderland
Manchester United vs. West Ham United
Newcastle United vs. Tottenham Hotspur
Wolves vs. Nottingham Forest
Saturday, Dec. 6, 2025
AFC Bournemouth vs. Chelsea
Aston Villa vs. Arsenal
Brighton vs. West Ham United
Everton vs. Nottingham Forest
Fulham vs. Crystal Palace
Leeds United vs. Liverpool
Manchester City vs. Sunderland
Newcastle United vs. Burnley
Tottenham Hotspur vs. Brentford
Wolves vs. Manchester United
Saturday, Dec. 13, 2025
Arsenal vs. Wolves
Brentford vs. Leeds United
Burnley vs. Fulham
Chelsea vs. Everton
Crystal Palace vs. Manchester City
Liverpool vs. Brighton
Manchester United vs. AFC Bournemouth
Nottingham Forest vs. Tottenham Hotspur
Sunderland vs. Newcastle United
West Ham United vs. Aston Villa
Saturday, Dec. 20, 2025
AFC Bournemouth vs. Burnley
Aston Villa vs. Manchester United
Brighton vs. Sunderland
Everton vs. Arsenal
Fulham vs. Nottingham Forest
Leeds United vs. Crystal Palace
Manchester City vs. West Ham United
Newcastle United vs. Chelsea
Tottenham Hotspur vs. Liverpool
Wolves vs. Brentford
Saturday, Dec. 27, 2025
Arsenal vs. Brighton
Brentford vs. AFC Bournemouth
Burnley vs. Everton
Chelsea vs. Aston Villa
Crystal Palace vs. Tottenham Hotspur
Liverpool vs. Wolves
Manchester United vs. Newcastle United
Nottingham Forest vs. Manchester City
Sunderland vs. Leeds United
West Ham United vs. Fulham
Tuesday, Dec. 30, 2025
Arsenal vs. Aston Villa
Brentford vs. Tottenham Hotspur
Burnley vs. Newcastle United
Chelsea vs. AFC Bournemouth
Crystal Palace vs. Fulham
Liverpool vs. Leeds United
Manchester United vs. Wolves
Nottingham Forest vs. Everton
Sunderland vs. Manchester City
West Ham United vs. Brighton
Saturday, Jan. 3, 2026
AFC Bournemouth vs. Arsenal
Aston Villa vs. Nottingham Forest
Brighton vs. Burnley
Everton vs. Brentford
Fulham vs. Liverpool
Leeds United vs. Manchester United
Manchester City vs. Chelsea
Newcastle United vs. Crystal Palace
Tottenham Hotspur vs. Sunderland
Wolves vs. West Ham United
Wednesday, Jan. 7, 2026
AFC Bournemouth vs. Tottenham Hotspur
Arsenal vs. Liverpool
Brentford vs. Sunderland
Burnley vs. Manchester United
Crystal Palace vs. Aston Villa
Everton vs. Wolves
Fulham vs. Chelsea
Manchester City vs. Brighton
Newcastle United vs. Leeds United
West Ham United vs. Nottingham Forest
Saturday, Jan. 17, 2026
Aston Villa vs. Everton
Brighton vs. AFC Bournemouth
Chelsea vs. Brentford
Leeds United vs. Fulham
Liverpool vs. Burnley
Manchester United vs. Manchester City
Nottingham Forest vs. Arsenal
Sunderland vs. Crystal Palace
Tottenham Hotspur vs. West Ham United
Wolves vs. Newcastle United
Saturday, Jan. 24, 2026
AFC Bournemouth vs. Liverpool
Arsenal vs. Manchester United
Brentford vs. Nottingham Forest
Burnley vs. Tottenham Hotspur
Crystal Palace vs. Chelsea
Everton vs. Leeds United
Fulham vs. Brighton
Manchester City vs. Wolves
Newcastle United vs. Aston Villa
West Ham United vs. Sunderland
Saturday, Jan. 31, 2026
Aston Villa vs. Brentford
Brighton vs. Everton
Chelsea vs. West Ham United
Leeds United vs. Arsenal
Liverpool vs. Newcastle United
Manchester United vs. Fulham
Nottingham Forest vs. Crystal Palace
Sunderland vs. Burnley
Tottenham Hotspur vs. Manchester City
Wolves vs. AFC Bournemouth
Saturday, Feb. 7, 2026
AFC Bournemouth vs. Aston Villa
Arsenal vs. Sunderland
Brighton vs. Crystal Palace
Burnley vs. West Ham United
Fulham vs. Everton
Leeds United vs. Nottingham Forest
Liverpool vs. Manchester City
Manchester United vs. Tottenham Hotspur
Newcastle United vs. Brentford
Wolves vs. Chelsea
Wednesday, Feb. 11, 2026
Aston Villa vs. Brighton
Brentford vs. Arsenal
Chelsea vs. Leeds United
Crystal Palace vs. Burnley
Everton vs. AFC Bournemouth
Manchester City vs. Fulham
Nottingham Forest vs. Wolves
Sunderland vs. Liverpool
Tottenham Hotspur vs. Newcastle United
West Ham United vs. Manchester United
Saturday, Feb. 21, 2026
Aston Villa vs. Leeds United
Brentford vs. Brighton
Chelsea vs. Burnley
Crystal Palace vs. Wolves
Everton vs. Manchester United
Manchester City vs. Newcastle United
Nottingham Forest vs. Liverpool
Sunderland vs. Fulham
Tottenham Hotspur vs. Arsenal
West Ham United vs. AFC Bournemouth
Saturday, Feb. 28, 2026
AFC Bournemouth vs. Sunderland
Arsenal vs. Chelsea
Brighton vs. Nottingham Forest
Burnley vs. Brentford
Fulham vs. Tottenham Hotspur
Leeds United vs. Manchester City
Liverpool vs. West Ham United
Manchester United vs. Crystal Palace
Newcastle United vs. Everton
Wolves vs. Aston Villa
Wednesday, March 4, 2026
AFC Bournemouth vs. Brentford
Aston Villa vs. Chelsea
Brighton vs. Arsenal
Everton vs. Burnley
Fulham vs. West Ham United
Leeds United vs. Sunderland
Manchester City vs. Nottingham Forest
Newcastle United vs. Manchester United
Tottenham Hotspur vs. Crystal Palace
Wolves vs. Liverpool
Saturday, March 14, 2026
Arsenal vs. Everton
Brentford vs. Wolves
Burnley vs. AFC Bournemouth
Chelsea vs. Newcastle United
Crystal Palace vs. Leeds United
Liverpool vs. Tottenham Hotspur
Manchester United vs. Aston Villa
Nottingham Forest vs. Fulham
Sunderland vs. Brighton
West Ham United vs. Manchester City
Saturday, March 21, 2026
AFC Bournemouth vs. Manchester United
Aston Villa vs. West Ham United
Brighton vs. Liverpool
Everton vs. Chelsea
Fulham vs. Burnley
Leeds United vs. Brentford
Manchester City vs. Crystal Palace
Newcastle United vs. Sunderland
Tottenham Hotspur vs. Nottingham Forest
Wolves vs. Arsenal
Saturday, April 11, 2026
Arsenal vs. AFC Bournemouth
Brentford vs. Everton
Burnley vs. Brighton
Chelsea vs. Manchester City
Crystal Palace vs. Newcastle United
Liverpool vs. Fulham
Manchester United vs. Leeds United
Nottingham Forest vs. Aston Villa
Sunderland vs. Tottenham Hotspur
West Ham United vs. Wolves
Saturday, April 18, 2026
Aston Villa vs. Sunderland
Brentford vs. Fulham
Chelsea vs. Manchester United
Crystal Palace vs. West Ham United
Everton vs. Liverpool
Leeds United vs. Wolves
Manchester City vs. Arsenal
Newcastle United vs. AFC Bournemouth
Nottingham Forest vs. Burnley
Tottenham Hotspur vs. Brighton
Saturday, April 25, 2026
AFC Bournemouth vs. Leeds United
Arsenal vs. Newcastle United
Brighton vs. Chelsea
Burnley vs. Manchester City
Fulham vs. Aston Villa
Liverpool vs. Crystal Palace
Manchester United vs. Brentford
Sunderland vs. Nottingham Forest
West Ham United vs. Everton
Wolves vs. Tottenham Hotspur
Saturday, May 2, 2026
AFC Bournemouth vs. Crystal Palace
Arsenal vs. Fulham
Aston Villa vs. Tottenham Hotspur
Brentford vs. West Ham United
Chelsea vs. Nottingham Forest
Everton vs. Manchester City
Leeds United vs. Burnley
Manchester United vs. Liverpool
Newcastle United vs. Brighton
Wolves vs. Sunderland
Saturday, May 9, 2026
Brighton vs. Wolves
Burnley vs. Aston Villa
Crystal Palace vs. Everton
Fulham vs. AFC Bournemouth
Liverpool vs. Chelsea
Manchester City vs. Brentford
Nottingham Forest vs. Newcastle United
Sunderland vs. Manchester United
Tottenham Hotspur vs. Leeds United
West Ham United vs. Arsenal
Sunday, May 17, 2026
AFC Bournemouth vs. Manchester City
Arsenal vs. Burnley
Aston Villa vs. Liverpool
Brentford vs. Crystal Palace
Chelsea vs. Tottenham Hotspur
Everton vs. Sunderland
Leeds United vs. Brighton
Manchester United vs. Nottingham Forest
Newcastle United vs. West Ham United
Wolves vs. Fulham
Sunday, May 24, 2026
Brighton vs. Manchester United
Burnley vs. Wolves
Crystal Palace vs. Arsenal
Fulham vs. Newcastle United
Liverpool vs. Brentford
Manchester City vs. Aston Villa
Nottingham Forest vs. AFC Bournemouth
Sunderland vs. Chelsea
Tottenham Hotspur vs. Everton
West Ham United vs. Leeds United
"""

OUT = Path("data/fixtures_2526.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Canonical mapping -> must match our training names
CANON = {
    "Tottenham Hotspur":"Tottenham","Spurs":"Tottenham",
    "Brighton & Hove Albion":"Brighton","Brighton and Hove Albion":"Brighton",
    "Wolverhampton Wanderers":"Wolverhampton","Wolves":"Wolverhampton",
    "West Ham United":"West Ham","Newcastle United":"Newcastle","Leeds United":"Leeds",
    "Nottingham Forest":"Nottingham Forest","Nott'm Forest":"Nottingham Forest","Nott Forest":"Nottingham Forest",
    "AFC Bournemouth":"Bournemouth",
    "Man Utd":"Manchester United","Man United":"Manchester United","Manchester Utd":"Manchester United",
    "Man City":"Manchester City","Manchester City":"Manchester City"
}

# Expected 25/26 teams (after canonicalization)
TEAMS_2526 = {
    "Arsenal","Aston Villa","Bournemouth","Brentford","Brighton","Burnley","Chelsea","Crystal Palace",
    "Everton","Fulham","Leeds","Liverpool","Manchester City","Manchester United","Newcastle",
    "Nottingham Forest","Sunderland","Tottenham","West Ham","Wolverhampton"
}

date_patterns = [
    "%A, %b. %d, %Y",   # Saturday, Sept. 13, 2025
    "%A, %b %d, %Y",    # Saturday, Sep 13, 2025
    "%A, %b. %d %Y",    # Friday, Aug. 22 2025
    "%A, %b %d %Y",     # Friday, Aug 22 2025
]

WEEKDAYS = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")

def normalize_months(s: str) -> str:
    # ESPN uses "Sept." and sometimes omits commas
    s = s.replace("Sept.", "Sep.")
    s = s.replace("Sept", "Sep")
    return s

def parse_fixtures(text: str) -> pd.DataFrame:
    current_date = None
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("*"):
            continue

        # Date lines start with weekday and contain a year
        if line.startswith(WEEKDAYS) and re.search(r"\b(2025|2026)\b", line):
            norm = normalize_months(line.replace("  ", " "))
            parsed = None
            for fmt in date_patterns:
                try:
                    parsed = datetime.strptime(norm, fmt)
                    break
                except ValueError:
                    continue
            if parsed:
                current_date = parsed.date()
                continue  # move to next line

        # Fixture line "Team A vs. Team B (optional time)"
        if " vs. " in line and current_date is not None:
            left, right = line.split(" vs. ", 1)
            # strip anything in parentheses on the right (e.g., "(12:30 UK)")
            right = re.sub(r"\s*\(.*?\)", "", right).strip()
            home = left.strip()
            away = right.strip()
            rows.append({"Date": current_date.isoformat(), "HomeTeam": home, "AwayTeam": away})

    df = pd.DataFrame(rows, columns=["Date","HomeTeam","AwayTeam"])
    if df.empty:
        raise SystemExit("Parsed 0 fixtures. Check that you pasted the ESPN text correctly.")
    return df

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HomeTeam"] = df["HomeTeam"].map(lambda x: CANON.get(x, x))
    df["AwayTeam"] = df["AwayTeam"].map(lambda x: CANON.get(x, x))
    return df

def sanity_checks(df: pd.DataFrame):
    # team set
    teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))
    unknown = [t for t in teams if t not in TEAMS_2526]
    if unknown:
        print("WARNING: Team names not in 25/26 list (after canonicalization):", unknown)

    # count
    n = len(df)
    print(f"Parsed fixtures: {n} rows")
    if n != 380:
        raise SystemExit(f"Expected 380 matches, got {n}. Your text is probably incomplete.")

    # quick duplicates check
    dups = df.duplicated(subset=["Date","HomeTeam","AwayTeam"]).sum()
    if dups:
        print(f"WARNING: found {dups} duplicate rows.")

def main():
    df = parse_fixtures(raw_text)
    df = canonicalize(df)
    # normalize Date to YYYY-MM-DD (string) or actual date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df.sort_values("Date").reset_index(drop=True)
    sanity_checks(df)
    df.to_csv(OUT, index=False)
    print(f"Wrote normalized fixtures -> {OUT}")

if __name__ == "__main__":
    main()