
import pandas as pd
import numpy as np
from pathlib import Path
import re

RAW = Path("data/raw/cache.footballdata")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def read_csv_smart(p: Path) -> pd.DataFrame:
    for kwargs in (
        dict(encoding="latin-1", on_bad_lines="skip"),
        dict(encoding="utf-8", on_bad_lines="skip"),
        dict(encoding="latin-1", sep=";", on_bad_lines="skip"),
        dict(encoding="utf-8", sep=";", on_bad_lines="skip"),
    ):
        try:
            return pd.read_csv(p, **kwargs)
        except Exception:
            pass
    return pd.DataFrame()

def path_looks_like_epl(path: Path) -> bool:
    s = str(path).lower()
    # must clearly be england + premier league
    good = (
        ("england" in s or "/eng." in s or "e0" in s) and
        ("premierleague" in s or "premier-league" in s or "/1-premierleague" in s or "eng.1" in s or "e0" in s)
    )
    # explicitly exclude cups/others
    bad_terms = (
        "fa-cup","facup","league-cup","efl-cup","carabao","community-shield","charity-shield",
        "championship","league-one","league-1","league_two","national","conference",
        "playoff","play-off","friendly","u21","u23","women"
    )
    return good and not any(bt in s for bt in bad_terms)

def dataframe_is_epl(df: pd.DataFrame) -> bool:
    cols_lower = {c.lower(): c for c in df.columns}

    # 1) classic football-data: Div == 'E0'
    if "div" in cols_lower:
        try:
            return df[cols_lower["div"]].astype(str).str.upper().eq("E0").any()
        except Exception:
            pass

    # 2) competition/league text contains "Premier League" (English)
    for key in ["competition","league","tournament"]:
        if key in cols_lower:
            col = cols_lower[key]
            text = df[col].astype(str).str.lower()
            if text.str.contains("premier league").any():
                # and not cups
                if not text.str.contains("fa cup|league cup|community shield|efl", regex=True).any():
                    return True

    # 3) team-name cue as a fallback (very strict)
    team_cols = [cols_lower.get(x) for x in ["hometeam","awayteam","team 1","team1","home","away","team 2","team2"] if x in cols_lower]
    if team_cols:
        sample = pd.concat([df[team_cols[0]].astype(str).head(200),
                            df[team_cols[-1]].astype(str).head(200)], ignore_index=True).str.lower()
        cues = ["arsenal","chelsea","liverpool","manchester united","man utd","manchester city","tottenham","spurs"]
        if sample.apply(lambda x: any(k in x for k in cues)).mean() > 0.10:
            return True

    return False

def normalize_to_standard(df: pd.DataFrame) -> pd.DataFrame:
    """Return frame with Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, optional odds."""
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame()

    # Case A: football-data style
    if all(k in cols for k in ["hometeam","awayteam"]) and ({"fthg","ftag"}.issubset(cols) or "ftr" in cols):
        out["HomeTeam"] = df[cols["hometeam"]]
        out["AwayTeam"] = df[cols["awayteam"]]
        if "fthg" in cols and "ftag" in cols:
            out["FTHG"] = pd.to_numeric(df[cols["fthg"]], errors="coerce")
            out["FTAG"] = pd.to_numeric(df[cols["ftag"]], errors="coerce")
        if "ftr" in cols and "FTR" not in out.columns:
            out["FTR"] = df[cols["ftr"]].astype(str).str.upper().str.slice(0,1)
        # Odds if present
        for k in ["b365h","b365d","b365a"]:
            if k in cols:
                out[k.upper()] = pd.to_numeric(df[cols[k]], errors="coerce")
        # Date
        if "date" in cols:
            out["Date"] = pd.to_datetime(df[cols["date"]], errors="coerce", dayfirst=True)
        elif "matchdate" in cols:
            out["Date"] = pd.to_datetime(df[cols["matchdate"]], errors="coerce")
    # Case B: openfootball-style (Team 1 / Team 2 / FT)
    elif any(k in cols for k in ["team 1","team1","home"]) and any(k in cols for k in ["team 2","team2","away"]):
        home = cols.get("team 1") or cols.get("team1") or cols.get("home")
        away = cols.get("team 2") or cols.get("team2") or cols.get("away")
        out["HomeTeam"] = df[home].astype(str)
        out["AwayTeam"] = df[away].astype(str)
        # score may be in FT, Score, Result
        score_col = cols.get("ft") or cols.get("score") or cols.get("result")
        if score_col:
            m = df[score_col].astype(str).str.extract(r"(?P<FTHG>-?\d+)\s*[-:]\s*(?P<FTAG>-?\d+)")
            out["FTHG"] = pd.to_numeric(m["FTHG"], errors="coerce")
            out["FTAG"] = pd.to_numeric(m["FTAG"], errors="coerce")
        # Date
        date_col = cols.get("date") or cols.get("matchdate")
        if date_col:
            out["Date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    else:
        return pd.DataFrame()  # unsupported format

    # Build FTR if missing
    if "FTR" not in out.columns and {"FTHG","FTAG"}.issubset(out.columns):
        out["FTR"] = np.where(out["FTHG"] > out["FTAG"], "H",
                       np.where(out["FTHG"] < out["FTAG"], "A", "D"))

    # Basic cleaning
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"])
    # Team canonicalization (light)
    canon = {
        "Man United":"Manchester United","Man Utd":"Manchester United","Manchester Utd":"Manchester United",
        "Man City":"Manchester City","Spurs":"Tottenham","Tottenham Hotspur":"Tottenham",
        "Wolves":"Wolverhampton","Brighton and Hove Albion":"Brighton","West Brom":"West Bromwich Albion",
        "West Bromwich":"West Bromwich Albion"
    }
    for c in ["HomeTeam","AwayTeam"]:
        if c in out.columns:
            out[c] = out[c].replace(canon)

    keep = [c for c in ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","B365H","B365D","B365A"] if c in out.columns]
    return out[keep].dropna(subset=["HomeTeam","AwayTeam"])

def main():
    if not RAW.exists():
        raise SystemExit("Input folder not found: data/raw/cache.footballdata")
    csvs = list(RAW.rglob("*.csv"))
    if not csvs:
        raise SystemExit("No CSV files under data/raw/cache.footballdata")

    frames = []
    scanned = 0
    shortlisted = 0

    for p in csvs:
        scanned += 1

        # Case 1: path clearly looks like EPL -> read full file
        if path_looks_like_epl(p):
            df = read_csv_smart(p)
            if df.empty or not dataframe_is_epl(df):
                continue

        # Case 2: path is ambiguous -> sample for detection, then re-read full file
        else:
            df_sample = read_csv_smart(p).head(200)  # small sample ONLY for detection
            if df_sample.empty or not dataframe_is_epl(df_sample):
                continue
            df = read_csv_smart(p)  # <-- re-read FULL file now

        shortlisted += 1
        norm = normalize_to_standard(df)
        if not norm.empty:
            frames.append(norm)

    if not frames:
        raise SystemExit("No Premier League rows found. Repo layout may differ; ping me with two sample CSV paths and their first 5 rows.")
    pl = pd.concat(frames, ignore_index=True).drop_duplicates()
    # final sanity
    if "Date" in pl.columns:
        pl = pl.sort_values("Date")
    pl.to_csv(OUT/"pl_matches.csv", index=False)
    print(f"Scanned {scanned} CSVs, shortlisted {shortlisted}, saved {len(pl)} EPL rows to data/processed/pl_matches.csv")

if __name__ == "__main__":
    main()