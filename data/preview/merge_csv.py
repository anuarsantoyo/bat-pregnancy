#!/usr/bin/env python3
"""Merge all Trovan logger CSVs in the data folder into a single clean CSV.

Input files follow the naming scheme: YYYYMMDD_COLONY_BOX_LOGGER_Species_Details.csv

Each file is a Trovan reader export that may be ';' or ',' delimited.
The merged output has a single, clean `date` column (ISO YYYY-MM-DD):
  - the logger's Date: field when it is a valid DD.MM.YYYY date in 2024
  - otherwise the filename date, plus one day per distinct corrupted value
    seen in the file. The corrupted reader values still roll over at midnight,
    so the first distinct value maps to the filename date and each later
    distinct value maps to the following day.

Empty / constant columns from the raw exports are dropped. Kept columns:
date, colony, box, logger, species, details, Transponder Code:, Time:.

Usage:
    python merge_csv.py
"""

import glob
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = DATA_DIR / "combined_data.csv"

EXPECTED_COLUMNS = [
    "#",
    "Transponder Type:",
    "Transponder Code:",
    "Date:",
    "Time:",
    "Event:",
    "Unit #:",
    "Antenna #:",
    "Memo:",
    "Custom:",
]

METADATA_COLUMNS = ["date", "colony", "box", "logger", "species", "details"]

KEPT_COLUMNS = [
    "date",
    "colony",
    "box",
    "logger",
    "species",
    "details",
    "Transponder Code:",
    "Time:",
]


def detect_separator(path):
    """Return ';' or ',' based on which delimiter appears in the header line."""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        header = fh.readline()
    return ";" if ";" in header else ","


def parse_filename(stem):
    """Parse YYYYMMDD_COLONY_BOX_LOGGER_Species_Details into a metadata dict."""
    parts = stem.split("_")
    if len(parts) < 6:
        raise ValueError(
            f"Filename does not match YYYYMMDD_COLONY_BOX_LOGGER_Species_Details: {stem}"
        )
    date_str, colony, box, logger, species = parts[:5]
    details = "_".join(parts[5:])

    try:
        date = datetime.strptime(date_str, "%Y%m%d").date().isoformat()
    except ValueError:
        date = date_str  # keep raw if the date prefix is not a valid YYYYMMDD

    return {
        "date": date,
        "colony": colony,
        "box": box,
        "logger": logger,
        "species": species,
        "details": details,
    }


def parse_logger_date(value):
    """Return ISO date if value is a valid DD.MM.YYYY date in 2024, else None."""
    if pd.isna(value):
        return None
    try:
        dt = datetime.strptime(str(value), "%d.%m.%Y")
    except ValueError:
        return None
    if dt.year != 2024:
        return None
    return dt.date().isoformat()


def read_logger_csv(path):
    sep = detect_separator(path)
    df = pd.read_csv(
        path, sep=sep, encoding="utf-8-sig", dtype=str, engine="python"
    )
    # Drop the trailing unnamed column produced by a trailing ';'
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
    return df


def main():
    csv_files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    csv_files = [f for f in csv_files if Path(f).name != OUTPUT_FILE.name]

    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    frames = []
    total_rows = 0
    for path in csv_files:
        stem = Path(path).stem
        meta = parse_filename(stem)
        df = read_logger_csv(path)

        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            print(
                f"WARNING: {Path(path).name}: missing columns {missing}, "
                "keeping file columns as-is"
            )

        # Drop rows where every data column is empty (logger export artifacts)
        data_cols = [c for c in EXPECTED_COLUMNS if c in df.columns]
        empty_mask = df[data_cols].isna().all(axis=1)
        if empty_mask.any():
            df = df[~empty_mask]
            print(f"  (dropped {int(empty_mask.sum())} fully-empty row(s))")

        # Build the clean date: logger Date: where valid, otherwise filename
        # date plus one day per distinct corrupted value seen in the file.
        clean = df["Date:"].apply(parse_logger_date)
        corrupted = clean.isna()
        if corrupted.any():
            offsets = {}
            for value in df.loc[corrupted, "Date:"]:
                if value not in offsets:
                    offsets[value] = len(offsets)
            base = datetime.strptime(meta["date"], "%Y-%m-%d")
            mapping = {
                v: (base + timedelta(days=off)).date().isoformat()
                for v, off in offsets.items()
            }
            clean.loc[corrupted] = [
                mapping[v] for v in df.loc[corrupted, "Date:"]
            ]
            print(f"  (recovered {int(corrupted.sum())} corrupted date rows: {mapping})")

        # Final layout: metadata columns first, then the useful logger columns
        df["date"] = clean
        for i, col in enumerate(METADATA_COLUMNS[1:]):
            df.insert(i, col, meta[col])
        df = df[KEPT_COLUMNS]

        frames.append(df)
        total_rows += len(df)
        print(f"{Path(path).name}: {len(df)} rows")

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"\nMerged {len(frames)} files, {total_rows} total data rows")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Columns: {list(merged.columns)}")


if __name__ == "__main__":
    main()
