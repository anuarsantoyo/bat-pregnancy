#!/usr/bin/env python3
"""Build the wide, day-resolved PIT-tag activity matrix for the bat-pregnancy project.

For every ``sit`` threshold in the processed logger data
(``data/raw/ML_bat_repstat/yearly/<year>/<colony>/<year>_<colony>_lid_sit<X>s.csv``)
this produces one wide CSV with columns::

    Year, Bat Id, Colonie, Day1, Day2, ..., Day365, Lactating

* one row = one bat in one year
* ``DayN`` = number of independent visit events registered in *bat-day* N, where a
  bat-day runs 12:00 -> 12:00 of the next human day (Day1 = Jan 1 12:00 -> Jan 2 12:00).
* 29 February is dropped, so the table is always 365 columns wide.
* ``Lactating`` is ``Reproduction State`` from ``yearly_table_L_Mbec.xlsx`` sheet
  ``Repstat`` (the dictionary describes it as "reproductive classification based on
  Lactation"), joined by ``ID4``. Absent -> empty cell.

Normalisation / hygiene decisions (all logged to stdout):
* activity ids are upper-cased and left-zero-padded to 10 chars, which merges the
  7-digit truncated reads with their full 10-char counterparts;
* reader error codes (single-character ids, all-same-character ids such as
  ``FFFFFFFFFF``/``EEEEEEEEEE``/``9999999999``) are dropped;
* a row's colony comes from the label table when the bat is labelled that year,
  otherwise from the folder the detections came from;
* detections whose bat-day falls outside the file's calendar year are dropped
  (they belong to the neighbouring year's file edge).

Usage
-----
    python3 src/build_daily_counts.py                 # all six sit thresholds
    python3 src/build_daily_counts.py --sits 10 600   # subset
"""

from __future__ import annotations

import argparse
import calendar
import collections
import csv
import datetime as dt
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "data", "raw", "ML_bat_repstat")
YEARLY = os.path.join(RAW, "yearly")
XLSX = os.path.join(RAW, "yearly_table_L_Mbec.xlsx")
OUTDIR = os.path.join(REPO, "data", "processed", "daily_counts")

SITS = [10, 30, 60, 120, 300, 600]
N_DAYS = 365
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
HEX10 = re.compile(r"^[0-9A-F]{10}$")

# --------------------------------------------------------------------------- utils


def norm_tag(raw: str) -> str | None:
    """Uppercase + zero-pad an activity tag id; return None for junk/error codes."""
    tag = (raw or "").strip().upper()
    if not tag:
        return None
    if len(tag) < 4:  # truncated reads like '6', '7'
        return None
    if len(set(tag)) == 1:  # reader error codes: FFFFFFFFFF, EEEEEEEEEE, 9999999999
        return None
    tag = tag.zfill(10)
    if not HEX10.match(tag):
        if len(set(tag)) == 1:
            return None
        return None
    return tag


def day_index(d: dt.date) -> int | None:
    """1-based day slot with 29 February removed. None for 29 Feb itself."""
    if d.month == 2 and d.day == 29:
        return None
    doy = d.timetuple().tm_yday
    if calendar.isleap(d.year) and doy > 59:
        doy -= 1
    if 1 <= doy <= N_DAYS:
        return doy
    return None


def parse_ts(text: str) -> dt.datetime | None:
    text = (text or "").strip().strip('"')
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d.%m.%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def bat_day(ts: dt.datetime) -> dt.datetime:
    """A bat-day starts at 12:00; subtract 12 h and keep the calendar date."""
    return ts - dt.timedelta(hours=12)


# ------------------------------------------------------------------ label table


def _col_of(ref: str) -> int:
    """'AB12' -> 27 (zero-based column index). Empty cells are omitted from the
    XML, so the cell's `r` reference is the only reliable position."""
    letters = "".join(ch for ch in ref if ch.isalpha()).upper()
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx_rows(path: str, sheet_file: str) -> list[list[str]]:
    """Read a worksheet honouring each cell's `r` reference so that empty cells
    do not shift the row left."""
    z = zipfile.ZipFile(path)
    shared = [
        "".join(t.text or "" for t in si.iter(XLSX_NS + "t"))
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(XLSX_NS + "si")
    ]
    root = ET.fromstring(z.read(sheet_file))
    out = []
    for row in root.iter(XLSX_NS + "row"):
        cells: dict[int, str] = {}
        for c in row.iter(XLSX_NS + "c"):
            ref = c.get("r")
            idx = _col_of(ref) if ref else len(cells)
            v = c.find(XLSX_NS + "v")
            if v is None:
                cells[idx] = ""
            elif c.get("t") == "s":
                cells[idx] = shared[int(v.text)]
            else:
                cells[idx] = v.text or ""
        width = max(cells) + 1 if cells else 0
        out.append([cells.get(i, "") for i in range(width)])
    return out


def load_labels() -> dict[tuple[int, str], dict]:
    rows = read_xlsx_rows(XLSX, "xl/worksheets/sheet1.xml")
    hdr = rows[0]
    ix = {h: i for i, h in enumerate(hdr)}
    labels: dict[tuple[int, str], dict] = {}
    for r in rows[1:]:
        def get(name: str) -> str:
            i = ix.get(name)
            return (r[i] if i is not None and i < len(r) else "").strip()

        id4 = get("ID4").upper()
        year_s = get("Year")
        if not id4 or not year_s.isdigit():
            continue
        labels[(int(year_s), id4)] = {
            "lactating": get("Reproduction State"),
            "colony": get("Colony_clean").upper(),
            "birth_event": get("birth_event"),
        }
    return labels


# ------------------------------------------------------------------- the build


def scan_sit(sit: int) -> dict:
    """Return {(year, tag): {'colony_counts': Counter, 'days': Counter(int)}}."""
    files = sorted(glob.glob(os.path.join(YEARLY, "*", "*", f"*_sit{sit}s.csv")))
    agg: dict[tuple[int, str], dict] = {}
    stats = collections.Counter()
    for path in files:
        base = os.path.basename(path)
        year_s, colony = base.split("_")[0], os.path.basename(os.path.dirname(path)).upper()
        if not year_s.isdigit():
            continue
        year = int(year_s)
        stats["files"] += 1
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                stats["rows"] += 1
                tag = norm_tag(row.get("id", ""))
                if tag is None:
                    stats["dropped_junk_id"] += 1
                    continue
                ts = parse_ts(row.get("date_time", ""))
                if ts is None:
                    stats["unparsed_ts"] += 1
                    continue
                bd = bat_day(ts)
                if bd.year != year:
                    stats["dropped_year_edge"] += 1
                    continue
                di = day_index(bd.date())
                if di is None:
                    stats["dropped_feb29"] += 1
                    continue
                rec = agg.setdefault((year, tag), {"colony": collections.Counter(), "days": collections.Counter()})
                rec["colony"][colony] += 1
                rec["days"][di] += 1
                stats["kept"] += 1
    stats["bat_years"] = len(agg)
    return agg, stats


def suffix_label(tag: str, year: int, labels: dict, by_id4: dict[str, list[str]]) -> tuple[str | None, str | None]:
    """Resolve (lactating, colony) for an activity tag by suffix-matching ID4."""
    cands = [id4 for id4 in by_id4 if tag.endswith(id4)]
    if not cands:
        return None, None
    # prefer a candidate that has a label row for this year
    exact = [c for c in cands if (year, c) in labels]
    if len(cands) > 1 and not exact:
        return None, None  # genuinely ambiguous, do not guess
    id4 = (exact or cands)[0]
    lab = labels.get((year, id4))
    if lab is None:
        return None, None
    return lab["lactating"] or None, lab["colony"] or None


def build(sits: list[int]) -> None:
    labels = load_labels()
    by_id4: dict[str, list[str]] = collections.defaultdict(list)
    for (_y, id4) in labels:
        by_id4[id4].append(id4)

    os.makedirs(OUTDIR, exist_ok=True)
    table_years = collections.Counter(y for (y, _i) in labels)
    print(f"label table: {len(labels)} (year, ID4) rows across {len(set(table_years))} years; {len(by_id4)} distinct ID4")

    for sit in sits:
        agg, stats = scan_sit(sit)
        header = ["Year", "Bat Id", "Colonie"] + [f"Day{i}" for i in range(1, N_DAYS + 1)] + ["Lactating"]
        out = os.path.join(OUTDIR, f"daily_counts_sit{sit}s.csv")

        n_lab = 0
        n_unlab = 0
        n_multi_folder = 0
        n_ambiguous = 0
        zero_rows = 0
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for (year, tag) in sorted(agg):
                rec = agg[(year, tag)]
                lact, colony = suffix_label(tag, year, labels, by_id4)
                if lact is None and colony is None and len(rec["colony"]) > 1:
                    n_ambiguous += 1
                if colony is None:
                    colony = rec["colony"].most_common(1)[0][0]
                    n_unlab += 1
                    if len(rec["colony"]) > 1:
                        n_multi_folder += 1
                else:
                    n_lab += 1
                days = rec["days"]
                if not days:
                    zero_rows += 1
                row = [year, tag, colony] + [days.get(i, 0) for i in range(1, N_DAYS + 1)] + [lact or ""]
                w.writerow(row)

        size_mb = os.path.getsize(out) / 1e6
        print(
            f"sit{sit:>4}s -> {os.path.relpath(out, REPO)}  "
            f"rows={stats['bat_years']} ({n_lab} labelled / {n_unlab} unlabelled, "
            f"{n_ambiguous} ambiguous-colony, {n_multi_folder} multi-folder)  "
            f"{size_mb:.1f} MB"
        )
        print(
            f"        rows read={stats['rows']} kept={stats['kept']} | "
            f"junk ids={stats['dropped_junk_id']} feb29={stats['dropped_feb29']} "
            f"year-edge={stats['dropped_year_edge']} unparsed-ts={stats['unparsed_ts']}"
        )

    # coverage report: labelled bat-years with no activity in the 2008-2024 window
    years_with_data = {
        int(os.path.basename(p)) for p in glob.glob(os.path.join(YEARLY, "*")) if os.path.isdir(p)
    }
    print(f"\nyears present in yearly/: {min(years_with_data)}-{max(years_with_data)} ({len(years_with_data)})")
    print("done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sits", nargs="*", type=int, default=SITS)
    args = ap.parse_args()
    build(args.sits)


if __name__ == "__main__":
    sys.exit(main())
