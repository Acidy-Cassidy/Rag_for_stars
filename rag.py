#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional heavy deps (only needed for embedding/index steps)
try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    faiss = None
    SentenceTransformer = None


# -----------------------------
# Utilities
# -----------------------------

def _is_nan(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


def fmt(x: Any, nd: int = 3) -> str:
    if _is_nan(x) or x == "":
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        s = f"{float(x):.{nd}f}"
        s = s.rstrip("0").rstrip(".")
        return s
    return str(x)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _safe_int(x: Any) -> Optional[int]:
    if _is_nan(x) or x == "":
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def _row_lower_map(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if k is None:
            continue
        out[str(k).strip().lower()] = v
    return out


def get_field(row: Dict[str, Any], row_lc: Dict[str, Any], key: str) -> Any:
    if key in row:
        return row.get(key)
    return row_lc.get(key.strip().lower())


def _remove_if_exists(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _norm_colname(c: Any) -> str:
    """
    Normalize CSV column names to stable SQLite column names.
    """
    return str(c).strip().strip('"').strip("'").strip().lower()


def _pick_id_col(cols: set) -> Optional[str]:
    """
    Auto-detect the Gaia ID/join key across common export variants.
    Assumes cols are normalized (lowercase).
    """
    candidates = [
        "source_id",
        "sourceid",
        "gaia_source_id",
        "gaia_sourceid",
        "designation",
        "id",
    ]
    for want in candidates:
        if want in cols:
            return want
    return None


# -----------------------------
# Readable output helpers
# -----------------------------

def _re_find(pattern: str, text: str, flags: int = 0) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _to_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def shorten_id(s: Any, keep_start: int = 7, keep_end: int = 5) -> str:
    if s is None:
        return "-"
    st = str(s).strip()
    if not st or st == "None":
        return "-"
    if len(st) <= keep_start + keep_end + 3:
        return st
    return f"{st[:keep_start]}...{st[-keep_end:]}"


def summarize_card(card: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["type"] = _re_find(r"\btype:\s*([^\|]+)", card)

    out["ra"] = _to_float(_re_find(r"ra:\s*([0-9\.\-]+)\s*deg", card))
    out["dec"] = _to_float(_re_find(r"dec:\s*([0-9\.\-]+)\s*deg", card))

    out["dist_pc"] = _to_float(_re_find(r"dist:\s*([0-9\.\-]+)\s*pc", card))
    out["plx_mas"] = _to_float(_re_find(r"parallax:\s*([0-9\.\-]+)\s*mas", card))

    out["Kmag"] = _to_float(_re_find(r"\bK\s*([0-9\.\-]+)", card))
    out["Gmag"] = _to_float(_re_find(r"\bG\s*([0-9\.\-]+)", card))
    out["Vmag"] = _to_float(_re_find(r"\bV\s*([0-9\.\-]+)", card))

    out["Teff_K"] = _to_int(_re_find(r"Teff\s*([0-9\.\-]+)\s*K", card))
    out["R_Rsun"] = _to_float(_re_find(r"\bR\s*([0-9\.\-]+)\s*Rsun", card))

    out["MH"] = _to_float(_re_find(r"\[M/H\]\s*([0-9\.\-]+)", card))

    out["class"] = _re_find(r"\bclass\s*([A-Za-z0-9\-\_]+)", card)

    out["GAIA"] = _re_find(r"\bGAIA\s*([0-9]+)", card)
    out["HIP"] = _re_find(r"\bHIP\s*([0-9]+)", card)
    out["TWOMASS"] = _re_find(r"\b2MASS\s*([0-9A-Za-z\+\-]+)", card)

    return out


def _fmt_num(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    try:
        xf = float(x)
        s = f"{xf:.{nd}f}"
        s = s.rstrip("0").rstrip(".")
        return s if s else "0"
    except Exception:
        return str(x)


def _fmt_int(x: Any) -> str:
    if x is None:
        return "-"
    try:
        return str(int(x))
    except Exception:
        return str(x)


def sort_results_for_display(results: List[Dict[str, Any]], sort_key: str) -> List[Dict[str, Any]]:
    key = (sort_key or "").strip().lower()
    if key not in {"score", "gmag", "kmag", "dist", "teff", "r", "mh"}:
        return results

    def get_val(r: Dict[str, Any]) -> float:
        card = r.get("card", "") or ""
        s = summarize_card(card)
        if key == "score":
            v = r.get("score", None)
            return float(v) if v is not None else float("-inf")
        if key == "gmag":
            v = s.get("Gmag", None)
            return float(v) if v is not None else float("inf")
        if key == "kmag":
            v = s.get("Kmag", None)
            return float(v) if v is not None else float("inf")
        if key == "dist":
            v = s.get("dist_pc", None)
            return float(v) if v is not None else float("inf")
        if key == "teff":
            v = s.get("Teff_K", None)
            return float(v) if v is not None else float("inf")
        if key == "r":
            v = s.get("R_Rsun", None)
            return float(v) if v is not None else float("inf")
        if key == "mh":
            v = s.get("MH", None)
            return float(v) if v is not None else float("inf")
        return float("inf")

    reverse = True if key == "score" else False
    return sorted(results, key=get_val, reverse=reverse)


def print_results_table(results: List[Dict[str, Any]],
                        max_rows: int = 50,
                        wide: bool = False,
                        gaia_short: bool = True) -> None:
    rows: List[Dict[str, Any]] = []
    for r in results[:max_rows]:
        s = summarize_card(r.get("card", "") or "")
        gaia_val = s.get("GAIA") or "-"
        hip_val = s.get("HIP") or "-"
        if not wide and gaia_short:
            gaia_val = shorten_id(gaia_val)

        rows.append({
            "rowid": str(r.get("ID", "-")),
            "orig_ID": str(r.get("orig_ID", "-")) if r.get("orig_ID") is not None else "-",
            "score": _fmt_num(r.get("score"), 4) if r.get("score") is not None else "-",
            "G": _fmt_num(s.get("Gmag"), 3),
            "K": _fmt_num(s.get("Kmag"), 3),
            "Teff": _fmt_int(s.get("Teff_K")),
            "MH": _fmt_num(s.get("MH"), 2),
            "R": _fmt_num(s.get("R_Rsun"), 1),
            "class": s.get("class") or "-",
            "dist_pc": _fmt_num(s.get("dist_pc"), 1),
            "ra": _fmt_num(s.get("ra"), 3),
            "dec": _fmt_num(s.get("dec"), 3),
            "GAIA": gaia_val,
            "HIP": hip_val,
        })

    headers = ["rowid", "orig_ID", "score", "G", "K", "Teff", "MH", "R", "class", "dist_pc", "ra", "dec", "GAIA", "HIP"]
    right_align = {"score", "G", "K", "Teff", "MH", "R", "dist_pc", "ra", "dec"}

    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    def sep(ch: str = "-") -> str:
        return "  ".join(ch * widths[h] for h in headers)

    def fmt_cell(h: str, v: str) -> str:
        if h in right_align:
            return v.rjust(widths[h])
        return v.ljust(widths[h])

    print(sep("-"))
    print("  ".join(fmt_cell(h, h) if h not in right_align else h.rjust(widths[h]) for h in headers))
    print(sep("="))
    for row in rows:
        print("  ".join(fmt_cell(h, str(row.get(h, ""))) for h in headers))
    print(sep("-"))
    if len(results) > max_rows:
        print(f"(showing first {max_rows} of {len(results)})")


def print_result_details(r: Dict[str, Any], wide: bool = False) -> None:
    card = r.get("card", "") or ""
    s = summarize_card(card)

    gaia = s.get("GAIA") or "-"
    hip = s.get("HIP") or "-"
    twomass = s.get("TWOMASS") or "-"
    if not wide:
        gaia = shorten_id(gaia)

    print("")
    print(f"rowid:   {r.get('ID')}   orig_ID: {r.get('orig_ID')}   score: {_fmt_num(r.get('score'), 4)}")
    print(f"type:    {s.get('type') or '-'}   class: {s.get('class') or '-'}")
    print(f"coords:  ra { _fmt_num(s.get('ra'), 6) } deg   dec { _fmt_num(s.get('dec'), 6) } deg")
    print(f"dist:    { _fmt_num(s.get('dist_pc'), 3) } pc   plx { _fmt_num(s.get('plx_mas'), 3) } mas")
    print(f"mags:    G { _fmt_num(s.get('Gmag'), 3) }   K { _fmt_num(s.get('Kmag'), 3) }   V { _fmt_num(s.get('Vmag'), 3) }")
    print(f"params:  Teff { _fmt_int(s.get('Teff_K')) } K   [M/H] { _fmt_num(s.get('MH'), 3) }   R { _fmt_num(s.get('R_Rsun'), 3) } Rsun")
    print(f"ids:     GAIA {gaia}   HIP {hip}   2MASS {twomass}")


# -----------------------------
# Star card builder
# -----------------------------

def build_star_card(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    row_lc = _row_lower_map(row)

    rid = _safe_int(get_field(row, row_lc, "_rid"))

    orig_id = fmt(get_field(row, row_lc, "ID"))
    gaia = fmt(get_field(row, row_lc, "GAIA"))
    twomass = fmt(get_field(row, row_lc, "TWOMASS"))
    hip = fmt(get_field(row, row_lc, "HIP"))
    objtype = fmt(get_field(row, row_lc, "objType")) or "STAR"

    title_bits = [b for b in [f"ID {orig_id}", f"GAIA {gaia}", f"2MASS {twomass}", f"HIP {hip}"]
                  if b and not b.endswith(" ")]
    if title_bits:
        parts.append(" | ".join(title_bits))

    if objtype:
        parts.append(f"type: {objtype}")

    ra = fmt(get_field(row, row_lc, "ra"), 6)
    dec = fmt(get_field(row, row_lc, "dec"), 6)
    if ra and dec:
        parts.append(f"ra: {ra} deg, dec: {dec} deg")

    pmra = fmt(get_field(row, row_lc, "pmRA"))
    pmdec = fmt(get_field(row, row_lc, "pmDEC"))
    plx = fmt(get_field(row, row_lc, "plx"))
    dist = fmt(get_field(row, row_lc, "d"))
    kin_bits = []
    if pmra or pmdec:
        kin_bits.append(f"pm: ({pmra},{pmdec}) mas/yr".replace("(,", "(").replace(",)", ")"))
    if plx:
        kin_bits.append(f"parallax: {plx} mas")
    if dist:
        kin_bits.append(f"dist: {dist} pc")
    if kin_bits:
        parts.append("; ".join(kin_bits))

    mags = []
    for label, key in [
        ("V", "Vmag"), ("B", "Bmag"), ("G", "GAIAmag"),
        ("g", "gmag"), ("r", "rmag"), ("i", "imag"), ("z", "zmag"),
        ("J", "Jmag"), ("H", "Hmag"), ("K", "Kmag"),
        ("W1", "w1mag"), ("W2", "w2mag"),
    ]:
        v = fmt(get_field(row, row_lc, key))
        if v:
            mags.append(f"{label} {v}")
    if mags:
        parts.append("mags: " + ", ".join(mags))

    teff = fmt(get_field(row, row_lc, "Teff"))
    logg = fmt(get_field(row, row_lc, "logg"))
    mh = fmt(get_field(row, row_lc, "MH"))
    rad = fmt(get_field(row, row_lc, "rad"))

    phys_bits = []
    if teff:
        phys_bits.append(f"Teff {teff} K")
    if logg:
        phys_bits.append(f"logg {logg}")
    if mh:
        phys_bits.append(f"[M/H] {mh}")
    if rad:
        phys_bits.append(f"R {rad} Rsun")
    if phys_bits:
        parts.append("params: " + ", ".join(phys_bits))

    lumclass = fmt(get_field(row, row_lc, "lumclass"))
    if lumclass:
        parts.append(f"class {lumclass}")

    out = normalize_ws(" | ".join(parts))
    if not out:
        out = f"row {rid}" if rid is not None else "star"
    return out


# -----------------------------
# Legacy CSV detection + SQLite load
# -----------------------------

WANTED_COLS = {
    "id", "objid", "objtype",
    "ra", "dec",
    "teff", "logg", "mh",
    "kmag", "jmag", "hmag", "gaiamag",
    "plx", "d",
}

DELIMS = [",", "\t", "|", ";"]


def _read_head_lines(path: str, n: int = 200) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for _ in range(n):
            try:
                lines.append(next(f))
            except StopIteration:
                break
    return lines


def _parse_cols(line: str, delim: str) -> List[str]:
    try:
        row = next(csv.reader([line], delimiter=delim))
        return [c.strip().strip('"').strip() for c in row]
    except Exception:
        return []


def _score_header_candidate(line: str, delim: str) -> Tuple[int, int]:
    cols = _parse_cols(line, delim)
    cols_lc = [c.lower() for c in cols if c]
    hits = len(set(cols_lc).intersection(WANTED_COLS))
    ncols = len(cols_lc)
    return hits, ncols


def detect_format(csv_path: str) -> Tuple[str, int, List[str]]:
    head_lines = _read_head_lines(csv_path, n=200)

    best_delim = ","
    best_idx = 0
    best_hits = -1
    best_ncols = -1

    for i, line in enumerate(head_lines):
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue

        for d in DELIMS:
            hits, ncols = _score_header_candidate(line, d)
            if hits > best_hits or (hits == best_hits and ncols > best_ncols):
                best_hits, best_ncols = hits, ncols
                best_delim, best_idx = d, i

    return best_delim, best_idx, head_lines


def load_csv_to_sqlite(csv_path: str, db_path: str, table: str = "stars") -> None:
    delim, header_idx, head_lines = detect_format(csv_path)
    skiprows = header_idx

    try:
        df = pd.read_csv(
            csv_path,
            sep=delim,
            skiprows=skiprows,
            header=0,
            encoding="utf-8",
            engine="python",
            on_bad_lines="skip",
        )
    except Exception:
        df = pd.read_csv(
            csv_path,
            sep=delim,
            skiprows=skiprows,
            header=0,
            encoding="utf-8",
            engine="c",
            low_memory=False,
        )

    df.columns = [str(c).strip() for c in df.columns]
    cols_lc = set([c.lower() for c in df.columns])

    hits = len(cols_lc.intersection(WANTED_COLS))
    if hits < 3:
        header_preview = head_lines[header_idx].rstrip("\n") if header_idx < len(head_lines) else ""
        raise RuntimeError(
            "CSV parse sanity failed: not enough known columns detected.\n"
            f"Detected delimiter={repr(delim)} header_line_index={header_idx}\n"
            f"Header line preview: {header_preview[:200]}\n"
            f"Parsed columns (first 30): {df.columns[:30].tolist()}\n"
            "This means the file has a different delimiter/header layout than expected."
        )

    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)

        idx_cols = ["ID", "objID", "ra", "dec", "d", "Teff", "Kmag", "GAIAmag", "plx", "MH"]
        for col in idx_cols:
            if col in df.columns:
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table}("{col}");')
        conn.commit()


# -----------------------------
# Gaia folder ingest
# -----------------------------

def _iter_files(folder: str, pattern: str) -> List[str]:
    p = Path(folder)
    return sorted([str(x) for x in p.glob(pattern) if x.is_file()])


def _score_gaia_header(line: str, delim: str) -> int:
    cols = [_norm_colname(c) for c in line.split(delim)]
    wanted = {"source_id", "sourceid", "ra", "dec", "parallax", "pmra", "pmdec", "phot_g_mean_mag", "teff_gspphot"}
    return len(set(cols) & wanted)


def _detect_csv_layout(path: str) -> Tuple[str, int]:
    """
    Returns (delimiter, skiprows) for Gaia chunk CSVs.

    Robust against:
      - UTF-8 BOM
      - Excel "sep=," preamble line
      - ECSV metadata lines like "# %ECSV 1.0" (and any "# ..." YAML comment block)
      - leading whitespace before '#'
      - header line not being the first line
      - delimiter could be ',', ';', '\t', or '|'
    """
    delims = [",", ";", "\t", "|"]

    lines: List[str] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for _ in range(800):
            try:
                lines.append(next(f))
            except StopIteration:
                break

    if not lines:
        return ",", 0

    hinted_delim: Optional[str] = None
    candidates: List[Tuple[int, int, str]] = []  # (score, idx, delim)

    for i, raw in enumerate(lines):
        line_stripped = raw.strip()
        if not line_stripped:
            continue

        # Excel delimiter hint (not a comment)
        if line_stripped.lower().startswith("sep=") and len(line_stripped) >= 5:
            hinted = line_stripped[4]
            if hinted in delims:
                hinted_delim = hinted
            continue

        # Skip comment/metadata lines (ECSV uses these heavily)
        if raw.lstrip().startswith("#"):
            continue

        # Evaluate this line as a possible header for each delimiter
        for d in delims:
            score = _score_gaia_header(line_stripped, d)
            ncols = len(line_stripped.split(d))
            candidates.append((score * 1000 + ncols, i, d))

    if not candidates:
        return (hinted_delim or ","), 0

    candidates.sort(reverse=True)
    best_score, best_idx, best_delim = candidates[0]

    if hinted_delim is not None:
        hinted_candidates = [c for c in candidates if c[2] == hinted_delim]
        if hinted_candidates:
            hinted_best = hinted_candidates[0]
            if (best_score - hinted_best[0]) < 1000:
                best_score, best_idx, best_delim = hinted_best

    return best_delim, best_idx


def _read_csv_chunks(path: str, chunksize: int) -> Iterator[pd.DataFrame]:
    """
    Chunked CSV reader with:
      - auto delimiter detection
      - skips preamble/header offset
      - ignores comment lines (ECSV metadata) via comment="#"
    This is a true generator (no list(it) materialization).
    """
    delim, skiprows = _detect_csv_layout(path)

    try:
        it = pd.read_csv(
            path,
            sep=delim,
            skiprows=skiprows,
            header=0,
            encoding="utf-8-sig",
            engine="c",
            low_memory=False,
            chunksize=chunksize,
            on_bad_lines="skip",
            comment="#",
        )
    except TypeError:
        it = pd.read_csv(
            path,
            sep=delim,
            skiprows=skiprows,
            header=0,
            encoding="utf-8-sig",
            engine="python",
            chunksize=chunksize,
            on_bad_lines="skip",
        )

    for chunk in it:
        yield chunk


def load_gaia_folder_to_sqlite(folder: str, db_path: str, chunksize: int = 200_000,
                              limit_files: int = 0) -> None:
    gaia_files = _iter_files(folder, "GaiaSource_*.csv")
    ap_files = _iter_files(folder, "AstrophysicalParameters_*.csv")

    if limit_files and limit_files > 0:
        gaia_files = gaia_files[:limit_files]
        ap_files = ap_files[:limit_files]

    if not gaia_files:
        raise RuntimeError(f"No GaiaSource_*.csv files found in {folder}")
    if not ap_files:
        raise RuntimeError(f"No AstrophysicalParameters_*.csv files found in {folder}")

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS gaia_source_raw;")
        conn.execute("DROP TABLE IF EXISTS gaia_astro_raw;")
        conn.commit()

        print(f"[GAIA] Loading {len(gaia_files)} GaiaSource files into gaia_source_raw ...")
        first = True
        for fp in gaia_files:
            print(f"  - {fp}")
            for chunk in _read_csv_chunks(fp, chunksize=chunksize):
                chunk.columns = [_norm_colname(c) for c in chunk.columns]
                chunk.to_sql("gaia_source_raw", conn, if_exists="replace" if first else "append", index=False)
                first = False

        print(f"[GAIA] Loading {len(ap_files)} AstrophysicalParameters files into gaia_astro_raw ...")
        first = True
        for fp in ap_files:
            print(f"  - {fp}")
            for chunk in _read_csv_chunks(fp, chunksize=chunksize):
                chunk.columns = [_norm_colname(c) for c in chunk.columns]
                chunk.to_sql("gaia_astro_raw", conn, if_exists="replace" if first else "append", index=False)
                first = False

        print("[GAIA] Creating indexes ...")

        def cols_of(table: str) -> set:
            rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
            return set([_norm_colname(r[1]) for r in rows])

        src_cols = cols_of("gaia_source_raw")
        ap_cols = cols_of("gaia_astro_raw")

        src_id = _pick_id_col(src_cols)
        ap_id = _pick_id_col(ap_cols)

        if src_id:
            try:
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_gaia_source_raw_{src_id} ON gaia_source_raw("{src_id}");')
            except Exception:
                pass
        if ap_id:
            try:
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_gaia_astro_raw_{ap_id} ON gaia_astro_raw("{ap_id}");')
            except Exception:
                pass

        conn.commit()


def build_stars_table_from_gaia(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS stars;")
        conn.commit()

        def cols_of(table: str) -> set:
            rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
            return set([_norm_colname(r[1]) for r in rows])

        src_cols = cols_of("gaia_source_raw")
        ap_cols = cols_of("gaia_astro_raw")

        src_id = _pick_id_col(src_cols)
        ap_id = _pick_id_col(ap_cols)

        if not src_id:
            raise RuntimeError(
                "gaia_source_raw missing an ID/join column.\n"
                "Expected one of: source_id, sourceid, gaia_source_id, designation, id\n"
                f"Have (first 80): {sorted(list(src_cols))[:80]}"
            )
        if not ap_id:
            raise RuntimeError(
                "gaia_astro_raw missing an ID/join column.\n"
                "Expected one of: source_id, sourceid, gaia_source_id, designation, id\n"
                f"Have (first 80): {sorted(list(ap_cols))[:80]}"
            )

        ra_col = "ra" if "ra" in src_cols else None
        dec_col = "dec" if "dec" in src_cols else None
        plx_col = "parallax" if "parallax" in src_cols else None
        pmra_col = "pmra" if "pmra" in src_cols else None
        pmdec_col = "pmdec" if "pmdec" in src_cols else None
        gmag_col = "phot_g_mean_mag" if "phot_g_mean_mag" in src_cols else None

        teff_col = "teff_gspphot" if "teff_gspphot" in ap_cols else ("teff" if "teff" in ap_cols else None)
        logg_col = "logg_gspphot" if "logg_gspphot" in ap_cols else ("logg" if "logg" in ap_cols else None)
        mh_col = "mh_gspphot" if "mh_gspphot" in ap_cols else ("mh" if "mh" in ap_cols else None)
        rad_col = "radius_gspphot" if "radius_gspphot" in ap_cols else ("radius" if "radius" in ap_cols else None)

        sel = []
        sel.append(f's."{src_id}" AS "GAIA"')
        sel.append(f's."{src_id}" AS "ID"')

        sel.append(f's."{ra_col}" AS "ra"' if ra_col else 'NULL AS "ra"')
        sel.append(f's."{dec_col}" AS "dec"' if dec_col else 'NULL AS "dec"')
        sel.append(f's."{pmra_col}" AS "pmRA"' if pmra_col else 'NULL AS "pmRA"')
        sel.append(f's."{pmdec_col}" AS "pmDEC"' if pmdec_col else 'NULL AS "pmDEC"')

        if plx_col:
            sel.append(f's."{plx_col}" AS "plx"')
            sel.append(f'CASE WHEN s."{plx_col}" > 0 THEN 1000.0 / s."{plx_col}" ELSE NULL END AS "d"')
        else:
            sel.append('NULL AS "plx"')
            sel.append('NULL AS "d"')

        sel.append(f's."{gmag_col}" AS "GAIAmag"' if gmag_col else 'NULL AS "GAIAmag"')
        sel.append(f'a."{teff_col}" AS "Teff"' if teff_col else 'NULL AS "Teff"')
        sel.append(f'a."{logg_col}" AS "logg"' if logg_col else 'NULL AS "logg"')
        sel.append(f'a."{mh_col}" AS "MH"' if mh_col else 'NULL AS "MH"')
        sel.append(f'a."{rad_col}" AS "rad"' if rad_col else 'NULL AS "rad"')

        q = f"""
        CREATE TABLE stars AS
        SELECT
          {", ".join(sel)}
        FROM gaia_source_raw s
        LEFT JOIN gaia_astro_raw a
          ON a."{ap_id}" = s."{src_id}";
        """
        print(f"[GAIA] Building normalized stars table (join on {src_id}) ...")
        conn.execute(q)
        conn.commit()

        print("[GAIA] Creating stars indexes ...")
        for col in ["ID", "GAIA", "ra", "dec", "d", "Teff", "MH", "GAIAmag", "plx"]:
            try:
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_stars_{col} ON stars("{col}");')
            except Exception:
                pass
        conn.commit()


# -----------------------------
# Star cards
# -----------------------------

def create_star_cards(db_path: str, source_table: str = "stars", cards_table: str = "star_cards") -> None:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f'SELECT rowid as _rid, * FROM {source_table};', conn)

        cards: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            row = r.to_dict()
            rid = _safe_int(row.get("_rid"))
            if rid is None:
                continue

            card = build_star_card(row)

            cards.append({
                "ID": rid,
                "orig_ID": row.get("ID"),
                "objID": row.get("objID"),
                "card": card
            })

        cdf = pd.DataFrame(cards, columns=["ID", "orig_ID", "objID", "card"])
        cdf.to_sql(cards_table, conn, if_exists="replace", index=False)

        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{cards_table}_ID ON {cards_table}("ID");')
        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{cards_table}_objID ON {cards_table}("objID");')
        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{cards_table}_orig_ID ON {cards_table}("orig_ID");')
        conn.commit()


# -----------------------------
# Embeddings + FAISS index
# -----------------------------

@dataclass
class IndexMeta:
    model_name: str
    dim: int
    id_list: List[int]


def _resolve_device(user_device: str) -> str:
    """
    Returns "cuda" or "cpu" based on user choice and availability.
    """
    dev = (user_device or "auto").strip().lower()
    if dev not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("--device must be one of: auto, cpu, cuda")

    if dev == "cpu":
        return "cpu"

    # auto/cuda: try to use torch if available
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    if dev == "cuda":
        print("[EMB] WARNING: --device cuda requested but CUDA is not available. Falling back to CPU.")
    return "cpu"


def build_faiss_index(
    db_path: str,
    index_path: str,
    meta_path: str,
    cards_table: str = "star_cards",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 1024,
    device: str = "auto",
    embed_where: str = "",
    heartbeat_every_rows: int = 0,
    heartbeat_every_sec: float = 120.0,
) -> None:
    """
    TRUE STREAMING embedding + FAISS build:
    - does not load all cards into RAM
    - encodes batches and adds to FAISS incrementally
    - writes meta id_list at the end

    GPU:
    - if CUDA is available and --device cuda/auto, SentenceTransformer will run on GPU.

    Optional filter:
    - embed_where: SQL WHERE clause applied to star_cards selection.
      Example: "card LIKE '%Teff%'"
      Example: "ID IN (SELECT rowid FROM stars WHERE Teff IS NOT NULL OR MH IS NOT NULL)"  (advanced)

    Heartbeat:
    - prints a progress line periodically so you can confirm it's alive even if tqdm stops repainting.
    """
    if faiss is None or SentenceTransformer is None:
        raise RuntimeError("Missing deps. Install: pip install faiss-cpu sentence-transformers tqdm")

    resolved = _resolve_device(device)
    model = SentenceTransformer(model_name, device=resolved)
    print(f"[EMB] model={model_name} device={resolved} batch_size={batch_size}")

    # Heartbeat tunables (safe defaults)
    if heartbeat_every_rows <= 0:
        heartbeat_every_rows = max(batch_size * 50, 50_000)  # about every ~50 batches, at least 50k rows

    where_sql = ""
    if embed_where and embed_where.strip():
        where_sql = " WHERE " + embed_where.strip()

    with sqlite3.connect(db_path) as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM {cards_table}{where_sql};").fetchone()[0]
        if not total:
            raise RuntimeError("No rows selected for embedding. Check your --embed-where filter (or ingestion).")

        rowish = conn.execute(
            f"SELECT COUNT(*) FROM {cards_table}{where_sql} AND card LIKE 'row %';"
            if where_sql else
            f"SELECT COUNT(*) FROM {cards_table} WHERE card LIKE 'row %';"
        ).fetchone()[0]
        if rowish > int(0.8 * total):
            raise RuntimeError(
                "Cards look wrong: most cards are 'row N'.\n"
                "That means ingestion did not load expected columns.\n"
                "Fix the input ingestion first."
            )

        print(f"[EMB] Streaming encode {total} cards (batch_size={batch_size}) ...")

        cur = conn.cursor()
        cur.execute(f"SELECT ID, card FROM {cards_table}{where_sql} ORDER BY ID;")

        id_list: List[int] = []

        # Pull first batch to initialize dim + index
        first_rows = cur.fetchmany(batch_size)
        if not first_rows:
            raise RuntimeError("star_cards selection is empty after COUNT(*) said otherwise (unexpected).")

        first_ids: List[int] = []
        first_texts: List[str] = []
        for sid, card in first_rows:
            if sid is None:
                continue
            try:
                sid_i = int(sid)
            except Exception:
                continue
            first_ids.append(sid_i)
            first_texts.append("" if card is None else str(card))

        if not first_ids:
            raise RuntimeError("First batch had no valid IDs.")

        X0 = model.encode(
            first_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        X0 = np.asarray(X0, dtype=np.float32)
        dim = int(X0.shape[1])

        index = faiss.IndexFlatIP(dim)
        index.add(X0)
        id_list.extend(first_ids)

        done = len(first_rows)

        pbar = tqdm(total=total, desc="Batches", unit="rows")
        pbar.update(done)

        # Heartbeat state
        last_hb_done = done
        last_hb_time = time.time()

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break

            ids_batch: List[int] = []
            texts_batch: List[str] = []
            for sid, card in rows:
                if sid is None:
                    continue
                try:
                    sid_i = int(sid)
                except Exception:
                    continue
                ids_batch.append(sid_i)
                texts_batch.append("" if card is None else str(card))

            if ids_batch:
                X = model.encode(
                    texts_batch,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                X = np.asarray(X, dtype=np.float32)
                index.add(X)
                id_list.extend(ids_batch)

            done += len(rows)
            pbar.update(len(rows))

            # Heartbeat: every N rows OR every M seconds
            now = time.time()
            if (done - last_hb_done) >= heartbeat_every_rows or (now - last_hb_time) >= heartbeat_every_sec:
                pct = (done / total * 100.0) if total else 0.0
                rate = pbar.format_dict.get("rate", None)
                rate_s = f"{rate:.2f} rows/s" if isinstance(rate, (int, float)) and rate else "?"
                print(f"[EMB] heartbeat: {done}/{total} ({pct:.2f}%)  rate={rate_s} rows/s")
                last_hb_done = done
                last_hb_time = now

        pbar.close()

    faiss.write_index(index, index_path)
    meta = IndexMeta(model_name=model_name, dim=dim, id_list=id_list)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta.__dict__, f, indent=2)

    print(f"[EMB] Done. dim={dim} ntotal={int(getattr(index, 'ntotal', 0))} meta_ids={len(id_list)}")


def load_index(index_path: str, meta_path: str, device: str = "auto"):
    if faiss is None or SentenceTransformer is None:
        raise RuntimeError("Missing deps. Install: pip install faiss-cpu sentence-transformers")

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    resolved = _resolve_device(device)
    model = SentenceTransformer(meta["model_name"], device=resolved)
    return index, meta, model


# -----------------------------
# Hybrid retrieval
# -----------------------------

def sql_filter_ids(conn: sqlite3.Connection,
                   table: str,
                   where_sql: str,
                   params: Tuple[Any, ...]) -> List[int]:
    q = f"SELECT rowid FROM {table} WHERE {where_sql};"
    rows = conn.execute(q, params).fetchall()
    out: List[int] = []
    for r in rows:
        if r and r[0] is not None:
            try:
                out.append(int(r[0]))
            except Exception:
                pass
    return out


def query_hybrid(db_path: str, index_path: str, meta_path: str,
                 q: str,
                 k: int = 10,
                 where_sql: Optional[str] = None,
                 params: Optional[Tuple[Any, ...]] = None,
                 source_table: str = "stars",
                 cards_table: str = "star_cards",
                 device: str = "auto") -> List[Dict[str, Any]]:
    index, meta, model = load_index(index_path, meta_path, device=device)

    with sqlite3.connect(db_path) as conn:
        candidates: Optional[set] = None
        if where_sql:
            ids = sql_filter_ids(conn, source_table, where_sql, params or tuple())
            candidates = set(ids)

        qv = model.encode([q], normalize_embeddings=True)
        qv = np.asarray(qv, dtype=np.float32)

        if candidates is None:
            D, I = index.search(qv, k)
            results: List[Dict[str, Any]] = []
            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx < 0:
                    continue
                sid = int(meta["id_list"][idx])
                row = conn.execute(
                    f'SELECT orig_ID, card FROM {cards_table} WHERE ID=?;',
                    (sid,)
                ).fetchone()
                orig_id = row[0] if row else None
                card = row[1] if row else ""
                results.append({"ID": sid, "orig_ID": orig_id, "score": float(score), "card": card})
            return results

        probe = max(k * 40, 500)
        D, I = index.search(qv, probe)

        filtered: List[Tuple[float, int]] = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            sid = int(meta["id_list"][idx])
            if sid in candidates:
                filtered.append((float(score), sid))
            if len(filtered) >= k:
                break

        out: List[Dict[str, Any]] = []
        for sc, sid in filtered[:k]:
            row = conn.execute(
                f'SELECT orig_ID, card FROM {cards_table} WHERE ID=?;',
                (sid,)
            ).fetchone()
            out.append({"ID": sid, "orig_ID": row[0] if row else None, "score": sc, "card": row[1] if row else ""})
        return out


# -----------------------------
# Sanity check
# -----------------------------

def sanity_check(db_path: str, index_path: str, meta_path: str,
                 cards_table: str = "star_cards",
                 source_table: str = "stars",
                 sample_n: int = 3) -> None:
    print("\n[SANITY] Checking DB + cards + index...")

    with sqlite3.connect(db_path) as conn:
        stars_n = conn.execute(f"SELECT COUNT(*) FROM {source_table};").fetchone()[0]
        cards_n = conn.execute(f"SELECT COUNT(*) FROM {cards_table};").fetchone()[0]
        empty_n = conn.execute(
            f"SELECT COUNT(*) FROM {cards_table} WHERE card IS NULL OR trim(card)='';"
        ).fetchone()[0]

        print(f"[SANITY] stars rows:      {stars_n}")
        print(f"[SANITY] star_cards rows: {cards_n}")
        print(f"[SANITY] empty cards:     {empty_n}")

        if cards_n == 0:
            raise RuntimeError("Sanity check failed: star_cards is empty.")
        if empty_n > 0:
            print("[SANITY] WARNING: some cards are empty.")

        rowish = conn.execute(
            f"SELECT COUNT(*) FROM {cards_table} WHERE card LIKE 'row %';"
        ).fetchone()[0]
        print(f"[SANITY] 'row N' cards:    {rowish}")

        if rowish > int(0.8 * cards_n):
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({source_table});").fetchall()]
            raise RuntimeError(
                "Sanity check failed: most cards are 'row N' (no star fields present).\n"
                f"stars table columns (first 40): {cols[:40]}\n"
                "This means ingestion did not load expected columns."
            )

        print("[SANITY] Sample cards (compact):")
        rows = conn.execute(
            f"SELECT ID, orig_ID, card FROM {cards_table} "
            f"WHERE card IS NOT NULL AND trim(card)<>'' LIMIT ?;",
            (int(sample_n),)
        ).fetchall()
        fake_results = [{"ID": rid, "orig_ID": oid, "score": None, "card": card} for (rid, oid, card) in rows]
        print_results_table(fake_results, max_rows=sample_n, wide=False, gaia_short=True)

    if faiss is None:
        raise RuntimeError("Sanity check failed: faiss not installed.")
    if not os.path.exists(index_path):
        raise RuntimeError("Sanity check failed: FAISS index file missing.")
    if not os.path.exists(meta_path):
        raise RuntimeError("Sanity check failed: meta json missing.")

    idx = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    ntotal = int(getattr(idx, "ntotal", 0))
    id_list_n = len(meta.get("id_list", []))
    print(f"[SANITY] FAISS ntotal:     {ntotal}")
    print(f"[SANITY] meta id_list len: {id_list_n}")

    if ntotal != id_list_n:
        raise RuntimeError("Sanity check failed: FAISS ntotal != meta id_list length.")
    if ntotal == 0:
        raise RuntimeError("Sanity check failed: FAISS index is empty.")

    print("[SANITY] OK.\n")


# -----------------------------
# CLI commands
# -----------------------------

def cmd_build(args: argparse.Namespace) -> None:
    if not args.no_clean:
        print("[0/4] Cleaning old outputs (db/index/meta)...")
        _remove_if_exists(args.db)
        _remove_if_exists(args.index)
        _remove_if_exists(args.meta)

    if args.dir:
        print("[1/4] Loading Gaia folder -> SQLite raw tables...")
        load_gaia_folder_to_sqlite(
            folder=args.dir,
            db_path=args.db,
            chunksize=args.chunksize,
            limit_files=args.limit_files,
        )

        print("[2/4] Building normalized stars table (join)...")
        build_stars_table_from_gaia(args.db)
    else:
        if not args.csv:
            raise SystemExit("build requires either --csv <file> OR --dir <folder>")
        print("[1/4] Loading CSV -> SQLite (stars)...")
        load_csv_to_sqlite(args.csv, args.db, table="stars")

    print("[3/4] Building star cards...")
    create_star_cards(args.db, source_table="stars", cards_table="star_cards")

    print("[4/4] Building FAISS index...")
    build_faiss_index(
        db_path=args.db,
        index_path=args.index,
        meta_path=args.meta,
        cards_table="star_cards",
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        embed_where=args.embed_where,
        heartbeat_every_rows=args.heartbeat_every_rows,
        heartbeat_every_sec=args.heartbeat_every_sec,
    )

    sanity_check(args.db, args.index, args.meta, cards_table="star_cards", source_table="stars")

    print("Done.")
    print(f"DB:    {args.db}")
    print(f"INDEX: {args.index}")
    print(f"META:  {args.meta}")


def cmd_build_gaia(args: argparse.Namespace) -> None:
    if not args.no_clean:
        print("[0/4] Cleaning old outputs (db/index/meta)...")
        _remove_if_exists(args.db)
        _remove_if_exists(args.index)
        _remove_if_exists(args.meta)

    print("[1/4] Loading Gaia folder -> SQLite raw tables...")
    load_gaia_folder_to_sqlite(
        folder=args.dir,
        db_path=args.db,
        chunksize=args.chunksize,
        limit_files=args.limit_files,
    )

    print("[2/4] Building normalized stars table (join)...")
    build_stars_table_from_gaia(args.db)

    print("[3/4] Building star cards...")
    create_star_cards(args.db, source_table="stars", cards_table="star_cards")

    print("[4/4] Building FAISS index...")
    build_faiss_index(
        db_path=args.db,
        index_path=args.index,
        meta_path=args.meta,
        cards_table="star_cards",
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        embed_where=args.embed_where,
        heartbeat_every_rows=args.heartbeat_every_rows,
        heartbeat_every_sec=args.heartbeat_every_sec,
    )

    sanity_check(args.db, args.index, args.meta, cards_table="star_cards", source_table="stars")

    print("Done (Gaia ingest).")
    print(f"DB:    {args.db}")
    print(f"INDEX: {args.index}")
    print(f"META:  {args.meta}")


def cmd_query(args: argparse.Namespace) -> None:
    where = []
    params: List[Any] = []

    if args.max_dist is not None:
        where.append('d IS NOT NULL AND d <= ?')
        params.append(float(args.max_dist))

    if args.min_teff is not None:
        where.append('Teff IS NOT NULL AND Teff >= ?')
        params.append(float(args.min_teff))

    if args.max_teff is not None:
        where.append('Teff IS NOT NULL AND Teff <= ?')
        params.append(float(args.max_teff))

    if args.max_gmag is not None:
        where.append('GAIAmag IS NOT NULL AND GAIAmag <= ?')
        params.append(float(args.max_gmag))

    if args.max_mh is not None:
        where.append('MH IS NOT NULL AND MH <= ?')
        params.append(float(args.max_mh))
    if args.min_mh is not None:
        where.append('MH IS NOT NULL AND MH >= ?')
        params.append(float(args.min_mh))

    where_sql = " AND ".join(where) if where else None

    results = query_hybrid(
        db_path=args.db,
        index_path=args.index,
        meta_path=args.meta,
        q=args.q,
        k=args.k,
        where_sql=where_sql,
        params=tuple(params) if params else None,
        source_table="stars",
        cards_table="star_cards",
        device=args.device,
    )

    results = sort_results_for_display(results, args.sort)
    print_results_table(results, wide=args.wide, gaia_short=(not args.no_gaia_short))

    if args.details:
        for r in results:
            print_result_details(r, wide=args.wide)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build SQLite + star cards + FAISS index")
    b.add_argument("--csv", default=None, help="Single CSV input (legacy mode)")
    b.add_argument("--dir", default=None, help=r"Folder with GaiaSource_*.csv + AstrophysicalParameters_*.csv (Gaia mode)")
    b.add_argument("--db", required=True)
    b.add_argument("--index", required=True)
    b.add_argument("--meta", required=True)
    b.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    b.add_argument("--batch-size", type=int, default=1024)
    b.add_argument("--device", default="auto", help="Embedding device: auto|cpu|cuda (3060 Ti: use cuda)")
    b.add_argument("--embed-where", default="", help="Optional SQL WHERE clause to filter star_cards for embedding (reduces time)")
    b.add_argument("--heartbeat-every-rows", type=int, default=0, help="Heartbeat print every N rows (0 = auto)")
    b.add_argument("--heartbeat-every-sec", type=float, default=120.0, help="Heartbeat print every N seconds")
    b.add_argument("--chunksize", type=int, default=200_000, help="CSV chunk size for Gaia ingestion")
    b.add_argument("--limit-files", type=int, default=0, help="For testing Gaia mode: ingest only first N files of each type")
    b.add_argument("--no-clean", action="store_true", help="Do not delete old db/index/meta before building")
    b.set_defaults(func=cmd_build)

    g = sub.add_parser("build-gaia", help="Build from Gaia folder (GaiaSource_*.csv + AstrophysicalParameters_*.csv)")
    g.add_argument("--dir", default=r"D:\g", help="Folder containing GaiaSource_*.csv and AstrophysicalParameters_*.csv")
    g.add_argument("--db", required=True)
    g.add_argument("--index", required=True)
    g.add_argument("--meta", required=True)
    g.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    g.add_argument("--batch-size", type=int, default=1024)
    g.add_argument("--device", default="auto", help="Embedding device: auto|cpu|cuda (3060 Ti: use cuda)")
    g.add_argument("--embed-where", default="", help="Optional SQL WHERE clause to filter star_cards for embedding (reduces time)")
    g.add_argument("--heartbeat-every-rows", type=int, default=0, help="Heartbeat print every N rows (0 = auto)")
    g.add_argument("--heartbeat-every-sec", type=float, default=120.0, help="Heartbeat print every N seconds")
    g.add_argument("--chunksize", type=int, default=200_000, help="CSV chunk size for ingestion (RAM control)")
    g.add_argument("--limit-files", type=int, default=0, help="For testing: only ingest first N files of each type")
    g.add_argument("--no-clean", action="store_true", help="Do not delete old db/index/meta before building")
    g.set_defaults(func=cmd_build_gaia)

    q = sub.add_parser("query", help="Hybrid query")
    q.add_argument("--db", required=True)
    q.add_argument("--index", required=True)
    q.add_argument("--meta", required=True)
    q.add_argument("--q", required=True, help="Natural language query")
    q.add_argument("--k", type=int, default=10)
    q.add_argument("--max-dist", type=float, default=None, help="Max distance (pc) (d column)")
    q.add_argument("--min-teff", type=float, default=None)
    q.add_argument("--max-teff", type=float, default=None)
    q.add_argument("--max-gmag", type=float, default=None, help="Max Gaia G magnitude (GAIAmag)")
    q.add_argument("--max-mh", type=float, default=None, help="Max metallicity [M/H] (e.g., -1.0 for metal-poor)")
    q.add_argument("--min-mh", type=float, default=None, help="Min metallicity [M/H]")
    q.add_argument("--details", action="store_true", help="Print multi-line details for each hit")
    q.add_argument("--sort", default="", help="Display sort: score|gmag|kmag|dist|teff|r|mh (display only)")
    q.add_argument("--wide", action="store_true", help="Show full long IDs (e.g., GAIA) in the table")
    q.add_argument("--no-gaia-short", action="store_true", help="Do not shorten GAIA in table (same as --wide for GAIA)")
    q.add_argument("--device", default="auto", help="Query embedding device: auto|cpu|cuda")
    q.set_defaults(func=cmd_query)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
