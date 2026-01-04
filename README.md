# StarRAG: RAG over Star Catalogs (Gaia DR3 + Legacy CSV)

A single-file Python CLI that turns star catalogs into a searchable **hybrid retrieval** system:

- **Structured filtering (SQLite):** numeric constraints like distance, Teff, magnitudes, metallicity, etc.
- **Semantic search (FAISS + embeddings):** natural language queries over compact “star cards”
- **Gaia DR3 folder ingest:** streams `GaiaSource_*.csv` + `AstrophysicalParameters_*.csv` into SQLite, normalizes, joins, and indexes

Designed to be **robust** against messy CSV exports (ECSV metadata blocks, delimiter hints, BOMs) and to be **memory-safe** via true streaming ingestion + embedding.

---

## Features

### 1) Two ingestion modes
**A. Legacy CSV (single file)**
- Loads a single CSV into SQLite table: `stars`

**B. Gaia DR3 folder mode (recommended)**
- Loads these chunked exports:
  - `GaiaSource_*.csv` → `gaia_source_raw`
  - `AstrophysicalParameters_*.csv` → `gaia_astro_raw`
- Builds a normalized joined table: `stars`
- Auto-detects join key across common column variants:
  - `source_id`, `sourceid`, `designation`, `id`, etc.

### 2) “Star cards” (text per row)
Builds compact, normalized text summaries for each object into:
- SQLite table: `star_cards`

Cards include:
- IDs (GAIA / HIP / 2MASS / legacy ID)
- Coordinates (RA/Dec)
- Kinematics / distance (pmRA/pmDEC, parallax, distance)
- Photometry (G/K/V/etc depending on available columns)
- Physical params (Teff, logg, [M/H], radius)
- Optional luminosity class

### 3) True streaming embedding + FAISS build
- Does **not** load all cards into RAM
- Encodes in batches and adds vectors to FAISS incrementally
- Saves:
  - FAISS index file (e.g. `gaia.faiss`)
  - Meta JSON with `model_name`, `dim`, and `id_list` mapping FAISS rows → SQLite IDs

### 4) Hybrid query pipeline
- Optional SQL filtering on `stars` (structured constraints)
- Semantic search on embeddings
- If SQL filter is present, results are intersected with filtered IDs (with a larger “probe” search)

### 5) CLI / UX improvements
- Compact aligned table output with right-aligned numeric columns
- Long IDs auto-shortened (toggle with `--wide` / `--no-gaia-short`)
- Optional display-only sorting: `--sort score|gmag|kmag|dist|teff|r|mh`
- Optional multi-line details: `--details`

### 6) Built-in sanity check
After building, validates:
- `stars` row count
- `star_cards` row count + empty card count
- Detects “bad ingestion” symptom where most cards become `"row N"`
- Confirms FAISS `ntotal` matches meta `id_list` length

---

## Project Layout (outputs)

After a successful build you typically have:

- **SQLite DB**: `gaia.db` or `stars.db`
  - `stars` (normalized / queryable table)
  - `star_cards` (ID → compact text)
  - (Gaia mode) `gaia_source_raw`, `gaia_astro_raw`
- **FAISS index**: `gaia.faiss`
- **Meta JSON**: `gaia_meta.json` (stores ID mapping + model info)

---

## Requirements

Python 3.9+ recommended.

### Core dependencies
```bash
pip install pandas numpy tqdm
