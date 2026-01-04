RAG over Star Catalogs
This project implements a Retrieval-Augmented Generation (RAG) pipeline for querying star catalogs, with support for legacy single-CSV files and large-scale Gaia DR3 chunk folders. It loads data into SQLite, generates compact "star card" summaries, builds a FAISS embeddings index for semantic search, and enables hybrid queries combining natural language with numeric filters.
Key features:

Ingest Options:
Legacy: Single CSV file (e.g., MAST IR stars).
Gaia DR3: Folder of GaiaSource_*.csv and AstrophysicalParameters_*.csv chunks.

Data Processing: Normalizes data into a unified stars table, creates textual "star cards" for each entry.
Embeddings: Uses Sentence Transformers to embed star cards into a FAISS index for efficient similarity search.
Hybrid Retrieval: Combine semantic search (e.g., "hot metal-poor stars") with SQL filters (e.g., Teff > 8000 K, [M/H] < -1.0).
Query UI: Tabular output with sorting, optional details, and shortened IDs for readability.
Robustness: Streaming ingestion/embedding (low RAM), GPU support, auto-detection of CSV formats/headers, heartbeat progress logs.

Dependencies

Python 3.8+ (tested on 3.12)
Core: pandas, numpy, tqdm
Embeddings/Index: faiss-cpu (or faiss-gpu), sentence-transformers
Optional: CUDA-enabled PyTorch for GPU acceleration (see PyTorch installation guide).

Install via pip:
textpip install pandas numpy faiss-cpu sentence-transformers tqdm
For GPU (recommended for large datasets like Gaia):

Install CUDA-compatible PyTorch first, then the above.

No internet access required beyond initial pip installs.
Usage
The script is CLI-driven with subcommands: build, build-gaia, query.
Building the Database and Index
Legacy Single CSV
textpython rag.py build --csv mast_ir_stars.csv --db stars.db --index stars.faiss --meta stars_meta.json
Gaia DR3 Folder (Recommended for Large Datasets)
textpython rag.py build-gaia --dir "/path/to/gaia/chunks" --db gaia.db --index gaia.faiss --meta gaia_meta.json

--dir: Folder containing GaiaSource_*.csv and AstrophysicalParameters_*.csv.
GPU acceleration: Add --device cuda --batch-size 4096.
Filter embeddings: Add --embed-where "Teff IS NOT NULL OR MH IS NOT NULL" to embed only rows with useful params (speeds up build).
Testing: --limit-files 5 to process only first 5 files per type.
Chunk size: --chunksize 200000 (default; adjust for RAM control).
Heartbeat: --heartbeat-every-sec 60 for more frequent progress logs.

Querying
textpython rag.py query --db gaia.db --index gaia.faiss --meta gaia_meta.json --q "hot metal-poor stars" --k 10 --min-teff 8000 --max-mh -1.0

--q: Natural language query.
Filters: --max-dist 100 (pc), --min-teff 6000, --max-teff 10000, --max-gmag 12, --max-mh -0.5, --min-mh -2.0.
Display: --sort teff (options: score|gmag|kmag|dist|teff|r|mh), --details for expanded info, --wide for full IDs.

Example output (tabular):
textrowid  orig_ID       score    G      K     Teff   MH    R    class   dist_pc      ra     dec               GAIA     HIP
-----  ------------- -----  -----  ----- -----  ----  ---- ------ -------  ------  ------ ------------------ ------
12345  6789012345    0.95   10.2   9.5   8500   -1.2  1.5  A0      50.3     123.45  -67.89 123456789...12345  54321
...
Code Structure

Utilities: Helper functions for formatting, normalization, ID shortening.
Output Helpers: Parsing/summarizing cards, table printing, sorting.
Star Cards: Builds compact text summaries from rows.
Ingest:
Legacy: Detects CSV format, loads to SQLite.
Gaia: Streams chunk CSVs (handles ECSV metadata), joins tables.

Embeddings: Streams cards to SentenceTransformer + FAISS (GPU/CPU).
Retrieval: Hybrid SQL filter + FAISS search.
CLI: argparse for build/query.

Notes

Performance: For full Gaia DR3 (~1.8B stars), use GPU for embedding; filter with --embed-where to embed subset (e.g., stars with Teff/MH).
Customization: Edit build_star_card for different fields; adjust model in --model.
Sanity Check: Runs post-build to verify DB/cards/index integrity.
Limitations: Assumes specific Gaia column names; extend _pick_id_col for variants. No real-time updates; rebuild for new data.
