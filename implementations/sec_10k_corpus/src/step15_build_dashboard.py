"""Phase 9/10/11 dashboard: build a self-contained dashboard/index.html.

Input: dashboard/template.html plus the five result CSVs (data/validation_report.csv,
qa_pairs.csv, retrieval_eval.csv, gap_report.csv, privacy_report.csv).
Output: dashboard/index.html.

The template has no embedded data -- opened directly, its script falls back to a
live, cache-busted `fetch()` of the five CSVs (see the template's own comment),
which only works served over http:// (browsers block `fetch()` over file://, and a
local server also lets you edit the template and reload without re-running this
script). This script instead reads the five CSVs itself and injects them as a
`window.__DASHBOARD_DATA__` object right before the template's main script runs, so
the built file needs no server at all: open it directly from a file manager, or
double-click it, same as any other file.

Like every other output in this pipeline, the built file goes stale the moment the
CSVs it was built from change -- re-run this after re-running step09/11/12/13/14, the
same as you'd re-run step12 after step11, or step03 after the synthetic corpus
changes. Nothing here refreshes itself.

Run from the project directory (after step09, step11, step12, step13, step14):
    python -m src.step15_build_dashboard
"""

import csv
import json

from src.config import PROJECT_DIR, RAW_DIR
from src.utils import setup_logging, write_atomic


logger = setup_logging("build_dashboard")

DASHBOARD_DIR = PROJECT_DIR / "dashboard"
TEMPLATE_PATH = DASHBOARD_DIR / "template.html"
OUTPUT_PATH = DASHBOARD_DIR / "index.html"
DATA_DIR = RAW_DIR.parent

DATASETS = {
    "validation": "validation_report.csv",
    "qa": "qa_pairs.csv",
    "retrieval": "retrieval_eval.csv",
    "gap": "gap_report.csv",
    "privacy": "privacy_report.csv",
}

D3_SCRIPT_TAG = '<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>'


def load_csv(filename: str) -> list[dict]:
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    """Embed the five result CSVs into dashboard/template.html, writing dashboard/index.html."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if D3_SCRIPT_TAG not in template:
        raise SystemExit(f"Expected to find {D3_SCRIPT_TAG!r} in {TEMPLATE_PATH} to anchor the injected data")

    data = {key: load_csv(filename) for key, filename in DATASETS.items()}

    # `</script` inside a JSON string would otherwise close this script tag early
    # (the HTML parser doesn't know it's inside a string) -- none of the current
    # data contains it, but a future narrative edit could.
    data_json = json.dumps(data).replace("</script", "<\\/script")
    data_script = f"<script>\nwindow.__DASHBOARD_DATA__ = {data_json};\n</script>"

    built = template.replace(D3_SCRIPT_TAG, f"{D3_SCRIPT_TAG}\n{data_script}")
    write_atomic(OUTPUT_PATH, built)

    counts = ", ".join(f"{key}={len(rows)}" for key, rows in data.items())
    logger.info("Wrote %s (%s)", OUTPUT_PATH, counts)


if __name__ == "__main__":
    main()
