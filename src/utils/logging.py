"""CSV-based results logger.

Each run writes a single row to results/all_runs.csv. Plotting code reads
this file. No experiment-tracking service required.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

RESULTS_CSV = Path("results/all_runs.csv")


def log_run(row: dict[str, Any], path: Path = RESULTS_CSV) -> None:
    """Append one row to the results CSV. Creates header on first write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
