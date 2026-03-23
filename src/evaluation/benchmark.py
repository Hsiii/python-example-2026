from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def summarize_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    preferred_columns = ["model", "c_index", "ibs", "mean_auc", "notes"]
    ordered_columns = [column for column in preferred_columns if column in frame.columns]
    remaining_columns = [column for column in frame.columns if column not in ordered_columns]
    return frame[ordered_columns + remaining_columns]


def save_summary_table(results: list[dict[str, Any]], output_path: str) -> pd.DataFrame:
    summary = summarize_results(results)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    markdown_path = path.with_suffix(".md")
    markdown_path.write_text(summary.to_markdown(index=False), encoding="utf-8")
    return summary