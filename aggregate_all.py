#!/usr/bin/env python3
"""
aggregate_all.py
================

Combine **all** FPS pipeline data into *one* HDF5 file for simpler downstream
analysis.  It merges the capabilities of `aggregate_data.py` (time-series logs)
and `aggregate_events.py` (event CSV / JSON).

Output layout (HDF5 keys)
------------------------
- /timeseries : table with time-series metrics (timestamp-indexed)
- /events     : table with discrete events / discoveries

Usage
-----
$ python aggregate_all.py -o aggregated/fps_dataset.h5 \
                          --metrics "S(t),A_mean(t),effort(t)"

Dependencies
------------
- pandas   >= 2.0 (HDF5 support via PyTables)
- tqdm     (progress bar)
- tables   (PyTables backend for HDF5)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm
import importlib

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate FPS logs + events into one HDF5 file")
    p.add_argument("-o", "--output", default="aggregated/fps_dataset.h5", help="Output HDF5 file")
    p.add_argument(
        "--metrics",
        help="Comma-separated list of metrics to keep from time-series (keep all if absent)",
    )
    p.add_argument("--chunk-size", type=int, default=100_000, help="CSV chunk size (time-series)")
    return p.parse_args(argv)

# ---------------------------------------------------------------------------
# Time-series helpers (from logs/run_*.csv)
# ---------------------------------------------------------------------------

def find_csv_logs() -> List[Path]:
    return sorted(Path("logs").glob("run_*.csv"))


def derive_run_id(csv_path: Path) -> str:
    m = re.match(r"run_(.*)\.csv$", csv_path.name)
    return m.group(1) if m else csv_path.stem


def stream_csv(
    csv_path: Path, run_id: str, chunk_size: int, keep_metrics: Optional[List[str]]
) -> pd.DataFrame:
    usecols: Optional[List[str]] = None
    if keep_metrics is not None:
        usecols = ["t"] + keep_metrics

    dfs: List[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=usecols):
        chunk["run_id"] = run_id
        dfs.append(chunk)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ---------------------------------------------------------------------------
# Event helpers (CSV + JSON)
# ---------------------------------------------------------------------------
CSV_EVENT_RE = re.compile(r"emergence_events_run_(.*?)\.csv$")
DISCOVERY_RE = re.compile(r"discoveries_[^/]*run_(.*?)_")


def find_event_csvs() -> List[Path]:
    return list(Path("fps_pipeline_output").rglob("emergence_events_run_*.csv"))


def derive_run_id_from_event_csv(path: Path) -> str:
    m = CSV_EVENT_RE.search(path.name)
    return m.group(1) if m else path.name


def load_event_csv(path: Path, run_id: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return pd.DataFrame()

    df["run_id"] = run_id
    if "t" in df.columns and "t_start" not in df.columns:
        df = df.rename(columns={"t": "t_start"})
        df["t_end"] = pd.NA
    for col in ["t_start", "t_end", "metric", "value", "severity"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def find_event_jsons() -> List[Path]:
    return list(Path("logs").rglob("discoveries_*_parts/part_*.json"))


def derive_run_id_from_json(path: Path) -> str:
    m = DISCOVERY_RE.search(str(path.parent))
    return m.group(1) if m else path.name


def load_event_json(path: Path, run_id: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text())
    except Exception as exc:
        print(f"[WARN] Failed to parse {path}: {exc}")
        return []

    events: List[Dict[str, Any]] = []
    gamma = data.get("gamma_discoveries", {})
    for bt in gamma.get("breakthrough_moments", []):
        events.append(
            {
                "run_id": run_id,
                "event_type": bt.get("type", "breakthrough"),
                "t": bt.get("t"),
                "metric": pd.NA,
                "value": bt.get("score"),
                "severity": pd.NA,
                "details": json.dumps({k: v for k, v in bt.items() if k not in {"t", "type", "score"}}),
            }
        )
    return events

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # -------- Time-series aggregation --------------------------------------------------
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else None
    ts_frames: List[pd.DataFrame] = []
    for csv_path in tqdm(find_csv_logs(), desc="Time-series CSV", unit="file"):
        run_id = derive_run_id(csv_path)
        ts_frames.append(stream_csv(csv_path, run_id, args.chunk_size, metrics))
    df_ts = pd.concat(ts_frames, ignore_index=True) if ts_frames else pd.DataFrame()
    if "t" in df_ts.columns:
        df_ts = df_ts.rename(columns={"t": "timestamp"})
    df_ts["run_id"] = df_ts["run_id"].astype("category")

    # -------- Event aggregation --------------------------------------------------------
    event_frames: List[pd.DataFrame] = []
    for evt_csv in tqdm(find_event_csvs(), desc="Event CSV", unit="file"):
        event_frames.append(load_event_csv(evt_csv, derive_run_id_from_event_csv(evt_csv)))
    json_records: List[Dict[str, Any]] = []
    for json_path in tqdm(find_event_jsons(), desc="Event JSON", unit="file"):
        json_records.extend(load_event_json(json_path, derive_run_id_from_json(json_path)))
    if json_records:
        event_frames.append(pd.DataFrame(json_records))

    df_evt = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()

    # Ensure harmonised columns
    required_cols = [
        "run_id",
        "event_type",
        "t_start",
        "t_end",
        "t",
        "metric",
        "value",
        "severity",
        "details",
    ]
    for col in required_cols:
        if col not in df_evt.columns:
            df_evt[col] = pd.NA
    df_evt["run_id"] = df_evt["run_id"].astype("category")

    # -------- Persist everything -------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Decide persistence backend based on file extension or availability
    if out_path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        if importlib.util.find_spec("tables") is None:
            print("[WARN] PyTables not available – switching to Parquet outputs.")
        else:
            with pd.HDFStore(out_path, mode="w") as store:
                store.put("timeseries", df_ts, format="table", data_columns=["run_id", "timestamp"] if not df_ts.empty else None)
                store.put("events", df_evt, format="table", data_columns=["run_id", "event_type"] if not df_evt.empty else None)
            print(f"[INFO] Saved {len(df_ts):,} time-series rows and {len(df_evt):,} events → {out_path}")
            return

    # Fallback: write two Parquet files
    ts_path = out_path.with_suffix("")
    ts_path_ts = ts_path.with_name(ts_path.name + "_timeseries.parquet")
    ts_path_ev = ts_path.with_name(ts_path.name + "_events.parquet")
    df_ts.to_parquet(ts_path_ts, index=False)
    df_evt.to_parquet(ts_path_ev, index=False)
    print(
        f"[INFO] Saved {len(df_ts):,} time-series rows → {ts_path_ts}\n"
        f"       {len(df_evt):,} events          → {ts_path_ev}"
    )


if __name__ == "__main__":
    main() 