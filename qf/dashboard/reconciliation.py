#!/usr/bin/env python3
"""
reconciliation.py

Reusable reconciliation logic for directories and files.

Functions:
- reconcile_directories(before_dir, after_dir, key, fields=None) -> str
- reconcile_files(before_file, after_file, key, fields=None) -> str

Notes:
- CSV comparison is supported: records are keyed by the provided 'key' column.
- Optional 'fields' limits the comparison to those columns; if None, all shared
  columns are compared.
- Non-CSV files fall back to a message indicating unsupported format.
"""

import os
import csv
import re
from typing import Dict, List, Optional, Tuple
from difflib import unified_diff


def _read_csv_to_dict(path: str, key: str) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Read a CSV into a dict keyed by 'key'. Returns (records, columns).

    Last occurrence of a repeated key wins.
    """
    records: Dict[str, Dict[str, str]] = {}
    columns: List[str] = []

    # Robust CSV reading: detect delimiter, handle BOM, normalize header names
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            sep = dialect.delimiter
        except Exception:
            sep = ","

        reader = csv.reader(f, delimiter=sep)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV is empty: {path}")

        # If header parsed as a single cell containing commas, try splitting by comma
        header = [h.strip().strip('"').strip("'") for h in header]
        if len(header) == 1 and "," in header[0]:
            header = [h.strip().strip('"').strip("'") for h in header[0].split(",")]
            # reset reader to use comma separator
            f.seek(0)
            reader = csv.reader(f, delimiter=",")
            try:
                next(reader)
            except StopIteration:
                pass

        columns = header

        def _norm(s: Optional[str]) -> str:
            if s is None:
                return ""
            s2 = s.strip().lower()
            # remove non-alphanumeric characters to make comparisons tolerant
            s2 = re.sub(r"[^0-9a-z]+", "", s2)
            return s2

        norm_map = { _norm(c): c for c in columns }
        key_col = None
        if key in columns:
            key_col = key
        else:
            kn = _norm(key)
            if kn in norm_map:
                key_col = norm_map[kn]
            else:
                # try case-insensitive exact match as a last resort
                for c in columns:
                    if c.strip().lower() == key.strip().lower():
                        key_col = c
                        break
        if not key_col:
            raise ValueError(f"Key column '{key}' not in CSV: {path}. Detected columns: {columns}")

        # Rewind and use DictReader with detected separator. Skip initial header row if DictReader treats it as data.
        f.seek(0)
        dreader = csv.DictReader(f, delimiter=sep)
        # Normalize fieldnames in DictReader if needed
        if dreader.fieldnames:
            # strip quotes/spaces
            dreader.fieldnames = [fn.strip().strip('"').strip("'") for fn in dreader.fieldnames]

        for row in dreader:
            k = str(row.get(key_col, "") or "")
            records[k] = {c: (row.get(c, "") or "") for c in columns}

    return records, columns


def _compare_rows(before: Dict[str, str], after: Dict[str, str], fields: Optional[List[str]]) -> Dict[str, Tuple[str, str]]:
    """Return a dict of column -> (before, after) where values differ.

    If fields is provided, only those columns are considered.
    """
    diffs: Dict[str, Tuple[str, str]] = {}
    cols = fields if fields else sorted(set(before.keys()) | set(after.keys()))
    for c in cols:
        b = before.get(c, "")
        a = after.get(c, "")
        if b != a:
            diffs[c] = (b, a)
    return diffs


def reconcile_files(before_file: str, after_file: str, key: str, fields: Optional[List[str]] = None) -> str:
    """Reconcile two files. For CSVs, compare rows by key and selected fields.

    Returns a text report.
    """
    bf = before_file
    af = after_file
    header = [
        "=== File Reconciliation ===",
        f"Before: {bf}",
        f"After:  {af}",
        f"Key:    {key}",
        f"Fields: {', '.join(fields) if fields else '(all)'}",
        "",
    ]
    # CSV case
    if bf.lower().endswith(".csv") and af.lower().endswith(".csv"):
        before, bcols = _read_csv_to_dict(bf, key)
        after, acols = _read_csv_to_dict(af, key)
        keys_before = set(before.keys())
        keys_after = set(after.keys())
        missing_in_after = sorted(list(keys_before - keys_after))
        missing_in_before = sorted(list(keys_after - keys_before))
        common = sorted(list(keys_before & keys_after))
        lines: List[str] = []
        lines.extend(header)
        lines.append(f"Records only in BEFORE: {len(missing_in_after)}")
        if missing_in_after:
            lines.append(", ".join(missing_in_after[:50]) + (" ..." if len(missing_in_after) > 50 else ""))
        lines.append(f"Records only in AFTER:  {len(missing_in_before)}")
        if missing_in_before:
            lines.append(", ".join(missing_in_before[:50]) + (" ..." if len(missing_in_before) > 50 else ""))
        lines.append(f"Common records:        {len(common)}")
        lines.append("")
        # Compare common records
        diff_count = 0
        for k in common:
            d = _compare_rows(before[k], after[k], fields)
            if d:
                diff_count += 1
                lines.append(f"- Key {k} differs in {len(d)} field(s):")
                for col, (bv, av) in d.items():
                    lines.append(f"  * {col}: BEFORE='{bv}' | AFTER='{av}'")
        lines.append("")
        lines.append(f"Summary: {diff_count} of {len(common)} common records differ.")
        return "\n".join(lines)

    # Fallback: text diff
    try:
        with open(bf, "r", encoding="utf-8", errors="ignore") as b:
            btxt = b.readlines()
        with open(af, "r", encoding="utf-8", errors="ignore") as a:
            atxt = a.readlines()
        diff = list(unified_diff(btxt, atxt, fromfile=bf, tofile=af))
        return "\n".join(header + ["(Non-CSV) Unified diff:", ""] + diff)
    except Exception as e:
        return "\n".join(header + [f"Error generating diff: {e}"])


def reconcile_directories(before_dir: str, after_dir: str, key: str, fields: Optional[List[str]] = None) -> str:
    """Reconcile CSV files present in both directories by filename.

    - Lists files only in BEFORE and only in AFTER.
    - For files present in both, runs reconcile_files and aggregates the results.
    """
    bd = os.path.abspath(before_dir)
    ad = os.path.abspath(after_dir)
    lines: List[str] = []
    lines.append("=== Directory Reconciliation ===")
    lines.append(f"Before dir: {bd}")
    lines.append(f"After dir:  {ad}")
    lines.append(f"Key:        {key}")
    lines.append(f"Fields:     {', '.join(fields) if fields else '(all)'}")
    lines.append("")

    if not os.path.isdir(bd) or not os.path.isdir(ad):
        lines.append("One or both directories do not exist.")
        return "\n".join(lines)

    before_files = {f for f in os.listdir(bd) if f.endswith(".csv")}
    after_files = {f for f in os.listdir(ad) if f.endswith(".csv")}

    only_before = sorted(list(before_files - after_files))
    only_after = sorted(list(after_files - before_files))
    common = sorted(list(before_files & after_files))

    lines.append(f"CSV files only in BEFORE: {len(only_before)}")
    if only_before:
        lines.append(", ".join(only_before))
    lines.append(f"CSV files only in AFTER:  {len(only_after)}")
    if only_after:
        lines.append(", ".join(only_after))
    lines.append(f"CSV files in both:        {len(common)}")
    lines.append("")

    diff_files = 0
    for fname in common:
        bf = os.path.join(bd, fname)
        af = os.path.join(ad, fname)
        try:
            report = reconcile_files(bf, af, key, fields)
            # Summarize per-file differences by scanning for 'Summary: '
            lines.append(f"--- {fname} ---")
            lines.append(report)
            lines.append("")
            if "Summary:" in report:
                diff_files += 1
        except Exception as e:
            lines.append(f"--- {fname} ---")
            lines.append(f"Error: {e}")
            lines.append("")

    lines.append(f"Compared {len(common)} common CSV file(s).")
    lines.append(f"Reports generated for {diff_files} CSV file(s) with differences.")
    return "\n".join(lines)
