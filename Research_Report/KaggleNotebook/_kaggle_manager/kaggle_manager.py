#!/usr/bin/env python3
"""Read-only Kaggle CLI operations. Never prints notebook contents or credentials."""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REF_RE = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
VERSION_RE = re.compile(r"^[1-9][0-9]*$")


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def kaggle(*args):
    cmd = ["kaggle", "kernels", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode:
        # Do not echo CLI stderr: it can contain data from a private account.
        raise RuntimeError(f"Kaggle CLI failed (exit {result.returncode}) at '{args[0]}'.")
    return result.stdout


def list_page(page, size):
    output = kaggle("list", "-m", "-v", "-p", str(page),
                    "--page-size", str(size), "--sort-by", "dateRun")
    if not output.strip():
        return []
    reader = csv.DictReader(io.StringIO(output))
    if not reader.fieldnames or "ref" not in reader.fieldnames:
        # Some CLI versions print a plain-language message instead of CSV for
        # out-of-range pages. Treat only this known empty-page marker as EOF.
        if "no kernels found" in output.casefold():
            return []
        raise RuntimeError(f"Unexpected Kaggle CLI CSV format on page {page} (missing ref column).")
    return [dict(row) for row in reader if row.get("ref", "").strip()]


def inventory(output_dir, page_size, max_pages):
    all_rows = []
    seen = set()
    completed = False
    for page in range(1, max_pages + 1):
        rows = list_page(page, page_size)
        print(f"Kaggle inventory: read page {page}, {len(rows)} entries.")
        if not rows:
            completed = True
            break
        signature = tuple(row["ref"] for row in rows)
        if signature in seen:
            raise RuntimeError("Kaggle returned a repeated page; inventory is incomplete.")
        seen.add(signature)
        for row in rows:
            row["source_page"] = page
            all_rows.append(row)
    if not completed:
        raise RuntimeError("Page limit reached; increase --max-pages (no partial export).")

    # Keep Kaggle's CSV column names so future CLI fields are not lost.
    columns = []
    for row in all_rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    if not columns:
        columns = ["ref", "title", "source_page"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "notebooks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {"collected_at_utc": now_utc(), "listed_rows": len(all_rows),
               "unique_notebook_refs": len({row["ref"] for row in all_rows}),
               "note": "Inventory is notebook-level, NOT complete Kaggle version history."}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Inventory completed: {summary['unique_notebook_refs']} unique notebook refs.")
    print("No notebook names, refs, CSV or tokens are printed in job logs.")


def pull(output_dir, notebook_ref, version):
    if not REF_RE.fullmatch(notebook_ref):
        raise ValueError("Use an owner/notebook-slug reference, with letters/digits/_/-.")
    if version and not VERSION_RE.fullmatch(version):
        raise ValueError("Version must be a positive integer, or empty for latest.")
    requested = notebook_ref + ("/" + version if version else "")
    dest = output_dir / "notebook"
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("pull", requested, "-p", str(dest), "-m")
    items = []
    for path in sorted(dest.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(dest).as_posix()
        items.append({"path": relative, "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not items:
        raise RuntimeError("Kaggle returned no files; nothing to export.")
    manifest = {"notebook_ref": notebook_ref, "requested_version": version or "latest",
                "pulled_at_utc": now_utc(), "files": items,
                "note": "Source and metadata only; output data is not downloaded."}
    (output_dir / "pull_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Notebook pull completed ({len(items)} files); no contents printed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "inventory", "pull"], required=True)
    parser.add_argument("--notebook-ref", default="")
    parser.add_argument("--version", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=100)
    args = parser.parse_args()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("KAGGLE_API_TOKEN secret is not configured.")
    if args.mode == "check":
        list_page(1, 1)
        print("Kaggle CLI authentication: OK. Notebook details hidden.")
    elif args.mode == "inventory":
        if not 1 <= args.page_size <= 100 or args.max_pages < 2:
            raise ValueError("Invalid pagination limits.")
        inventory(args.output_dir, args.page_size, args.max_pages)
    else:
        pull(args.output_dir, args.notebook_ref.strip(), args.version.strip())


if __name__ == "__main__":
    main()
