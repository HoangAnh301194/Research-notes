#!/usr/bin/env python3
"""Read-only Kaggle provenance audit. Do not disclose private notebook titles or refs.

Reads each latest notebook source in a temporary runner directory. Never pushes,
runs, deletes, uploads artifacts, or commits Kaggle source to public GitHub.
"""
import argparse
import csv
import io
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


def command(*args):
    p = subprocess.run(["kaggle", "kernels", *args], capture_output=True,
                       text=True, check=False, timeout=120)
    if p.returncode:
        raise RuntimeError(f"Kaggle CLI operation {args[0]} failed (code {p.returncode}).")
    return p.stdout


def classify(body):
    s = body.lower()
    ctp = "ctpgesture" in s or "police_gesture_v1" in s
    ipn = "ipn-hand" in s or "ipn_hand" in s or "ipn hand" in s
    train = "train_one_epoch" in s or "optimizer.step()" in s
    if ctp and train:
        return "CTP TRAIN (+ possibly isolated test)"
    if ctp and ("continuous_raw_cache.pkl" in s or
                "full official continuous test split" in s):
        return "CTP ISOLATED + CONTINUOUS EVAL"
    if ipn and train and "videomamba" in s:
        return "IPN TRAIN"
    if ipn and ("videomamba_ipn_isolated_test.csv" in s or
                "full continuous test split" in s):
        return "IPN ISOLATED + CONTINUOUS EVAL"
    return None


def known_public_fingerprint(role, body):
    """Only print role names of files already published in public GitHub."""
    name = {
      "CTP TRAIN (+ possibly isolated test)": "videomamba_ctpgesture_v1_train.ipynb",
      "CTP ISOLATED + CONTINUOUS EVAL": "VideoMambaTinyCTPGestureFinal.ipynb",
      "IPN ISOLATED + CONTINUOUS EVAL": "videomamba_man_configsBenchmark.ipynb",
      "IPN TRAIN": "No IPN train notebook verified in public Research_Report"
    }
    return name.get(role, "Unknown")


def known_versions():
    """Try Kaggle Python API for latest version numbers; never print private ref."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi()
        api.authenticate()
        results=api.kernels_list(mine=True, page=1, page_size=100)
        versions={}
        for item in results:
            ref=getattr(item, "ref", None)
            v=(getattr(item, "currentVersionNumber", None) or
               getattr(item, "current_version_number", None))
            if ref and isinstance(v,int) and v>0:
                versions[ref]=v
        return versions
    except Exception:
        return {}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--page-size", type=int, default=50)
    args=p.parse_args()
    if not os.getenv("KAGGLE_API_TOKEN"):
        raise RuntimeError("KAGGLE_API_TOKEN missing.")
    if not 1 <= args.page_size <= 100:
        raise ValueError("Invalid page size.")
    raw=command("list","-m","-v","-p","1","--page-size",str(args.page_size))
    rows=list(csv.DictReader(io.StringIO(raw)))
    if len(rows) >= args.page_size:
        raise RuntimeError("More notebooks than one page; audit requires pagination.")
    print("Read-only provenance audit, LATEST Kaggle notebook versions only.")
    print(f"Inventory size: {len(rows)}.")
    print("CLI CSV columns: " + ", ".join(rows[0].keys()) if rows else "No notebook rows.")
    versions=known_versions()
    print(f"Version numbers resolved from Kaggle API: {len(versions)}.")
    count=Counter()
    errors=0
    extra_ipn_eval_candidates=0
    for i,row in enumerate(rows):
        ref=row.get("ref","").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", ref):
            errors+=1
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="kaggle_prov_") as d:
                command("pull",ref,"-p",d,"-m")
                meta_path=Path(d)/"kernel-metadata.json"
                meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
                source=""
                for f in sorted(Path(d).glob("*")):
                    if f.suffix == ".ipynb":
                        nb=json.loads(f.read_text(encoding="utf-8"))
                        source="\n".join(
                          "".join(c.get("source",[])) if isinstance(c.get("source"),list)
                          else str(c.get("source",""))
                          for c in nb.get("cells",[])
                        )
                        break
                    if f.suffix==".py":
                        source=f.read_text(encoding="utf-8")
                if not source:
                    continue
                role=classify(source)
                if not role:
                    source_lower=source.lower()
                    if ("ipn-hand" in source_lower or "ipn_hand" in source_lower or
                        "ipn hand" in source_lower) and "videomamba" in source_lower:
                        extra_ipn_eval_candidates+=1
                    continue
                count[role]+=1
                raw_is_private=meta.get("is_private",True)
                is_private=str(raw_is_private).lower() not in ("false","0")
                # Safety: no notebook titles or refs for private notebooks in public Actions logs.
                version=(versions.get(ref) or row.get("currentVersionNumber")
                         or row.get("current_version_number") or row.get("version") or "unknown")
                version=str(version)
                if not re.fullmatch(r"\d+",version):
                    version="unknown"
                public_ref=ref if not is_private else "[private ref hidden]"
                print(f"Match {role}: {known_public_fingerprint(role,source)}; Kaggle {public_ref}; latest version field={version}")
        except Exception:
            errors+=1
            # Deliberately do not expose errors containing private ref or notebook content.
    print("Role counts: "+json.dumps(dict(count),sort_keys=True))
    print(f"Other IPN VideoMamba candidates (not classified): {extra_ipn_eval_candidates}")
    print(f"Pull/parse errors: {errors}")
    print("WARNING: A matching latest source does not prove which earlier version produced a checkpoint or metric.")
    print("No private Kaggle notebook source, title, or ref was written to public logs or artifacts.")


if __name__=="__main__":
    main()
