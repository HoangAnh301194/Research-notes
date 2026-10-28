#!/usr/bin/env python3
"""Match public GitHub research .ipynb files to Kaggle notebooks and saved versions.

Kaggle data is read only. Print no private Kaggle notebook ref/title/source.
Not a guarantee about which version generated a checkpoint; source provenance only.
"""
import csv
import difflib
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path("Research_Report")
GIT_KEEP = ("IPN", "CTP", "VideoMamba", "videomamba", "evaluation", "train")
MAX_V = int(os.getenv("KAGGLE_MATCH_MAX_VERSION", "60"))
GAP_LIMIT = 4

def cli(*args):
    r = subprocess.run(["kaggle", "kernels", *args], capture_output=True,
                       text=True, timeout=85, check=False)
    return r.returncode == 0, r.stdout

def source_cells(nb):
    def src(cell):
        s=cell.get("source","")
        if isinstance(s,list): s="".join(s)
        return s.replace("\r\n","\n").strip()
    cells=nb.get("cells",[])
    return [src(x) for x in cells if x.get("cell_type")=="code" and src(x)], \
           [src(x) for x in cells if x.get("cell_type")=="markdown" and src(x)]

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def sig(nb):
    code,md=source_cells(nb)
    return {
        "code": [digest(x) for x in code],
        "md": [digest(x) for x in md],
        "code_exact":digest("\n--CELL--\n".join(code)),
        "all_exact":digest("\n--CELL--\n".join(code+md)),
        "n":len(code),
    }

def get_json_nb(dirpath):
    files=sorted(Path(dirpath).rglob("*.ipynb"))
    if not files: return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return None

def git_notebooks():
    result={}
    for p in sorted(ROOT.rglob("*.ipynb")):
        if not any(x in p.name for x in GIT_KEEP):
            continue
        try:
            nb=json.loads(p.read_text(encoding="utf-8"))
            s=sig(nb)
            if s["n"]>=3: result[p.as_posix()]=s
        except Exception:
            continue
    return result

def fetch(ref,version=None):
    with tempfile.TemporaryDirectory(prefix="kg_match_") as d:
        key=ref+(("/"+str(version)) if version else "")
        ok,_=cli("pull",key,"-p",d,"-m")
        if not ok:return None, None
        nb=get_json_nb(d)
        if not nb:return None,None
        meta=Path(d)/"kernel-metadata.json"
        metadata={}
        if meta.is_file():
            try: metadata=json.loads(meta.read_text(encoding="utf-8"))
            except Exception: pass
        return sig(nb),metadata

def score(a,b):
    if a["code_exact"]==b["code_exact"]:return 1.0,"exact-code"
    if a["all_exact"]==b["all_exact"]:return 1.0,"exact-all"
    if not a["code"] or not b["code"]:return 0.0,"none"
    ratio=difflib.SequenceMatcher(None,a["code"],b["code"],autojunk=False).ratio()
    return ratio,"cell-overlap"

def relevant_git(g):
    name=g.lower()
    return ("videomamba" in name or "evaluation" in name or "train_failed" in name)

def is_related(nb):
    all_cells="\n".join(source_cells(nb)[0]).lower()
    return any(k in all_cells for k in ("ctpgesture","police_gesture","ipn-hand","ipn_hand","ipn hand"))

def main():
    if not os.getenv("KAGGLE_API_TOKEN"):
        raise RuntimeError("KAGGLE_API_TOKEN missing.")
    local=git_notebooks()
    print("Version-match audit: read-only, no outputs, no private refs in logs.")
    print("Public GitHub notebook copies inspected:",len(local))
    ok,out=cli("list","-m","-v","-p","1","--page-size","100")
    if not ok:raise RuntimeError("Kaggle inventory failed")
    rows=list(csv.DictReader(io.StringIO(out)))
    if len(rows)>=100: raise RuntimeError("Too many rows; pagination required.")
    refs=sorted(set(row.get("ref","").strip() for row in rows))
    refs=[x for x in refs if re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+",x)]
    print("Kaggle notebook refs inventoried:",len(refs))
    matches=[]
    candidate_count=0
    successful_versions=0
    unknown_latest=0
    for idx,ref in enumerate(refs,1):
        latest,meta=fetch(ref)
        if not latest:
            unknown_latest+=1
            continue
        # Filter without leaking names; source read on the runner.
        candidate=any(score(latest,g)[0]>=0.50 for g in local.values())
        # Candidate scan only if latest notebook clearly matches GitHub, or
        # if the latest source still describes IPN / CTP.
        # More distant versions of unrelated latest notebooks are outside scope.
        if not candidate:
            with tempfile.TemporaryDirectory(prefix="kg_probe_") as d:
                ok,_=cli("pull",ref,"-p",d)
                nb=get_json_nb(d)
                candidate=bool(nb and is_related(nb))
        if not candidate:
            continue
        candidate_count+=1
        private=str(meta.get("is_private","true")).lower() not in ("false","0")
        display_ref=ref if not private else "[private notebook]"
        alias="K"+str(idx).zfill(2)
        print(f"Candidate {alias} ({display_ref}): scanning numbered versions 1..{MAX_V}")
        gaps=0
        found=0
        highest_success=0
        for version in range(1,MAX_V+1):
            remote,unused=fetch(ref,version)
            if remote is None:
                gaps+=1
                if gaps>=GAP_LIMIT:
                    break
                continue
            gaps=0
            highest_success=version
            successful_versions+=1
            for path,local_sig in local.items():
                value,method=score(remote,local_sig)
                if value>=0.80 and remote["n"]>=3 and local_sig["n"]>=3:
                    matches.append({"git_path":path,"alias":alias,
                      "public_ref":ref if not private else None,
                      "version":version,"score":round(value,4),
                      "method":method,"local_code_cells":local_sig["n"],
                      "remote_code_cells":remote["n"]})
                    found+=1
        print(f"Candidate {alias}: tested through v{highest_success}; matched GitHub copies {found}; stopped on {GAP_LIMIT} missing versions or limit.")
    print("Candidate notebooks checked:",candidate_count)
    print("Numbered versions successfully retrieved:",successful_versions)
    print("Notebook latest-pull errors:",unknown_latest)
    print("MATCHES (never disclose private Kaggle refs):")
    for x in sorted(matches,key=lambda z:(z["git_path"],z["alias"],z["version"])):
        print(json.dumps(x,ensure_ascii=False))
    if not matches:
        print("No >=80% code-cell matches. This does not establish no version matches.")
    print("Audit cap:",MAX_V,"versions; matches are code evidence, not proof of model training output.")

if __name__=="__main__":
    main()
