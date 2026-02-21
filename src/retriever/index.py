#!/usr/bin/env python3
"""
Robust index builder (fixed for pyserini CLI which expects a directory for JsonCollection).

Run:
  conda activate retriever
  python index.py
"""
import json, os, sys, subprocess, tempfile, shutil
from pathlib import Path
import gzip, bz2, lzma

INPUT_JSONL = Path("/Users/jongwonjeong/Desktop/OneDrive_UWMadison/OneDrive-UW-Madison/Research/Tool-Delay/retriever/database/wikipedia/wiki-18.jsonl")
INDEX_DIR = INPUT_JSONL.parent / "bm25_Flat"

if not INPUT_JSONL.exists():
    print(f"ERROR: input file not found: {INPUT_JSONL}", file=sys.stderr)
    sys.exit(2)

print("Input:", INPUT_JSONL)
print("Output index dir will be:", INDEX_DIR)

# --- detect compression by magic bytes ---
def detect_compression(path: Path):
    with open(path, "rb") as fh:
        sig = fh.read(6)
    if sig.startswith(b"\x1f\x8b"):
        return "gzip"
    if sig.startswith(b"BZh"):
        return "bzip2"
    if sig.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    return None

comp = detect_compression(INPUT_JSONL)
print("Detected compression:", comp)

# helper to open (text mode) based on compression/encoding
def open_text(path: Path, comp_kind, encoding, errors="strict"):
    if comp_kind == "gzip":
        return gzip.open(path, mode="rt", encoding=encoding, errors=errors)
    if comp_kind == "bzip2":
        return bz2.open(path, mode="rt", encoding=encoding, errors=errors)
    if comp_kind == "xz":
        return lzma.open(path, mode="rt", encoding=encoding, errors=errors)
    return open(path, mode="rt", encoding=encoding, errors=errors)

# try several encodings to find one that can read lines without raising UnicodeDecodeError
encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

chosen_encoding = None
for enc in encodings_to_try:
    try:
        with open_text(INPUT_JSONL, comp, enc, errors="strict") as f:
            # try to read a few lines
            for _ in range(5):
                _ = f.readline()
        chosen_encoding = enc
        break
    except UnicodeDecodeError as e:
        print(f"Encoding {enc} failed: {e}")
    except Exception as e:
        print(f"Reading with {enc} raised {type(e).__name__}: {e}")
        continue

if chosen_encoding is None:
    print("No strict encoding worked. Falling back to latin-1 with replacement (may corrupt some chars).")
    chosen_encoding = "latin-1"
    final_errors = "replace"
else:
    final_errors = "strict"

print("Using encoding:", chosen_encoding, "errors policy:", final_errors)

# --- normalize JSON lines into temp file with keys id + contents ---
temp_fd, temp_path = tempfile.mkstemp(prefix="wiki18_normalized_", suffix=".jsonl")
os.close(temp_fd)
temp_path = Path(temp_path)
count = 0
skipped = 0

with open_text(INPUT_JSONL, comp, chosen_encoding, errors=final_errors) as fin, open(temp_path, "w", encoding="utf-8") as fout:
    for i, rawline in enumerate(fin):
        line = rawline.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            if skipped < 5:
                print(f"Warning: skipping non-json line {i}: {e}")
            skipped += 1
            continue
        docid = obj.get("id") or obj.get("docid") or obj.get("wiki_id") or obj.get("page_id") or obj.get("pageId") or obj.get("title")
        if docid is None:
            docid = f"doc-{i}"
        text = obj.get("contents") or obj.get("text") or obj.get("body") or obj.get("content") or obj.get("article") or obj.get("raw") or ""
        if not text:
            pieces = []
            for k in ("title", "abstract", "summary"):
                if obj.get(k):
                    pieces.append(str(obj.get(k)))
            text = " ".join(pieces)
        out_obj = {"id": str(docid), "contents": text}
        fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        count += 1

print(f"Wrote normalized temp JSONL with {count} docs to {temp_path}  (skipped {skipped} bad lines)")

# --- pyserini expects a directory for JsonCollection. Create a temp dir and move the temp file into it.
tmp_dir = Path(tempfile.mkdtemp(prefix="pyserini_jsoncoll_"))
collection_target = tmp_dir / "collection.jsonl"
shutil.move(str(temp_path), str(collection_target))
print("Placed normalized file into directory:", tmp_dir)

pyserini_input_dir = str(tmp_dir)   # <-- pass this directory to pyserini

CMD = [
    sys.executable, "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", pyserini_input_dir,
    "--index", str(INDEX_DIR),
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "8",
    "--storePositions", "--storeDocvectors", "--storeRaw"
]

print("\nRunning pyserini index command (this may take a while)...")
print(" ".join(CMD))
try:
    subprocess.run(CMD, check=True)
except subprocess.CalledProcessError as e:
    print("\nIndexing FAILED. See pyserini output above for more details.", file=sys.stderr)
    print("Temporary normalized JSONL/dir kept at:", tmp_dir)
    sys.exit(e.returncode)

print("\nIndexing finished. Index directory:", INDEX_DIR)
print("Listing up to 30 files in index dir:")
try:
    for ent in sorted(INDEX_DIR.iterdir())[:30]:
        print(" ", ent.name)
except Exception as e:
    print("  (Could not list index dir: )", e)

# cleanup temporary normalized dir
try:
    shutil.rmtree(tmp_dir)
    print("Removed temporary normalized directory:", tmp_dir)
except Exception:
    print("Could not remove temporary directory (you can delete it manually):", tmp_dir)

print("\nDONE. You can now search with LuceneSearcher from pyserini using this index.")