"""Split Masking_to_Extraction_pipeline_(1).ipynb into two connected notebooks:
   PartA_Masking.ipynb  ->  produces masked (blackened) images
   PartB_Extraction.ipynb -> consumes masked images, extracts via OpenAI+Claude

Original cell sources are copied verbatim by index; only the install / secrets /
hand-off / input cells are newly authored. Cell sources below are RAW triple
strings (no triple-quotes inside).
"""
import nbformat as nbf

SRC = "Masking_to_Extraction_pipeline_(1).ipynb"
orig = nbf.read(SRC, as_version=4)
O = orig.cells  # original cells, by index

def src(i):
    return O[i].source

# ---------------------------------------------------------------------------
# Build PART A : Masking
# ---------------------------------------------------------------------------
nbA = nbf.v4.new_notebook()
A = []
def Amd(s): A.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def Aco(s): A.append(nbf.v4.new_code_cell(s.strip("\n")))
def Araw(i, kind):
    A.append(nbf.v4.new_code_cell(src(i)) if kind == "code"
             else nbf.v4.new_markdown_cell(src(i)))

Amd(r"""
# Part A - Masking  (Notebook 1 of 2)

Locally redacts **confidential branding** from engineering drawings *before* any
data leaves the machine for the cloud extractor.

- Detects the **company name** (text runs -> red boxes in debug) and the **logo**
  (blue box) using a local Qwen3-VL model + Tesseract OCR + PyMuPDF text layer.
- **Blackens** those regions out.
- Hands off **only the masked images** to Part B.

**Output of this notebook**
- `masked_drawings/` - the clean, blackened images (no debug overlays).
- `masked_drawings.zip` - the same set, zipped for transfer to **`PartB_Extraction.ipynb`**.

**Runtime:** needs a **GPU** (Colab T4 is enough; the VLM loads in 4-bit). No API keys required here.
""")

Amd(r"### A0 - Install masking dependencies")
Aco(r"""
# Masking stack only (local VLM + OCR + PDF). No API SDKs needed in Part A.
!apt-get -qq install -y tesseract-ocr tesseract-ocr-eng poppler-utils
!pip install -q -U transformers accelerate bitsandbytes
!pip install -q PyMuPDF rapidfuzz pytesseract opencv-python-headless
print("Masking dependencies installed.")
""")

Araw(1, "md")     # ## 1 Global config
Araw(2, "code")   # RAW_INPUT_ZIP / RAW_DRAWING_DIR / MASKED_INPUT_DIR
Araw(3, "md")     # ## 2 Unzip
Araw(4, "code")   # unzip
Araw(5, "md")     # PART A - Masking
for i in range(6, 24):   # A1..A9 (imports, helpers, logo, VLM load, identify, redact, config, run, verify)
    Araw(i, "code" if O[i].cell_type == "code" else "md")

Amd(r"""
## Hand-off - keep ONLY the masked images and zip them for Part B

The debug overlays are dropped. The clean masked set is copied into
`masked_drawings/` and zipped to `masked_drawings.zip`. On Colab the zip is also
downloaded so you can upload it into Part B (or just run Part B in the same
runtime, where it will pick up the folder automatically).
""")
Aco(r"""
import shutil, glob, os, gc, torch

STRIP_MASKED_SUFFIX = True   # rename "<stem>_masked.png" -> "<stem>.png" for the hand-off

# fresh hand-off folder
os.makedirs(MASKED_INPUT_DIR, exist_ok=True)
for _old in glob.glob(f"{MASKED_INPUT_DIR}/*"):
    os.remove(_old)

masked_files = sorted(glob.glob(f"{OUTPUT_DIR}/*_masked.png"))
debug_files  = glob.glob(f"{OUTPUT_DIR}/*_debug.png")

for _src in masked_files:
    base = os.path.basename(_src)
    if STRIP_MASKED_SUFFIX:
        base = base.replace("_masked.png", ".png")
    shutil.copy(_src, os.path.join(MASKED_INPUT_DIR, base))

print(f"Handed off {len(masked_files)} masked image(s) -> {MASKED_INPUT_DIR}")
print(f"Excluded {len(debug_files)} debug image(s) from the hand-off.")

# zip the clean set for transfer to Part B (Notebook 2)
if os.path.exists("masked_drawings.zip"):
    os.remove("masked_drawings.zip")
shutil.make_archive("masked_drawings", "zip", MASKED_INPUT_DIR)
print("Hand-off archive ready: masked_drawings.zip")

# free the local VLM from GPU memory (Part B is API-only, in the other notebook)
for _v in ("vlm_model", "vlm_processor"):
    if _v in globals():
        del globals()[_v]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("VLM unloaded; GPU memory freed.")

# on Colab, download the zip so it can be uploaded into Part B
try:
    from google.colab import files as colab_files
    colab_files.download("masked_drawings.zip")
except Exception as e:
    print("Download step skipped (not on Colab):", e)
""")

nbA["cells"] = A
nbA["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.x"}}

# ---------------------------------------------------------------------------
# Build PART B : Extraction
# ---------------------------------------------------------------------------
nbB = nbf.v4.new_notebook()
B = []
def Bmd(s): B.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def Bco(s): B.append(nbf.v4.new_code_cell(s.strip("\n")))
def Braw(i):
    B.append(nbf.v4.new_code_cell(src(i)) if O[i].cell_type == "code"
             else nbf.v4.new_markdown_cell(src(i)))

Bmd(r"""
# Part B - Extraction  (Notebook 2 of 2)

Consumes the **masked** drawings produced by `PartA_Masking.ipynb` and extracts a
structured fastener record from each sheet.

Pipeline: **GPT extractor || Claude extractor** (in parallel) -> **GPT arbiter**
reconciles them pixel-by-pixel -> color-coded **XLSX** + per-drawing audit JSON.

**Inputs**
- Masked images: either already present in `masked_drawings/` (same Colab runtime
  as Part A) **or** uploaded as `masked_drawings.zip` when prompted below.
- API keys: read from **`secret_keys.env`** (falls back to OS env vars / Colab Secrets).

**Confidentiality:** because Part A already blacked out the company name + logo,
the images sent to the OpenAI / Anthropic APIs here contain no owner branding.
""")

Bmd(r"### B-install - Extraction dependencies")
Bco(r"""
# Extraction stack only (cloud APIs + image/spreadsheet IO). No local VLM/GPU.
!apt-get -qq install -y poppler-utils          # for pdf2image
!pip install -q openai anthropic pdf2image pillow openpyxl pandas
print("Extraction dependencies installed.")
""")

Bmd(r"### B0 - API keys + clients  (loaded from secret_keys.env)")
Bco(r"""
import os

# Where to look for the shared secrets file (first hit wins).
SECRETS_FILE_CANDIDATES = [
    "secret_keys.env",
    "/content/secret_keys.env",
    os.path.join(os.getcwd(), "secret_keys.env"),
]

def _parse_env_file(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out

def load_keys(names=("OPENAI_API_KEY", "ANTHROPIC_API_KEY")):
    found = {}
    # 1) secrets file
    for p in SECRETS_FILE_CANDIDATES:
        if os.path.isfile(p):
            try:
                fk = _parse_env_file(p)
                for n in names:
                    if fk.get(n) and not fk[n].endswith("REPLACE_ME"):
                        found.setdefault(n, fk[n])
                print("Loaded keys from", p)
                break
            except Exception as e:
                print("Could not parse", p, ":", e)
    # 2) OS environment
    for n in names:
        if not found.get(n) and os.environ.get(n):
            found[n] = os.environ[n]
    # 3) Colab Secrets panel
    try:
        from google.colab import userdata
        for n in names:
            if not found.get(n):
                try:
                    v = userdata.get(n)
                    if v:
                        found[n] = v
                except Exception:
                    pass
    except Exception:
        pass
    # export so any downstream library can read them too
    for n, v in found.items():
        os.environ[n] = v
    missing = [n for n in names if not found.get(n)]
    if missing:
        print("WARNING - missing keys:", missing,
              "-> edit secret_keys.env (or set env vars / Colab Secrets).")
    return found

KEYS = load_keys()

from openai import OpenAI
from anthropic import Anthropic
client           = OpenAI(api_key=KEYS.get("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=KEYS.get("ANTHROPIC_API_KEY"))
print("OpenAI + Anthropic clients ready.")
""")

Bmd(r"### B-input - Get the masked images from Part A")
Bco(r"""
import os, glob, shutil

# Shared hand-off convention with Part A.
MASKED_INPUT_DIR = "/content/masked_drawings"
os.makedirs(MASKED_INPUT_DIR, exist_ok=True)

def _masked_present():
    return [p for p in glob.glob(f"{MASKED_INPUT_DIR}/*")
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".pdf"))]

existing = _masked_present()
if existing:
    print(f"Found {len(existing)} masked image(s) already in {MASKED_INPUT_DIR} "
          f"(same runtime as Part A).")
else:
    print("No masked images found. Upload the hand-off 'masked_drawings.zip' from Part A...")
    try:
        from google.colab import files as colab_files
        import zipfile, io
        up = colab_files.upload()                      # pick masked_drawings.zip (or loose images)
        for name, data in up.items():
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    z.extractall(MASKED_INPUT_DIR)
            else:
                with open(os.path.join(MASKED_INPUT_DIR, name), "wb") as fh:
                    fh.write(data)
    except Exception as e:
        print("Not on Colab - manually place masked images into",
              MASKED_INPUT_DIR, ":", e)
    # flatten any subfolder the zip may have created
    for p in glob.glob(f"{MASKED_INPUT_DIR}/**/*", recursive=True):
        if p.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")) and \
           os.path.dirname(p) != MASKED_INPUT_DIR:
            shutil.move(p, os.path.join(MASKED_INPUT_DIR, os.path.basename(p)))

final = _masked_present()
print(f"{len(final)} masked file(s) ready in {MASKED_INPUT_DIR}:",
      [os.path.basename(p) for p in final][:10])
""")

# original extraction cells 31..48 verbatim (B1 config through download)
for i in range(31, 49):
    Braw(i)

nbB["cells"] = B
nbB["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.x"}}

# ---------------------------------------------------------------------------
# Write + validate
# ---------------------------------------------------------------------------
for path, nb in [("PartA_Masking.ipynb", nbA), ("PartB_Extraction.ipynb", nbB)]:
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    errs = 0
    for j, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        # skip cells that contain Colab shell magics (! / %) - not valid plain Python
        if any(l.lstrip().startswith(("!", "%")) for l in c["source"].splitlines()):
            continue
        try:
            compile(c["source"], f"{path}:cell{j}", "exec")
        except SyntaxError as e:
            errs += 1
            print(f"SYNTAX ERROR {path} cell {j}: {e}")
    print(f"Wrote {path}: {len(nb['cells'])} cells, syntax errors (non-magic): {errs}")
