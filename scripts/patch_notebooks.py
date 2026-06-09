"""Patch notebooks/01_masking.ipynb and 02_extraction.ipynb so that:
 * the masking OUTPUT dir == extraction INPUT dir (one shared hand-off folder),
 * raw zip + secrets are found in the new repo layout (data/raw, repo root) and on Colab,
 * secret-key NAMES stay identical everywhere (OPENAI_API_KEY / ANTHROPIC_API_KEY).
Only specific cells are rewritten; everything else is untouched.
"""
import nbformat as nbf
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

# shared helper text injected into the relevant cells
HELPERS = r"""
from pathlib import Path

def _on_colab():
    try:
        import google.colab  # noqa
        return True
    except Exception:
        return False

def _repo_root():
    for d in [Path.cwd(), *Path.cwd().resolve().parents]:
        if (d / "src").is_dir() and (d / "notebooks").is_dir():
            return d
    return Path.cwd()

def _handoff_dir():
    # masking OUTPUT == extraction INPUT (identical on both notebooks)
    d = Path("/content/masked_drawings") if _on_colab() \
        else _repo_root() / "data" / "interim" / "masked_drawings"
    d.mkdir(parents=True, exist_ok=True)
    return d
""".strip("\n")

# ---- 01_masking : config cell ----
A_CONFIG = HELPERS + "\n\n" + r"""
def _find_raw(name="datasetSampleVLM.zip"):
    for p in [Path.cwd() / name, _repo_root() / "data" / "raw" / name,
              Path("/content") / name, Path.cwd() / "data" / "raw" / name]:
        if p.is_file():
            return str(p)
    return name   # bare name -> upload to /content on Colab

RAW_INPUT_ZIP    = _find_raw()
RAW_DRAWING_DIR  = "drawings" if _on_colab() else str(_repo_root() / "data" / "interim" / "drawings")
MASKED_INPUT_DIR = str(_handoff_dir())
Path(RAW_DRAWING_DIR).mkdir(parents=True, exist_ok=True)
print("RAW_INPUT_ZIP    =", RAW_INPUT_ZIP)
print("RAW_DRAWING_DIR  =", RAW_DRAWING_DIR)
print("MASKED_INPUT_DIR =", MASKED_INPUT_DIR, " (masking output == extraction input)")
""".strip("\n")

# ---- 02_extraction : the MASKED_INPUT_DIR definition inside the input cell ----
B_INPUT_DEF_OLD = 'MASKED_INPUT_DIR = "/content/masked_drawings"'
B_INPUT_DEF_NEW = (HELPERS + "\n\n# Shared hand-off convention with Part A (identical folder).\n"
                   "MASKED_INPUT_DIR = str(_handoff_dir())")

# ---- 02_extraction : secrets candidate paths ----
B_SECRETS_OLD = (
    'SECRETS_FILE_CANDIDATES = [\n'
    '    "secret_keys.env",\n'
    '    "/content/secret_keys.env",\n'
    '    os.path.join(os.getcwd(), "secret_keys.env"),\n'
    ']'
)
B_SECRETS_NEW = (
    'from pathlib import Path as _P\n'
    'def _root():\n'
    '    for d in [_P.cwd(), *_P.cwd().resolve().parents]:\n'
    '        if (d / "src").is_dir() and (d / "notebooks").is_dir():\n'
    '            return d\n'
    '    return _P.cwd()\n'
    'SECRETS_FILE_CANDIDATES = [str(_P.cwd() / "secret_keys.env"),\n'
    '                           str(_root() / "secret_keys.env"),\n'
    '                           "/content/secret_keys.env"]'
)


def patch_a():
    p = NB_DIR / "01_masking.ipynb"
    nb = nbf.read(p, as_version=4)
    done = 0
    for c in nb.cells:
        if c.cell_type == "code" and c.source.lstrip().startswith("RAW_INPUT_ZIP"):
            c.source = A_CONFIG; done += 1
    nbf.write(nb, open(p, "w", encoding="utf-8"))
    print(f"01_masking: patched config cell ({done})")


def patch_b():
    p = NB_DIR / "02_extraction.ipynb"
    nb = nbf.read(p, as_version=4)
    n_in = n_sec = 0
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        if B_INPUT_DEF_OLD in c.source:
            c.source = c.source.replace(B_INPUT_DEF_OLD, B_INPUT_DEF_NEW); n_in += 1
        if B_SECRETS_OLD in c.source:
            c.source = c.source.replace(B_SECRETS_OLD, B_SECRETS_NEW); n_sec += 1
    nbf.write(nb, open(p, "w", encoding="utf-8"))
    print(f"02_extraction: patched input dir ({n_in}), secrets paths ({n_sec})")


patch_a()
patch_b()

# validate syntax of non-magic code cells
for name in ["01_masking.ipynb", "02_extraction.ipynb"]:
    nb = nbf.read(NB_DIR / name, as_version=4)
    errs = 0
    for j, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        if any(l.lstrip().startswith(("!", "%")) for l in c.source.splitlines()):
            continue
        try:
            compile(c.source, f"{name}:{j}", "exec")
        except SyntaxError as e:
            errs += 1; print("SYNTAX", name, j, e)
    print(f"{name}: syntax errors (non-magic) = {errs}")
