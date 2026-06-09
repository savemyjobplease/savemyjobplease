"""Shared configuration: project paths, the masking->extraction hand-off folder,
and a single secret-key loader used across every stage of the AI RFQ project.

Key names are fixed here so they are identical everywhere (and in secret_keys.env):
    OPENAI_API_KEY, ANTHROPIC_API_KEY
"""
from __future__ import annotations
import os
from pathlib import Path

# Canonical secret-key names — the ONLY place these strings are defined.
OPENAI_KEY_NAME = "OPENAI_API_KEY"
ANTHROPIC_KEY_NAME = "ANTHROPIC_API_KEY"
KEY_NAMES = (OPENAI_KEY_NAME, ANTHROPIC_KEY_NAME)

SECRETS_FILENAME = "secret_keys.env"


def on_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def project_root() -> Path:
    """Repo root = nearest ancestor containing both 'src' and 'notebooks'."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src").is_dir() and (parent / "notebooks").is_dir():
            return parent
    return Path.cwd()


def _candidate_dirs() -> list[Path]:
    dirs = [Path.cwd(), project_root(), Path("/content")]
    dirs += list(Path.cwd().resolve().parents)[:3]
    seen, out = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def handoff_dir() -> Path:
    """The single folder where masking output == extraction input.

    Colab: /content/masked_drawings (matches the notebooks' upload target).
    Local/repo: <root>/data/interim/masked_drawings.
    """
    d = Path("/content/masked_drawings") if on_colab() \
        else project_root() / "data" / "interim" / "masked_drawings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_env_file(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_keys(names=KEY_NAMES, export: bool = True, verbose: bool = True) -> dict:
    """Resolve secret keys with priority: secret_keys.env -> OS env -> Colab Secrets.

    Placeholder values ending in 'REPLACE_ME' are ignored. Found keys are exported
    to os.environ so any downstream SDK (openai, anthropic) picks them up.
    """
    found: dict = {}

    # 1) secrets file (first existing candidate)
    for d in _candidate_dirs():
        p = d / SECRETS_FILENAME
        if p.is_file():
            try:
                fk = _parse_env_file(p)
                for n in names:
                    v = fk.get(n)
                    if v and not v.endswith("REPLACE_ME"):
                        found.setdefault(n, v)
                if verbose:
                    print(f"[keys] loaded from {p}")
                break
            except Exception as e:
                if verbose:
                    print(f"[keys] could not parse {p}: {e}")

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

    if export:
        for n, v in found.items():
            os.environ[n] = v

    missing = [n for n in names if not found.get(n)]
    if missing and verbose:
        print(f"[keys] WARNING missing: {missing} "
              f"-> edit {SECRETS_FILENAME} (copy from secret_keys.env.example)")
    return found
