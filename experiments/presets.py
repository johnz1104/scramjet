"""
Experiment-condition presets.

Loads a JSON preset from configs/ (e.g. tusq_m585.json, ncsu_m40.json) and
builds the matching InletConfig with explicit tunnel freestream conditions,
overriding the standard-atmosphere model. Sweep scripts accept
``--preset configs/tusq_m585.json`` (path or bare name).
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver import InletConfig

CONFIG_DIR = REPO_ROOT / "configs"


def resolve_preset_path(name_or_path):
    """Accept a full path, a configs/-relative path, or a bare preset name."""
    p = Path(name_or_path)
    candidates = [
        p,
        REPO_ROOT / p,
        CONFIG_DIR / p.name,
        CONFIG_DIR / f"{p.name}.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"preset '{name_or_path}' not found; looked in {CONFIG_DIR}")


def load_preset(name_or_path):
    """Return the raw preset dict."""
    path = resolve_preset_path(name_or_path)
    with path.open() as f:
        data = json.load(f)
    data["_path"] = str(path)
    return data


def inlet_from_preset(name_or_path, Yf_inlet=0.0):
    """
    Build an InletConfig with the preset's explicit tunnel conditions.

    Prints the preset's status note so placeholder conditions are never
    used silently.
    """
    data = load_preset(name_or_path)
    inlet = data["inlet"]
    status = data.get("status", "")
    if status:
        print(f"[preset {Path(data['_path']).name}] {status}")
    return InletConfig(
        mach=float(inlet["mach"]),
        T_inf=float(inlet["T_inf_K"]),
        p_inf=float(inlet["p_inf_Pa"]),
        gamma=float(inlet.get("gamma", 1.4)),
        R_gas=float(inlet.get("R_gas", 287.0)),
        Yf_inlet=Yf_inlet,
    )


if __name__ == "__main__":
    for name in ("tusq_m585", "ncsu_m40"):
        ic = inlet_from_preset(name)
        print(f"  {name}: M={ic.mach}, T_inf={ic.T_inf:.1f} K, "
              f"p_inf={ic.p_inf:.0f} Pa, u_inf={ic.u_inf:.0f} m/s")
