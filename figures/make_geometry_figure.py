"""
Generate the geometry-parameterization schematic used in README.md.

The figure shows the two research control knobs exposed by the effective-area
model, each mapped to a target high-fidelity study:

  (a) static throat-area parameter q  ->  inlet-contraction / unstart screening
  (b) unsteady area-breathing eps*phi(x)*sin(2 pi f t)  ->  aeroelastic intake proxy

This is an effective-area (quasi-1D) model: A(x) is a duct *area* law, not a
body-fitted wall. The plot is intentionally drawn as A(x), not as a duct
cross-section, to keep that distinction honest.

Run:
    python3 figures/make_geometry_figure.py
Writes:
    figures/geometry_parameterization.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mesh import GeometryProfile, LocalizedAreaPerturbation, PerturbedGeometryProfile


def build_perturbed(base, q, width=0.05):
    """Return a PerturbedGeometryProfile with a throat Gaussian of amplitude q."""
    pert = LocalizedAreaPerturbation(
        enabled=True, mode="throat_gaussian", amplitude=q,
        x_center=None, width=width, min_area=1.0e-4,
    )
    return PerturbedGeometryProfile(base, pert)


def main():
    base = GeometryProfile.default()
    x = np.linspace(0.0, base.L_total, 400)
    A0 = base.area(x)

    sections = [
        (0.0, base.x_throat, "Inlet\n(contraction)", "#e8f0fe"),
        (base.x_throat, base.x_comb_exit, "Combustor", "#fde7e7"),
        (base.x_comb_exit, base.L_total, "Nozzle\n(expansion)", "#e8f6ec"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    # ----- (a) static throat-area parameter q -> unstart screening -----
    ax = axes[0]
    for x0, x1, label, color in sections:
        ax.axvspan(x0, x1, color=color, zorder=0)
        ax.text(0.5 * (x0 + x1), 0.0245, label, ha="center", va="bottom",
                fontsize=9, color="#555")

    q_values = [-0.030, -0.015, 0.015, 0.030]
    cmap = plt.get_cmap("coolwarm")
    for q in q_values:
        geom = build_perturbed(base, q)
        ax.plot(x, geom.area(x), lw=1.4,
                color=cmap(0.5 + q / 0.08), alpha=0.9,
                label=f"q = {q:+.3f} m$^2$")
    ax.plot(x, A0, "k-", lw=2.6, label="baseline q = 0", zorder=5)
    ax.axvline(base.x_throat, color="0.35", ls="--", lw=0.9)
    ax.text(base.x_throat + 0.02, base.A_throat - 0.004, "throat",
            fontsize=9, color="0.25", ha="left", va="top")
    ax.text(0.44, 0.150, "q < 0: inlet contraction\n(toward unstart)",
            fontsize=8.5, color="#b2182b", ha="left", va="top")
    ax.set_title("(a)  Static throat-area parameter $q$  (ref. 1)\n"
                 "inlet contraction", fontsize=10.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("effective area  A(x; q)  [m$^2$]")
    ax.set_ylim(0.02, 0.16)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    # ----- (b) unsteady area-breathing -> aeroelastic intake proxy -----
    ax = axes[1]
    for x0, x1, label, color in sections:
        ax.axvspan(x0, x1, color=color, zorder=0)
        ax.text(0.5 * (x0 + x1), 0.0245, label, ha="center", va="bottom",
                fontsize=9, color="#555")

    eps = 0.02
    pert = LocalizedAreaPerturbation(enabled=True, amplitude=eps, width=0.05,
                                     min_area=1.0e-4)
    phi = pert.shape(x, base)
    env_hi = A0 + eps * phi
    env_lo = A0 - eps * phi
    ax.fill_between(x, env_lo, env_hi, color="#9ecae1", alpha=0.55,
                    label=r"breathing envelope  $A_0 \pm \varepsilon\,\phi(x)$")
    # explicit phase snapshots at the extremes of the breathing cycle
    ax.plot(x, env_hi, color="#3182bd", lw=1.3, ls="--",
            label=r"phase = $+\pi/2$")
    ax.plot(x, env_lo, color="#08519c", lw=1.3, ls=":",
            label=r"phase = $-\pi/2$")
    ax.plot(x, A0, "k-", lw=2.6, label="baseline", zorder=5)
    ax.axvline(base.x_throat, color="0.35", ls="--", lw=0.9)

    # inset: localization mode phi(x)
    axins = ax.inset_axes([0.60, 0.58, 0.36, 0.34])
    axins.plot(x, phi, color="#08519c", lw=1.4)
    axins.set_title(r"mode $\phi(x)$", fontsize=8)
    axins.set_xticks([])
    axins.set_yticks([0, 1])
    axins.tick_params(labelsize=7)
    axins.axvline(base.x_throat, color="0.5", ls="--", lw=0.7)

    ax.set_title(r"(b)  Unsteady area-breathing  $\varepsilon\,\phi(x)\sin(2\pi f t)$  (ref. 2)"
                 "\naeroelastic intake-ramp model", fontsize=10.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("effective area  A(x, t)  [m$^2$]")
    ax.set_ylim(0.02, 0.16)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    fig.suptitle("Effective-area parameterization of the scramjet duct "
                 "(quasi-1D model, not body-fitted)", fontsize=11.5, y=1.02)
    fig.tight_layout()
    out = REPO_ROOT / "figures" / "geometry_parameterization.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
