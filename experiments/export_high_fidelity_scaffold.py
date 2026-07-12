"""
Export scaffold metadata for later OpenFOAM/FUN3D studies.

This writes geometry and run-condition metadata from completed static sweep
cases. It does not generate complete OpenFOAM or FUN3D cases and does not run
external solvers.
"""
import argparse
import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from mesh import (
    GeometryProfile,
    LocalizedAreaPerturbation,
    PerturbedGeometryProfile,
    TabulatedAreaProfile,
)
from experiments.run_static_wall_sweep import (
    ARTIFACT_SCHEMA_VERSION,
    require_schema_v2,
    write_csv,
    write_json,
)


def load_json(path):
    """Read JSON from path."""
    with Path(path).open() as f:
        return json.load(f)


def geometry_from_dict(data):
    """Rebuild a geometry object from static sweep JSON metadata."""
    if data["type"] == "GeometryProfile":
        return GeometryProfile(
            L_inlet=data["L_inlet"],
            L_combustor=data["L_combustor"],
            L_nozzle=data["L_nozzle"],
            A_inlet=data["A_inlet"],
            A_throat=data["A_throat"],
            A_comb_exit=data["A_comb_exit"],
            A_exit=data["A_exit"],
        )

    if data["type"] == "PerturbedGeometryProfile":
        base = geometry_from_dict(data["base"])
        p = data["perturbation"]
        perturbation = LocalizedAreaPerturbation(
            enabled=p.get("enabled", True),
            mode=p.get("mode", "throat_gaussian"),
            amplitude=p.get("amplitude", 0.0),
            x_center=p.get("x_center", None),
            width=p.get("width", 0.05),
            min_area=p.get("min_area", data.get("min_area", 1.0e-6)),
        )
        return PerturbedGeometryProfile(base, perturbation)

    if data["type"] == "TabulatedAreaProfile":
        return TabulatedAreaProfile(
            data["x_samples"], data["A_samples"],
            name=data.get("name", "tabulated"),
        )

    if data["type"] == "TimeDependentPerturbedGeometryProfile":
        base = geometry_from_dict(data["base"])
        p = data["perturbation"]
        forcing = data.get("forcing", {})
        mean_q = float(forcing.get("mean", 0.0)) if forcing.get("enabled", True) else 0.0
        perturbation = LocalizedAreaPerturbation(
            enabled=p.get("enabled", True),
            mode=p.get("mode", "throat_gaussian"),
            amplitude=mean_q,
            x_center=p.get("x_center"),
            width=p.get("width", 0.05),
            min_area=p.get("min_area", data.get("min_area", 1.0e-6)),
        )
        mean_geometry = PerturbedGeometryProfile(base, perturbation)
        mean_geometry.exported_forcing_spec = forcing.copy()
        return mean_geometry

    raise ValueError(f"Unsupported geometry type for export: {data['type']}")


def area_profile_rows(geometry, n_points=300):
    """Return sampled area profile rows."""
    x = np.linspace(0.0, geometry.L_total, n_points)
    A = geometry.area(x)
    dAdx = geometry.area_gradient(x)
    return [
        {"x": float(xi), "area": float(ai), "dAdx": float(gi)}
        for xi, ai, gi in zip(x, A, dAdx)
    ]


def wall_contour_rows(geometry, n_points=300):
    """
    Return a simple per-unit-depth wall-contour suggestion.

    The current Python model uses effective area, so height = area per unit
    depth is only scaffolding for later body-fitted geometry work.
    """
    x = np.linspace(0.0, geometry.L_total, n_points)
    A = geometry.area(x)
    return [
        {
            "x": float(xi),
            "y_lower": 0.0,
            "y_upper_suggested": float(ai),
            "height_per_unit_depth": float(ai),
        }
        for xi, ai in zip(x, A)
    ]


def qoi_definitions():
    """Shared QoI metadata for low/high-fidelity comparisons."""
    return {
        "tpr": ("total-pressure recovery: mass-flux-weighted exit stagnation "
                "pressure divided by freestream stagnation pressure "
                "(primary experiment-matched QoI)"),
        "shock_x": ("dominant centerline shock location [m] from strongest "
                    "four-cell stagnation-pressure drop, refined by the "
                    "sonic crossing; null when no shock "
                    "(primary experiment-matched QoI)"),
        "exit_mach": "dy-weighted exit Mach at the last physical station (Python i=nx-2)",
        "max_mach": "maximum cell Mach number",
        "pressure_recovery": ("LEGACY static ratio p_exit/p_inf — not total-"
                              "pressure recovery; prefer tpr"),
        "mdot_prescribed": ("inlet mass-flow proxy per unit depth; prescribed by "
                            "the supersonic Dirichlet inlet, not a mass-capture QoI"),
        "mdot_exit": "exit mass flow at the same physical station",
        "mass_defect": "(mdot_exit-mdot_prescribed)/abs(mdot_prescribed)",
        "thrust": "momentum plus pressure thrust proxy per unit depth (out of paper scope)",
        "pressure_min": "minimum static pressure in domain",
        "pressure_max": "maximum static pressure in domain",
        "notes": [
            "All QoI are low-fidelity Python proxies. High-fidelity equivalents",
            "should be defined by the OpenFOAM/FUN3D post-processing workflow",
            "using these same definitions (especially tpr and shock_x).",
        ],
    }


def turbulence_notes():
    """Turbulence-readiness notes for downstream solver setup."""
    return {
        "python_solver_turbulence": "none (molecular-diffusion prototype only)",
        "suggested_high_fidelity_starting_point": "SST k-omega RANS for screening",
        "alternatives_to_consider": ["k-omega SST", "Spalart-Allmaras", "Wilcox k-omega"],
        "rationale": (
            "Python prototype provides only inviscid + molecular-diffusion modes; "
            "RANS/LES closure must be added in the high-fidelity solver."
        ),
        "wall_treatment": "select wall function or low-Re wall integration based on y+ target",
    }


def combustion_notes(config):
    """Combustion-readiness metadata."""
    physics = config.get("physics", {})
    return {
        "python_solver_combustion": (
            "enabled" if physics.get("combustion_enabled") else "disabled (cold flow)"
        ),
        "python_heat_release_model": physics.get("heat_release_model", "none"),
        "passive_scalar_enabled": physics.get("passive_scalar_enabled", False),
        "high_fidelity_recommendation": (
            "Use a finite-rate chemistry mechanism appropriate for the fuel; "
            "Python prototype combustion is reduced-order only."
        ),
        "limitations": [
            "No ignition delay or flameholding modeling in the Python prototype.",
            "No turbulence-chemistry interaction modeling.",
            "Do not use Python combustion outputs as high-fidelity initialization.",
        ],
    }


def mesh_requirements_notes(geometry_dict):
    """Mesh-requirements scaffolding for body-fitted meshing."""
    return {
        "current_python_mesh": "uniform 2D Cartesian (effective-area source, not body-fitted)",
        "recommended_high_fidelity_mesh": [
            "Body-fitted mesh using suggested_wall_contour.csv as a starting wall.",
            "Wall-normal spacing chosen to hit target y+ for the selected turbulence model.",
            "Refinement around the throat and shock train region.",
            "Outflow domain extension to avoid spurious reflections.",
        ],
        "target_y_plus_for_low_re": 1.0,
        "target_y_plus_for_wall_functions": 30.0,
        "expansion_ratio_limit_recommended": 1.2,
        "geometry_summary": geometry_dict,
    }


def freestream_inflow_notes(inlet):
    """Freestream/inflow condition metadata, including derived stagnation values."""
    gamma = float(inlet.get("gamma", 1.4))
    mach = float(inlet.get("mach", 0.0))
    T_inf = float(inlet.get("T_inf", 0.0))
    p_inf = float(inlet.get("p_inf", 0.0))
    rho_inf = float(inlet.get("rho_inf", 0.0))
    u_inf = float(inlet.get("u_inf", 0.0))
    stagnation = {}
    if mach > 0.0:
        factor = 1.0 + 0.5 * (gamma - 1.0) * mach * mach
        stagnation = {
            "T0": float(T_inf * factor),
            "p0": float(p_inf * factor**(gamma / (gamma - 1.0))),
            "rho0": float(rho_inf * factor**(1.0 / (gamma - 1.0))),
        }
    return {
        "freestream": {
            "mach": mach, "altitude": inlet.get("altitude"),
            "T_inf": T_inf, "p_inf": p_inf, "rho_inf": rho_inf, "u_inf": u_inf,
            "gamma": gamma, "R_gas": inlet.get("R_gas"),
        },
        "derived_stagnation": stagnation,
        "boundary_condition_notes": [
            "Specify supersonic inflow (rho, u, p) consistent with the Python freestream.",
            "Outlet supersonic extrapolation is sufficient for cold-flow screening; for ",
            "  reacting/subsonic-trapped flow use pressure-outlet or characteristic BCs.",
            "Side walls map to slip in Python; replace with body-fitted no-slip walls in CFD.",
        ],
    }


def openfoam_metadata(case_name, config, selection_overlay=None):
    """OpenFOAM-oriented placeholder metadata."""
    return {
        "case_name": case_name,
        "status": ("OpenFOAM screening template with executable shared-QoI bridge; "
                   "not yet meshed/run/validated"),
        "recommended_future_solver_family": "compressible RANS/URANS as appropriate",
        "suggested_turbulence_placeholder": "SST k-omega for later high-fidelity RANS screening",
        "boundaries": {
            "inlet": "supersonic inflow/freestream state from config.json",
            "outlet": "supersonic or wave-transmissive outlet, to be selected in OpenFOAM setup",
            "walls": "body-fitted no-slip adiabatic/isothermal wall definitions required later",
        },
        "mesh_notes": [
            "Current CSV contour is derived from effective area per unit depth.",
            "Generate a body-fitted mesh with wall-normal spacing set by target y+.",
            "This Python export does not include dynamic mesh or moving-wall setup.",
        ],
        "inlet": config["inlet"],
        "qoi_definitions_file": "qoi_definitions.json",
        "selection_overlay": selection_overlay or {},
    }


def fun3d_metadata(case_name, config, selection_overlay=None):
    """FUN3D-oriented placeholder metadata."""
    return {
        "case_name": case_name,
        "status": "metadata scaffold only, not a runnable FUN3D case",
        "geometry_summary": config["geometry"],
        "run_conditions": config["inlet"],
        "boundary_tags": {
            "inflow": "placeholder",
            "outflow": "placeholder",
            "viscous_walls": "placeholder",
        },
        "notes": [
            "Boundary tags and mesh files must be generated by the high-fidelity workflow.",
            "Use this metadata to preserve q, area profile, and QoI definitions.",
        ],
        "qoi_definitions_file": "qoi_definitions.json",
        "selection_overlay": selection_overlay or {},
    }


def export_case(case_dir, output_dir, n_points=300, selection_overlay=None,
                emit_gmsh=True, nx_cells=200, ny_cells=60):
    """Export one static sweep case (+ optional Gmsh/OpenFOAM emission)."""
    case_dir = Path(case_dir)
    output_dir = Path(output_dir)
    config_data = load_json(case_dir / "config.json")
    config = config_data["config"]
    geometry = geometry_from_dict(config["geometry"])
    case_name = ((selection_overlay or {}).get("export_name")
                 or case_dir.name)

    area_rows = area_profile_rows(geometry, n_points=n_points)
    contour_rows = wall_contour_rows(geometry, n_points=n_points)
    area_values = np.array([row["area"] for row in area_rows])
    positive_area = bool(np.all(area_values > 0.0))

    base = getattr(geometry, "base_geometry", geometry)
    q_value = config_data.get("q", None)
    baseline_match = None
    if q_value is not None and abs(float(q_value)) == 0.0:
        x = np.array([row["x"] for row in area_rows])
        baseline_match = bool(np.allclose(geometry.area(x), base.area(x)))

    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_dict = config["geometry"]
    write_json(output_dir / "geometry_parameters.json", {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_case": str(case_dir),
        "case_name": case_name,
        "q": q_value,
        "geometry": geometry_dict,
        "mean_geometry_exported": geometry_dict.get("type") == "TimeDependentPerturbedGeometryProfile",
        "forcing_spec": geometry_dict.get("forcing"),
        "selection_overlay": selection_overlay or {},
    })
    write_csv(output_dir / "area_profile.csv", area_rows)
    write_csv(output_dir / "suggested_wall_contour.csv", contour_rows)
    write_json(output_dir / "openfoam_metadata.json",
               openfoam_metadata(case_name, config, selection_overlay))
    write_json(output_dir / "fun3d_metadata.json",
               fun3d_metadata(case_name, config, selection_overlay))
    write_json(output_dir / "qoi_definitions.json", qoi_definitions())
    write_json(output_dir / "turbulence_notes.json", turbulence_notes())
    write_json(output_dir / "combustion_notes.json", combustion_notes(config))
    write_json(output_dir / "mesh_requirements.json",
               mesh_requirements_notes(geometry_dict))
    write_json(output_dir / "freestream_conditions.json",
               freestream_inflow_notes(config["inlet"]))

    gmsh_report = {}
    if emit_gmsh:
        from experiments.gmsh_openfoam_export import emit_gmsh_openfoam
        geo_path, foam_case = emit_gmsh_openfoam(
            output_dir, contour_rows, config["inlet"],
            nx_cells=nx_cells, ny_cells=ny_cells)
        gmsh_report = {
            "gmsh_geo": str(geo_path.relative_to(output_dir)),
            "openfoam_case": str(foam_case.relative_to(output_dir)),
            "qoi_bridge": str((foam_case / "postprocess_qoi.py").relative_to(output_dir)),
            "notes": [
                "Structured transfinite quad mesh, one-cell z-extrusion,",
                "named Physical groups; build with 'gmsh -3 -format msh2'.",
                "openfoam_case/ is a rhoCentralFoam TEMPLATE (inviscid, slip",
                "walls, laminar) — review against your pinned version and",
                "run openfoam_case/Allrun.sh inside its environment.",
            ],
        }

    report = {
        "case_name": case_name,
        "positive_area": positive_area,
        "min_area": float(np.min(area_values)),
        "max_area": float(np.max(area_values)),
        "q_zero_matches_baseline": baseline_match,
        "complete_high_fidelity_case": False,
        "scaffold_only": not emit_gmsh,
        "produces_runnable_openfoam_case": emit_gmsh,
        "produces_runnable_fun3d_case": False,
        "gmsh_openfoam": gmsh_report,
        "remaining_manual_steps": [
            "review solver dictionaries against the pinned OpenFOAM version",
            "grid-convergence study (3 mesh levels) before quoting results",
            "wall y+ target + no-slip walls + turbulence model for viscous runs",
            "fuel/oxidizer chemistry definition for reacting cases",
            "dynamic-mesh / ALE setup if oscillating-wall studies are required",
        ],
        "selection_overlay": selection_overlay or {},
    }
    write_json(output_dir / "export_report.json", report)
    if not positive_area:
        raise ValueError(f"Exported area profile is not positive for {case_name}")
    if baseline_match is False:
        raise ValueError(f"q=0 export does not match baseline geometry for {case_name}")
    return report


def selected_case_dirs(sweep_root, case_names=None):
    """Return selected case directories from a static sweep."""
    cases_root = Path(sweep_root) / "cases"
    if case_names:
        return [cases_root / name for name in case_names]
    return sorted([p for p in cases_root.iterdir() if p.is_dir()])


def _collect_sweep_q_values(sweep_root):
    """Return {case_name: q} from a static-sweep cases directory."""
    sweep_root = Path(sweep_root)
    mapping = {}
    cases_root = sweep_root / "cases"
    if not cases_root.is_dir():
        return mapping
    for case_dir in sorted(cases_root.iterdir()):
        if not case_dir.is_dir():
            continue
        cfg_path = case_dir / "config.json"
        if not cfg_path.is_file():
            continue
        data = load_json(cfg_path)
        q = data.get("q")
        if q is None:
            continue
        mapping[case_dir.name] = float(q)
    return mapping


def match_static_case(sweep_q, requested_q, q_match_tol=1.0e-8,
                      allow_nearest=False):
    """Return the nearest static case, enforcing an explicit mismatch policy."""
    names = list(sweep_q.keys())
    q_arr = np.array([sweep_q[name] for name in names], dtype=float)
    idx = int(np.argmin(np.abs(q_arr - float(requested_q))))
    mismatch = abs(float(q_arr[idx]) - float(requested_q))
    if mismatch > float(q_match_tol) and not allow_nearest:
        available = ", ".join(f"{value:.8g}" for value in sorted(q_arr))
        raise ValueError(
            f"requested q={float(requested_q):.8g} has nearest mismatch "
            f"{mismatch:.3g} > tolerance {q_match_tol:.3g}; "
            f"available static q values: [{available}]. Use "
            "--allow-nearest only for an explicitly approximate handoff."
        )
    warning = None
    if mismatch > float(q_match_tol):
        warning = (
            f"ALLOW-NEAREST: requested q={float(requested_q):.8g}, matched "
            f"q={float(q_arr[idx]):.8g}, mismatch={mismatch:.3g}"
        )
        warnings.warn(warning, RuntimeWarning)
    return names[idx], float(q_arr[idx]), mismatch, warning


def select_cases_from_selection_json(selection_path, sweep_root,
                                     q_match_tol=1.0e-8,
                                     allow_nearest=False):
    """Build the list of (case_dir, selection_overlay) to export.

    The selection_cases.json from Phase 7 can contain:
      * static_sweep_candidates: rank/q -> matched directly by nearest q.
      * doe_candidates: q_offset -> matched to nearest static-sweep q
        (the static sweep is the source of geometry for the export).

    Returns:
        list of (case_dir: Path, overlay: dict).
    """
    data = load_json(selection_path)
    if data.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"selection file must use schema_version={ARTIFACT_SCHEMA_VERSION}; "
            f"got {data.get('schema_version')!r}"
        )
    sweep_root = Path(sweep_root)
    cases_root = sweep_root / "cases"
    if not cases_root.is_dir():
        raise FileNotFoundError(
            f"sweep root has no cases/ directory: {cases_root}",
        )

    sweep_q = _collect_sweep_q_values(sweep_root)
    if not sweep_q:
        raise ValueError(
            f"could not collect any q values from sweep cases under {cases_root}",
        )

    def nearest_static_case(q):
        return match_static_case(
            sweep_q, q, q_match_tol=q_match_tol,
            allow_nearest=allow_nearest,
        )

    out = []
    doe_block = data.get("doe_candidates", []) or []
    for entry in doe_block:
        design = entry.get("design", {}) or {}
        q_offset = design.get("q_offset")
        if q_offset is None:
            continue
        name, matched_q, mismatch, match_warning = nearest_static_case(q_offset)
        doe_case_id = entry.get("case_id") or f"doe_rank_{entry.get('rank')}"
        overlay = {
            "source": "doe_candidate",
            "export_name": f"{doe_case_id}__{name}",
            "rank": entry.get("rank"),
            "score": entry.get("score"),
            "doe_case_id": entry.get("case_id"),
            "doe_design": design,
            "requested_q_offset": float(q_offset),
            "matched_static_q": matched_q,
            "q_match_mismatch": mismatch,
            "q_match_warning": match_warning,
            "warnings": entry.get("warnings", ""),
            "note": (
                "DOE candidates are unsteady; this export uses the nearest "
                "static-sweep case as the geometry scaffold."
            ),
        }
        out.append((cases_root / name, overlay))

    static_block = data.get("static_sweep_candidates", []) or []
    for entry in static_block:
        q = entry.get("q")
        if q is None:
            continue
        name, matched_q, mismatch, match_warning = nearest_static_case(q)
        overlay = {
            "source": "static_sweep_candidate",
            "export_name": f"static_rank_{entry.get('rank')}__{name}",
            "rank": entry.get("rank"),
            "score": entry.get("score"),
            "requested_q": float(q),
            "matched_static_q": matched_q,
            "q_match_mismatch": mismatch,
            "q_match_warning": match_warning,
        }
        out.append((cases_root / name, overlay))

    if not out:
        raise ValueError(
            f"selected_cases.json contains no exportable candidates: {selection_path}",
        )
    return out, data


def export_sweep(sweep_root, output_root=None, case_names=None, n_points=300,
                 emit_gmsh=True, nx_cells=200, ny_cells=60,
                 selected_cases=None, q_match_tol=1.0e-8,
                 allow_nearest=False):
    """Export selected cases from a completed static sweep.

    If ``selected_cases`` is provided, it overrides ``case_names`` and the
    output is filtered to the candidates selected by Phase 7.
    """
    sweep_root = Path(sweep_root)
    if not sweep_root.is_absolute():
        sweep_root = REPO_ROOT / sweep_root
    require_schema_v2(sweep_root, "static sweep")
    if output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "runs" / f"high_fidelity_export_scaffold_{stamp}"
    else:
        output_root = Path(output_root)
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    selection_metadata = None
    case_overlays = None
    if selected_cases:
        case_overlays, selection_metadata = select_cases_from_selection_json(
            selected_cases, sweep_root,
            q_match_tol=q_match_tol, allow_nearest=allow_nearest,
        )
        cases_iter = case_overlays
    else:
        cases_iter = [(p, None) for p in selected_case_dirs(sweep_root, case_names)]

    reports = []
    for case_dir, overlay in cases_iter:
        if not case_dir.is_dir():
            continue
        reports.append(export_case(
            case_dir,
            output_root / ((overlay or {}).get("export_name") or case_dir.name),
            n_points=n_points,
            selection_overlay=overlay,
            emit_gmsh=emit_gmsh,
            nx_cells=nx_cells,
            ny_cells=ny_cells,
        ))

    write_json(output_root / "export_summary.json", {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_sweep": str(sweep_root),
        "selected_cases_source": str(selected_cases) if selected_cases else None,
        "status": "scaffold + gmsh/openfoam templates" if emit_gmsh else "scaffold only",
        "produces_runnable_openfoam_case": emit_gmsh,
        "produces_runnable_fun3d_case": False,
        "n_cases": len(reports),
        "cases": reports,
        "selection_metadata": selection_metadata,
        "limitations": [
            "Designed for apples-to-apples comparison; the loop is not closed:",
            "no exported case has been meshed, run, and postprocessed in this repo.",
            "Emitted OpenFOAM cases are inviscid slip-wall screening templates:",
            "no moving wall, no RANS/LES, no reacting flow — review dictionaries",
            "against the pinned OpenFOAM version and add viscous/turbulence setup",
            "for the Phase 2 studies.",
        ],
    })
    return output_root, reports


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export high-fidelity metadata scaffolds")
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--case-names", nargs="*", default=None)
    parser.add_argument("--selected-cases", default=None,
                        help=("Path to selected_cases.json from rank_candidate_cases.py; "
                              "filters the export to ranked candidates."))
    parser.add_argument("--n-points", type=int, default=300)
    parser.add_argument("--no-gmsh", action="store_true",
                        help="Skip the Gmsh .geo + OpenFOAM case emission.")
    parser.add_argument("--nx-cells", type=int, default=200,
                        help="Structured mesh streamwise cell count.")
    parser.add_argument("--ny-cells", type=int, default=60,
                        help="Structured mesh wall-normal cell count.")
    parser.add_argument("--q-match-tol", type=float, default=1.0e-8)
    parser.add_argument("--allow-nearest", action="store_true",
                        help="Permit an out-of-tolerance q match with a loud warning.")
    args = parser.parse_args(argv)

    output_root, reports = export_sweep(
        sweep_root=args.sweep_root,
        output_root=args.output_root,
        case_names=args.case_names,
        n_points=args.n_points,
        selected_cases=args.selected_cases,
        emit_gmsh=not args.no_gmsh,
        nx_cells=args.nx_cells,
        ny_cells=args.ny_cells,
        q_match_tol=args.q_match_tol,
        allow_nearest=args.allow_nearest,
    )
    print(f"Export scaffold written to: {output_root}")
    print(json.dumps({"n_cases": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
