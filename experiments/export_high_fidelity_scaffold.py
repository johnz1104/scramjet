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
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from mesh import GeometryProfile, LocalizedAreaPerturbation, PerturbedGeometryProfile
from experiments.run_static_wall_sweep import write_csv, write_json


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
        "exit_mach": "area-averaged exit Mach proxy from final x-column",
        "max_mach": "maximum cell Mach number",
        "pressure_recovery": "exit static pressure divided by freestream pressure",
        "mdot": "inlet mass-flow proxy per unit depth",
        "thrust": "momentum plus pressure thrust proxy per unit depth",
        "pressure_min": "minimum static pressure in domain",
        "pressure_max": "maximum static pressure in domain",
        "notes": [
            "All QoI are low-fidelity Python proxies. High-fidelity equivalents",
            "should be defined by the OpenFOAM/FUN3D post-processing workflow.",
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
        "status": "metadata scaffold only, not a runnable OpenFOAM case",
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


def export_case(case_dir, output_dir, n_points=300, selection_overlay=None):
    """Export one static sweep case."""
    case_dir = Path(case_dir)
    output_dir = Path(output_dir)
    config_data = load_json(case_dir / "config.json")
    config = config_data["config"]
    geometry = geometry_from_dict(config["geometry"])
    case_name = case_dir.name

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
        "source_case": str(case_dir),
        "case_name": case_name,
        "q": q_value,
        "geometry": geometry_dict,
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

    report = {
        "case_name": case_name,
        "positive_area": positive_area,
        "min_area": float(np.min(area_values)),
        "max_area": float(np.max(area_values)),
        "q_zero_matches_baseline": baseline_match,
        "complete_high_fidelity_case": False,
        "scaffold_only": True,
        "produces_runnable_openfoam_case": False,
        "produces_runnable_fun3d_case": False,
        "missing_for_runnable_case": [
            "body-fitted mesh",
            "boundary patch/tag assignment",
            "solver dictionaries or FUN3D namelists",
            "wall y+ target and mesh convergence settings",
            "turbulence model selection and inflow turbulence quantities",
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


def select_cases_from_selection_json(selection_path, sweep_root):
    """Build the list of (case_dir, selection_overlay) to export.

    The selection_cases.json from Phase 7 can contain:
      * static_sweep_candidates: rank/q -> matched directly by nearest q.
      * doe_candidates: q_offset -> matched to nearest static-sweep q
        (the static sweep is the source of geometry for the export).

    Returns:
        list of (case_dir: Path, overlay: dict).
    """
    data = load_json(selection_path)
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
        names = list(sweep_q.keys())
        q_arr = np.array([sweep_q[n] for n in names], dtype=float)
        idx = int(np.argmin(np.abs(q_arr - float(q))))
        return names[idx], float(q_arr[idx])

    out = []
    seen = set()
    static_block = data.get("static_sweep_candidates", []) or []
    for entry in static_block:
        q = entry.get("q")
        if q is None:
            continue
        name, matched_q = nearest_static_case(q)
        if name in seen:
            continue
        seen.add(name)
        overlay = {
            "source": "static_sweep_candidate",
            "rank": entry.get("rank"),
            "score": entry.get("score"),
            "requested_q": float(q),
            "matched_static_q": matched_q,
        }
        out.append((cases_root / name, overlay))

    doe_block = data.get("doe_candidates", []) or []
    for entry in doe_block:
        design = entry.get("design", {}) or {}
        q_offset = design.get("q_offset")
        if q_offset is None:
            continue
        name, matched_q = nearest_static_case(q_offset)
        if name in seen:
            continue
        seen.add(name)
        overlay = {
            "source": "doe_candidate",
            "rank": entry.get("rank"),
            "score": entry.get("score"),
            "doe_case_id": entry.get("case_id"),
            "doe_design": design,
            "requested_q_offset": float(q_offset),
            "matched_static_q": matched_q,
            "warnings": entry.get("warnings", ""),
            "note": (
                "DOE candidates are unsteady; this export uses the nearest "
                "static-sweep case as the geometry scaffold."
            ),
        }
        out.append((cases_root / name, overlay))

    if not out:
        raise ValueError(
            f"selected_cases.json contains no exportable candidates: {selection_path}",
        )
    return out, data


def export_sweep(sweep_root, output_root=None, case_names=None, n_points=300,
                  selected_cases=None):
    """Export selected cases from a completed static sweep.

    If ``selected_cases`` is provided, it overrides ``case_names`` and the
    output is filtered to the candidates selected by Phase 7.
    """
    sweep_root = Path(sweep_root)
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
            output_root / case_dir.name,
            n_points=n_points,
            selection_overlay=overlay,
        ))

    write_json(output_root / "export_summary.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_sweep": str(sweep_root),
        "selected_cases_source": str(selected_cases) if selected_cases else None,
        "status": "scaffold only",
        "produces_runnable_openfoam_case": False,
        "produces_runnable_fun3d_case": False,
        "n_cases": len(reports),
        "cases": reports,
        "selection_metadata": selection_metadata,
        "limitations": [
            "Python prototype produces effective-area scaffolds only.",
            "No body-fitted mesh, no moving wall, no RANS/LES, no reacting flow.",
            "Use with a real high-fidelity workflow to produce runnable cases.",
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
    args = parser.parse_args(argv)

    output_root, reports = export_sweep(
        sweep_root=args.sweep_root,
        output_root=args.output_root,
        case_names=args.case_names,
        n_points=args.n_points,
        selected_cases=args.selected_cases,
    )
    print(f"Export scaffold written to: {output_root}")
    print(json.dumps({"n_cases": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
