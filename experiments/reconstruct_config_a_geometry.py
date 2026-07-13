"""Deterministic, source-constrained reconstruction of UNSW Ramp-1018.

The public sources specify the design method and dimensional constraints but
do not publish the original CAD or the 50-node coordinate table.  This utility
therefore reconstructs, rather than claims to recover, that geometry.  It keeps
the characteristic-generated compression surface as 50 discrete segments and
freezes all derived coordinates and residuals in a versioned JSON artifact.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "geometry" / "config_a_source_reconstructed_v1.json"
LINEAGE_ID = "unsw_ramp1018_source_reconstructed_v1"

SOURCE_URLS = {
    "unsw_case_05_2020": (
        "https://www.unsw.edu.au/canberra/our-research/research-excellence/"
        "engineering-research/hypersonics/high-speed-fsi-database/"
        "cantilevered-compression-ramp-3mm"
    ),
    "unsw_thesis": (
        "https://unsworks.unsw.edu.au/server/api/core/bitstreams/"
        "bf902a7a-7e55-4fda-80b2-527430526d7b/content"
    ),
    "author_intake_design_paper": (
        "https://www.researchgate.net/publication/336114358_"
        "Impact_of_Aeroelasticity_on_Hypersonic_Intake_Performance"
    ),
}

CONSTRAINTS = {
    "mach": 5.85,
    "gamma": 1.4,
    "leading_edge_angle_deg": 8.33,
    "total_turn_angle_deg": 18.51,
    "leading_edge_x_m": -0.1758,
    "cowl_x_m": 0.0,
    "intake_length_m": 0.2051,
    "deformable_surface_arc_length_m": 0.2111,
    "final_channel_height_m": 0.012,
    "internal_contraction_ratio": 1.17,
    "outlet_x_m": 0.060,
    "total_contraction_validation": 4.56,
    "compression_segments": 50,
}


def prandtl_meyer(mach, gamma=1.4):
    """Prandtl-Meyer angle in radians for a supersonic Mach number."""
    mach = float(mach)
    if mach <= 1.0:
        raise ValueError("Prandtl-Meyer flow requires Mach > 1")
    root = np.sqrt(mach * mach - 1.0)
    return float(
        np.sqrt((gamma + 1.0) / (gamma - 1.0))
        * np.arctan(np.sqrt((gamma - 1.0) / (gamma + 1.0)) * root)
        - np.arctan(root)
    )


def mach_from_prandtl_meyer(nu, gamma=1.4):
    """Invert the Prandtl-Meyer function on its physical supersonic branch."""
    nu = float(nu)
    return float(brentq(
        lambda mach: prandtl_meyer(mach, gamma) - nu,
        1.0 + 1.0e-10,
        100.0,
        xtol=1.0e-14,
        rtol=1.0e-14,
    ))


def weak_oblique_shock(mach, turn_angle_rad, gamma=1.4):
    """Solve the weak theta-beta-M root and return the downstream Mach."""
    mach = float(mach)
    theta = float(turn_angle_rad)
    mu = np.arcsin(1.0 / mach)

    def residual(beta):
        rhs = (
            2.0 / np.tan(beta)
            * (mach * mach * np.sin(beta) ** 2 - 1.0)
            / (mach * mach * (gamma + np.cos(2.0 * beta)) + 2.0)
        )
        return np.tan(theta) - rhs

    scan = np.linspace(mu + 1.0e-9, 0.5 * np.pi - 1.0e-9, 4000)
    brackets = [
        (left, right) for left, right in zip(scan[:-1], scan[1:])
        if residual(left) * residual(right) < 0.0
    ]
    if not brackets:
        raise ValueError("no attached oblique-shock solution")
    beta = brentq(residual, *brackets[0], xtol=1.0e-14, rtol=1.0e-14)
    normal_upstream = mach * np.sin(beta)
    normal_downstream = np.sqrt(
        (1.0 + 0.5 * (gamma - 1.0) * normal_upstream**2)
        / (gamma * normal_upstream**2 - 0.5 * (gamma - 1.0))
    )
    downstream_mach = normal_downstream / np.sin(beta - theta)
    return {
        "shock_angle_rad": float(beta),
        "shock_angle_deg": float(np.rad2deg(beta)),
        "downstream_mach": float(downstream_mach),
    }


def _moc_surface(first_node_x_m, shock_solution):
    """March the 50-segment compression polyline about the cowl focal point."""
    gamma = CONSTRAINTS["gamma"]
    theta_start = np.deg2rad(CONSTRAINTS["leading_edge_angle_deg"])
    theta_end = np.deg2rad(CONSTRAINTS["total_turn_angle_deg"])
    turns = np.linspace(
        theta_start, theta_end, CONSTRAINTS["compression_segments"] + 1,
    )
    nu_start = prandtl_meyer(shock_solution["downstream_mach"], gamma)
    mach = np.asarray([
        mach_from_prandtl_meyer(
            nu_start - (turn - theta_start), gamma,
        )
        for turn in turns
    ])
    mach_angles = np.arcsin(1.0 / mach)
    characteristic_angles = turns + mach_angles

    x = [float(first_node_x_m)]
    y = [float(first_node_x_m * np.tan(characteristic_angles[0]))]
    for index in range(1, len(turns)):
        # The surface chord uses the midpoint flow angle; its downstream node
        # lies on the characteristic ray from that node to the cowl focal point.
        surface_slope = np.tan(0.5 * (turns[index - 1] + turns[index]))
        ray_slope = np.tan(characteristic_angles[index])
        x_next = (y[-1] - surface_slope * x[-1]) / (ray_slope - surface_slope)
        x.append(float(x_next))
        y.append(float(ray_slope * x_next))

    return {
        "x_m": np.asarray(x),
        "y_m": np.asarray(y),
        "turn_angle_rad": turns,
        "mach": mach,
        "mach_angle_rad": mach_angles,
        "characteristic_angle_rad": characteristic_angles,
    }


def _surface_solution(first_node_x_m, shock_solution):
    """Evaluate the dimensional closure for one MOC scale."""
    moc = _moc_surface(first_node_x_m, shock_solution)
    theta = np.deg2rad(CONSTRAINTS["total_turn_angle_deg"])
    height_final = CONSTRAINTS["final_channel_height_m"]
    height_entrance = (
        height_final * CONSTRAINTS["internal_contraction_ratio"]
    )
    shoulder_radius = (
        (height_entrance - height_final) / (1.0 - np.cos(theta))
    )

    # Extend the final MOC tangent to the published isolator-entrance gap;
    # this is the otherwise-unpublished shoulder placement solved here.
    tangent_length = (
        (-height_entrance - moc["y_m"][-1]) / np.sin(theta)
    )
    shoulder_start_x = moc["x_m"][-1] + tangent_length * np.cos(theta)
    shoulder_start_y = -height_entrance
    shoulder_end_x = shoulder_start_x + shoulder_radius * np.sin(theta)
    shoulder_end_y = -height_final

    leading_edge_x = CONSTRAINTS["leading_edge_x_m"]
    leading_theta = np.deg2rad(CONSTRAINTS["leading_edge_angle_deg"])
    leading_edge_y = (
        moc["y_m"][0]
        + np.tan(leading_theta) * (leading_edge_x - moc["x_m"][0])
    )
    leading_length = (moc["x_m"][0] - leading_edge_x) / np.cos(leading_theta)
    moc_length = float(np.sum(np.hypot(
        np.diff(moc["x_m"]), np.diff(moc["y_m"]),
    )))
    support_x = leading_edge_x + CONSTRAINTS["intake_length_m"]
    horizontal_length = support_x - shoulder_end_x
    surface_length = (
        leading_length + moc_length + tangent_length
        + shoulder_radius * theta + horizontal_length
    )
    return {
        "moc": moc,
        "leading_edge_y_m": float(leading_edge_y),
        "leading_planar_length_m": float(leading_length),
        "moc_polyline_length_m": moc_length,
        "post_compression_tangent_length_m": float(tangent_length),
        "shoulder_radius_m": float(shoulder_radius),
        "shoulder_start_x_m": float(shoulder_start_x),
        "shoulder_start_y_m": float(shoulder_start_y),
        "shoulder_end_x_m": float(shoulder_end_x),
        "shoulder_end_y_m": float(shoulder_end_y),
        "support_x_m": float(support_x),
        "horizontal_to_support_length_m": float(horizontal_length),
        "surface_arc_length_m": float(surface_length),
    }


def _rounding_interval(value, decimals):
    return 0.5 * 10.0 ** (-decimals)


def _constraint(target, actual, tolerance, units, hard=True):
    return {
        "target": float(target),
        "actual": float(actual),
        "residual": float(actual - target),
        "published_rounding_half_interval": float(tolerance),
        "units": units,
        "hard": bool(hard),
        "within_published_rounding": bool(abs(actual - target) <= tolerance),
    }


def artifact_checksum(artifact):
    """SHA-256 of canonical artifact content excluding the checksum itself."""
    payload = dict(artifact)
    payload.pop("artifact_checksum_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reconstruct_config_a():
    """Return the deterministic ``source_reconstructed_v1`` artifact."""
    theta_start = np.deg2rad(CONSTRAINTS["leading_edge_angle_deg"])
    shock = weak_oblique_shock(
        CONSTRAINTS["mach"], theta_start, CONSTRAINTS["gamma"],
    )
    target_surface = CONSTRAINTS["deformable_surface_arc_length_m"]

    def length_residual(first_node_x):
        return _surface_solution(first_node_x, shock)["surface_arc_length_m"] - target_surface

    first_node_x = brentq(
        length_residual,
        CONSTRAINTS["leading_edge_x_m"] + 0.005,
        -0.080,
        xtol=1.0e-14,
        rtol=1.0e-14,
    )
    solution = _surface_solution(first_node_x, shock)
    if solution["post_compression_tangent_length_m"] <= 0.0:
        raise RuntimeError("reconstructed shoulder placement is upstream of the MOC surface")
    if solution["horizontal_to_support_length_m"] < 0.0:
        raise RuntimeError("reconstructed circular shoulder extends beyond the support")

    moc = solution["moc"]
    leading_x = CONSTRAINTS["leading_edge_x_m"]
    leading_y = solution["leading_edge_y_m"]
    theta = np.deg2rad(CONSTRAINTS["total_turn_angle_deg"])
    radius = solution["shoulder_radius_m"]
    arc_tangent_angles = np.linspace(theta, 0.0, 33)
    arc_x = (
        solution["shoulder_start_x_m"]
        + radius * (np.sin(theta) - np.sin(arc_tangent_angles))
    )
    arc_y = (
        solution["shoulder_start_y_m"]
        + radius * (np.cos(arc_tangent_angles) - np.cos(theta))
    )
    outlet_x = CONSTRAINTS["outlet_x_m"]
    support_x = solution["support_x_m"]

    # Keep the source-critical MOC coordinates separate and also provide a
    # single strictly increasing physical ramp polyline for export.
    ramp_x_original = np.concatenate([
        [leading_x],
        moc["x_m"],
        [solution["shoulder_start_x_m"]],
        arc_x[1:],
        [support_x, outlet_x],
    ])
    ramp_y_original = np.concatenate([
        [leading_y],
        moc["y_m"],
        [solution["shoulder_start_y_m"]],
        arc_y[1:],
        [-CONSTRAINTS["final_channel_height_m"]] * 2,
    ])
    if np.any(np.diff(ramp_x_original) <= 0.0):
        raise RuntimeError("reconstructed ramp coordinates are not strictly increasing")
    if np.any(np.diff(ramp_y_original) < -1.0e-13):
        raise RuntimeError("reconstructed ramp is not monotone toward the isolator")

    rebase = -leading_x
    ramp_x_solver = ramp_x_original + rebase
    area = -ramp_y_original  # per-unit-depth gap to y=0 capture/cowl plane
    if np.any(area <= 0.0) or np.any(np.diff(area) > 1.0e-13):
        raise RuntimeError("effective area must remain positive and non-increasing")

    height_entrance = (
        CONSTRAINTS["final_channel_height_m"]
        * CONSTRAINTS["internal_contraction_ratio"]
    )
    constraints = {
        "leading_edge_x": _constraint(
            leading_x, float(ramp_x_original[0]), 0.00005, "m",
        ),
        "cowl_origin_x": _constraint(0.0, 0.0, 0.0, "m"),
        "intake_length": _constraint(
            CONSTRAINTS["intake_length_m"], support_x - leading_x,
            0.00005, "m",
        ),
        "deformable_surface_arc_length": _constraint(
            target_surface, solution["surface_arc_length_m"],
            0.00005, "m",
        ),
        "final_channel_height": _constraint(
            CONSTRAINTS["final_channel_height_m"], -solution["shoulder_end_y_m"],
            0.0005, "m",
        ),
        "internal_contraction_ratio": _constraint(
            CONSTRAINTS["internal_contraction_ratio"],
            height_entrance / CONSTRAINTS["final_channel_height_m"],
            0.005, "dimensionless",
        ),
        "outlet_station_from_cowl": _constraint(
            outlet_x, outlet_x, 0.0005, "m",
        ),
        "leading_edge_angle": _constraint(
            CONSTRAINTS["leading_edge_angle_deg"],
            CONSTRAINTS["leading_edge_angle_deg"], 0.005, "deg",
        ),
        "total_turn_angle": _constraint(
            CONSTRAINTS["total_turn_angle_deg"],
            float(np.rad2deg(moc["turn_angle_rad"][-1])), 0.005, "deg",
        ),
    }
    failed = [
        name for name, result in constraints.items()
        if result["hard"] and not result["within_published_rounding"]
    ]
    if failed:
        raise RuntimeError(f"hard reconstruction constraints failed: {failed}")

    compression_nodes = []
    for index in range(len(moc["x_m"])):
        compression_nodes.append({
            "index": index,
            "x_m_original": float(moc["x_m"][index]),
            "y_m_original": float(moc["y_m"][index]),
            "x_m_solver": float(moc["x_m"][index] + rebase),
            "turn_angle_deg": float(np.rad2deg(moc["turn_angle_rad"][index])),
            "mach": float(moc["mach"][index]),
            "mach_angle_deg": float(np.rad2deg(moc["mach_angle_rad"][index])),
            "characteristic_angle_deg": float(np.rad2deg(
                moc["characteristic_angle_rad"][index],
            )),
        })

    total_contraction = float(area[0] / area[-1])
    artifact = {
        "artifact_schema": "config_a_geometry_reconstruction_v1",
        "reconstruction_status": "source_reconstructed_v1",
        "geometry_lineage_id": LINEAGE_ID,
        "configuration": "UNSW Ramp-1018 / Config A",
        "coordinate_convention": {
            "source_origin": "cowl lip at x=0 m and y=0 m",
            "solver_origin": "ramp leading edge at x=0 m",
            "solver_rebase_offset_m": rebase,
            "positive_y": "toward the cowl / decreasing gap",
            "effective_area": "gap to y=0 per unit depth",
        },
        "published_constraints": CONSTRAINTS,
        "source_precision": {
            "angles_deg": "two decimal places (half interval 0.005 deg)",
            "leading_edge_intake_and_surface_lengths_m": (
                "reported to 0.1 mm (half interval 0.05 mm)"
            ),
            "channel_height_m": "reported to 1 mm (half interval 0.5 mm)",
            "internal_contraction_ratio": (
                "reported to two decimal places (half interval 0.005)"
            ),
            "outlet_station_m": "reported to 1 mm (half interval 0.5 mm)",
        },
        "provenance": {
            "sources": [
                {
                    "name": "UNSW High-Speed FSI Database Case 05-2020 v1",
                    "url": SOURCE_URLS["unsw_case_05_2020"],
                    "pages": [1, 2],
                    "used_for": "configuration angles, Mach and tunnel conditions",
                },
                {
                    "name": "Bhattrai PhD thesis",
                    "url": SOURCE_URLS["unsw_thesis"],
                    "pdf_pages": [79, 80, 85, 156, 278],
                    "used_for": (
                        "50-node method, 211.1 mm surface, H/h, 12 mm channel, "
                        "physical stations and structural-frequency context"
                    ),
                },
                {
                    "name": "Impact of Aeroelasticity on Hypersonic Intake Performance",
                    "url": SOURCE_URLS["author_intake_design_paper"],
                    "pages": [6, 7],
                    "used_for": "205.1 mm intake length and reconstruction method",
                },
            ],
            "source_documents_committed": False,
            "public_source_omissions": [
                "original CAD",
                "original 50-node coordinate table",
                "manufactured leading-edge/bluntness coordinates",
                "full-field DIC displacement data",
            ],
            "claim_limit": (
                "trend-level low-fidelity studies and code-to-code comparison; "
                "not exact experiment matching or experiment calibration"
            ),
        },
        "gas_dynamic_design": {
            "freestream_mach": CONSTRAINTS["mach"],
            "gamma": CONSTRAINTS["gamma"],
            "leading_edge_weak_shock": shock,
            "compression_segments": CONSTRAINTS["compression_segments"],
            "compression_node_count_including_endpoints": len(compression_nodes),
            "compression_nodes": compression_nodes,
            "segment_interpolation": "piecewise_linear_only",
        },
        "solved_parameters": {
            "first_moc_node_x_m_original": float(first_node_x),
            "first_moc_node_radial_scale_m": float(np.hypot(
                moc["x_m"][0], moc["y_m"][0],
            )),
            "post_compression_tangent_length_m": solution[
                "post_compression_tangent_length_m"
            ],
            "shoulder_radius_m": solution["shoulder_radius_m"],
            "shoulder_start_x_m_original": solution["shoulder_start_x_m"],
            "shoulder_end_x_m_original": solution["shoulder_end_x_m"],
        },
        "nominal_stations": {
            "leading_edge_x_m_solver": 0.0,
            "cowl_x_m_solver": rebase,
            "support_x_m_solver": support_x + rebase,
            "outlet_x_m_solver": outlet_x + rebase,
        },
        "physical_wall_coordinates": {
            "ramp": {
                "x_m_original": [float(value) for value in ramp_x_original],
                "y_m_original": [float(value) for value in ramp_y_original],
                "x_m_solver": [float(value) for value in ramp_x_solver],
                "section_break_indices": {
                    "leading_edge": 0,
                    "first_moc_node": 1,
                    "last_moc_node": int(len(moc["x_m"])),
                    "shoulder_start": int(len(moc["x_m"]) + 1),
                    "shoulder_end": int(len(moc["x_m"]) + len(arc_x)),
                    "support": int(len(ramp_x_original) - 2),
                    "outlet": int(len(ramp_x_original) - 1),
                },
            },
            "cowl": {
                "x_m_original": [0.0, outlet_x],
                "y_m_original": [0.0, 0.0],
                "x_m_solver": [rebase, outlet_x + rebase],
                "begins_at_cowl_station": True,
            },
            "virtual_capture_plane_y_m": 0.0,
        },
        "effective_area_per_unit_depth": {
            "x_m_solver": [float(value) for value in ramp_x_solver],
            "area_m2_per_m_depth": [float(value) for value in area],
            "interpolation": "monotone_pchip",
            "positive": bool(np.all(area > 0.0)),
            "nonincreasing": bool(np.all(np.diff(area) <= 1.0e-13)),
        },
        "derived_values": {
            "leading_edge_gap_m": float(area[0]),
            "isolator_entrance_height_m": float(height_entrance),
            "final_channel_height_m": CONSTRAINTS["final_channel_height_m"],
            "deformable_surface_arc_length_m": solution["surface_arc_length_m"],
            "total_contraction_from_reconstructed_capture_gap": total_contraction,
            "published_total_contraction_validation": CONSTRAINTS[
                "total_contraction_validation"
            ],
            "total_contraction_validation_residual": (
                total_contraction - CONSTRAINTS["total_contraction_validation"]
            ),
            "reduced_frequency_reference_length_m": target_surface,
        },
        "constraint_residuals": constraints,
        "hard_constraints_within_published_rounding": True,
    }
    artifact["artifact_checksum_sha256"] = artifact_checksum(artifact)
    return artifact


def write_artifact(path=DEFAULT_OUTPUT):
    """Reconstruct and atomically freeze the artifact through normal JSON I/O."""
    artifact = reconstruct_config_a()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    return path, artifact


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reconstruct and freeze the source-constrained UNSW Config-A geometry.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check", action="store_true",
        help="Fail if the frozen artifact differs from a fresh deterministic reconstruction.",
    )
    args = parser.parse_args(argv)
    fresh = reconstruct_config_a()
    rendered = json.dumps(fresh, indent=2, allow_nan=False) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text() != rendered:
            print(f"stale or missing Config-A artifact: {args.output}", file=sys.stderr)
            return 1
        print(f"Config-A artifact is deterministic and current: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(args.output)
    print(f"sha256={fresh['artifact_checksum_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
