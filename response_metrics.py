"""
Response metric extraction for unsteady effective-area forcing runs.

This is a reduced-fidelity workflow helper. Given time histories of the
sinusoidal forcing q(t) and selected QoI time series, it returns scalar
metrics (mean, amplitude, phase lag) for trend studies and surrogate
fitting. It does not claim to be a frequency-resolved system-identification
tool, and it deliberately reports null/NaN for phase lag when the data are
too short, too noisy, or have zero forcing amplitude.

The extractor is reused by:
    experiments/run_unsteady_area_breathing.py
    experiments/run_parametric_unsteady_doe.py
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional

import numpy as np


DEFAULT_QOI_KEYS = ("exit_mach", "max_mach", "pressure_recovery", "mdot", "thrust")
DEFAULT_PROBE_FIELDS = ("pressure",)


def _rows_to_array(rows: Iterable[dict], key: str) -> np.ndarray:
    """Pull `key` from each dict; return float array."""
    return np.array([float(r[key]) for r in rows], dtype=float)


def _drop_transient(rows: List[dict], discard_fraction: float) -> List[dict]:
    """Return rows after dropping the first ``discard_fraction`` of them."""
    if discard_fraction <= 0.0:
        return list(rows)
    if discard_fraction >= 1.0:
        return []
    start = int(len(rows) * float(discard_fraction))
    return list(rows[start:])


def _safe_lstsq(basis: np.ndarray, y: np.ndarray):
    """Solve least squares without crashing on rank-deficient inputs."""
    coeff, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)
    return coeff


def _amplitude_phase_from_coeffs(coeff_sin: float, coeff_cos: float):
    """Convert sin/cos fit to amplitude and phase angle."""
    amplitude = float(math.hypot(coeff_sin, coeff_cos))
    if amplitude == 0.0:
        return 0.0, 0.0
    phase = float(math.atan2(coeff_cos, coeff_sin))
    return amplitude, phase


def _is_flat(values: np.ndarray, rel_tol: float = 1.0e-9) -> bool:
    """Detect a numerically constant signal."""
    if values.size == 0:
        return True
    spread = float(np.max(values) - np.min(values))
    scale = max(float(np.max(np.abs(values))), 1.0)
    return spread <= rel_tol * scale


def _phase_lag_is_supported(forcing_amplitude: float, n_cycles: float,
                            n_samples: int, min_cycles: float, min_samples: int,
                            forcing_flat: bool, response_flat: bool):
    """Return (supported, reason)."""
    if forcing_flat or forcing_amplitude <= 1.0e-12:
        return False, "epsilon=0 or numerically flat forcing"
    if response_flat:
        return False, "response time history is numerically flat"
    if n_cycles < min_cycles:
        return False, f"insufficient cycles: {n_cycles:.2f} < {min_cycles}"
    if n_samples < min_samples:
        return False, f"insufficient samples: {n_samples} < {min_samples}"
    return True, ""


def _phase_lag_metric(t: np.ndarray, y: np.ndarray, omega: float,
                      forcing_phase: float, forcing_amplitude: float,
                      forcing_is_flat: bool, n_cycles: float,
                      min_cycles: float, min_samples: int):
    """Estimate phase lag of y(t) relative to q(t).

    Returns (amplitude, mean, phase_lag_rad, warning_or_empty).
    Phase lag is None when not supported.
    """
    basis = np.column_stack([np.sin(omega * t), np.cos(omega * t), np.ones_like(t)])
    coeff = _safe_lstsq(basis, y)
    amplitude, phase = _amplitude_phase_from_coeffs(float(coeff[0]), float(coeff[1]))
    mean = float(coeff[2])
    response_flat = _is_flat(y) or amplitude <= 1.0e-14 * max(abs(mean), 1.0)
    supported, reason = _phase_lag_is_supported(
        forcing_amplitude=forcing_amplitude,
        n_cycles=n_cycles, n_samples=len(t),
        min_cycles=min_cycles, min_samples=min_samples,
        forcing_flat=forcing_is_flat, response_flat=response_flat,
    )
    if not supported:
        return amplitude, mean, None, reason
    return amplitude, mean, float(phase - forcing_phase), ""


def _probe_pressure_field_names(probe_rows: List[dict]) -> List[str]:
    """Return all field names ending with ``_pressure``."""
    if not probe_rows:
        return []
    return [k for k in probe_rows[0].keys() if k.endswith("_pressure")]


def extract_response_metrics(qoi_rows: List[dict],
                              forcing_rows: List[dict],
                              probe_rows: Optional[List[dict]] = None,
                              frequency_hz: float = 0.0,
                              discard_fraction: float = 0.25,
                              qoi_keys: Iterable[str] = DEFAULT_QOI_KEYS,
                              min_cycles: float = 1.0,
                              min_samples: int = 8) -> dict:
    """Extract mean/amplitude/phase-lag metrics from unsteady histories.

    Args:
        qoi_rows: list of dicts with keys "time" + QoI fields.
        forcing_rows: list of dicts with keys "time" and "q".
        probe_rows: optional list with "*_pressure" entries.
        frequency_hz: forcing frequency [Hz]; <=0 disables sinusoidal fitting.
        discard_fraction: fraction of leading samples treated as transient.
        qoi_keys: QoI field names to extract from ``qoi_rows``.
        min_cycles: minimum number of forcing cycles required for phase lag.
        min_samples: minimum post-transient sample count for phase lag.

    Returns:
        dict with structure:
            {
              "transient_discard_fraction": float,
              "n_samples_after_transient": int,
              "n_cycles_after_transient": float,
              "forcing": {"mean": ..., "amplitude": ..., "phase_rad": ...},
              "qoi": {<key>: {"mean", "amplitude", "phase_lag_vs_q_rad", "warning"}},
              "probes": {<name>: {"pressure_mean", "pressure_amplitude",
                                  "pressure_phase_lag_vs_q_rad", "warning"}},
              "warnings": [...],
              "notes": [...],
              "stability": {"qoi_finite": bool, "forcing_finite": bool}
            }
        Phase-lag entries are None when not robustly supported.
    """
    qoi_keys = list(qoi_keys)
    warnings: List[str] = []
    notes: List[str] = []

    base = {
        "transient_discard_fraction": float(discard_fraction),
        "n_samples_after_transient": 0,
        "n_cycles_after_transient": 0.0,
        "forcing": {"mean": None, "amplitude": None, "phase_rad": None},
        "qoi": {key: {"mean": None, "amplitude": None,
                      "phase_lag_vs_q_rad": None, "warning": ""}
                for key in qoi_keys},
        "probes": {},
        "warnings": warnings,
        "notes": notes,
        "stability": {"qoi_finite": True, "forcing_finite": True},
    }

    if not qoi_rows or not forcing_rows:
        warnings.append("no time-history samples provided")
        return base

    n_total = min(len(qoi_rows), len(forcing_rows))
    qoi_trim = _drop_transient(qoi_rows[:n_total], discard_fraction)
    force_trim = _drop_transient(forcing_rows[:n_total], discard_fraction)
    probe_trim = _drop_transient((probe_rows or [])[:n_total], discard_fraction) if probe_rows else []
    base["n_samples_after_transient"] = len(qoi_trim)

    if len(qoi_trim) < 2 or len(force_trim) < 2:
        warnings.append("post-transient window is too short for any fitting")
        return base

    t = _rows_to_array(qoi_trim, "time")
    q = _rows_to_array(force_trim, "q")
    duration = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
    n_cycles = float(frequency_hz) * duration if frequency_hz > 0.0 else 0.0
    base["n_cycles_after_transient"] = n_cycles

    if not np.all(np.isfinite(q)):
        warnings.append("forcing history contains non-finite values")
        base["stability"]["forcing_finite"] = False
    forcing_mean = float(np.mean(q))

    if frequency_hz <= 0.0:
        notes.append("frequency_hz <= 0: skipping sinusoidal fit, mean only")
        base["forcing"] = {"mean": forcing_mean,
                           "amplitude": 0.0, "phase_rad": None}
        for key in qoi_keys:
            if key not in qoi_trim[0]:
                continue
            y = _rows_to_array(qoi_trim, key)
            if not np.all(np.isfinite(y)):
                warnings.append(f"qoi.{key} contains non-finite values")
                base["stability"]["qoi_finite"] = False
                base["qoi"][key]["warning"] = "non-finite values present"
                continue
            base["qoi"][key] = {
                "mean": float(np.mean(y)),
                "amplitude": float(np.std(y)),
                "phase_lag_vs_q_rad": None,
                "warning": "phase lag undefined for non-periodic forcing",
            }
        for field in _probe_pressure_field_names(probe_trim):
            y = _rows_to_array(probe_trim, field)
            base["probes"][_strip_pressure(field)] = {
                "pressure_mean": float(np.mean(y)) if y.size else None,
                "pressure_amplitude": float(np.std(y)) if y.size else None,
                "pressure_phase_lag_vs_q_rad": None,
                "warning": "phase lag undefined for non-periodic forcing",
            }
        return base

    omega = 2.0 * math.pi * float(frequency_hz)
    basis = np.column_stack([np.sin(omega * t), np.cos(omega * t), np.ones_like(t)])
    q_coeff = _safe_lstsq(basis, q)
    forcing_amplitude, forcing_phase = _amplitude_phase_from_coeffs(
        float(q_coeff[0]), float(q_coeff[1]),
    )
    forcing_is_flat = _is_flat(q)
    base["forcing"] = {
        "mean": float(q_coeff[2]),
        "amplitude": forcing_amplitude,
        "phase_rad": float(forcing_phase),
    }

    if forcing_is_flat or forcing_amplitude <= 1.0e-12:
        notes.append("epsilon=0 detected (flat or numerically zero forcing); phase lag is null")

    for key in qoi_keys:
        if not qoi_trim or key not in qoi_trim[0]:
            continue
        y = _rows_to_array(qoi_trim, key)
        if not np.all(np.isfinite(y)):
            warnings.append(f"qoi.{key} contains non-finite values")
            base["stability"]["qoi_finite"] = False
            base["qoi"][key]["warning"] = "non-finite values present"
            continue
        amplitude, mean, phase_lag, reason = _phase_lag_metric(
            t=t, y=y, omega=omega, forcing_phase=forcing_phase,
            forcing_amplitude=forcing_amplitude,
            forcing_is_flat=forcing_is_flat,
            n_cycles=n_cycles, min_cycles=min_cycles, min_samples=min_samples,
        )
        entry = {
            "mean": mean,
            "amplitude": amplitude,
            "phase_lag_vs_q_rad": phase_lag,
            "warning": reason,
        }
        if phase_lag is None and reason:
            warnings.append(f"qoi.{key}: phase lag unavailable ({reason})")
        base["qoi"][key] = entry

    for field in _probe_pressure_field_names(probe_trim):
        y = _rows_to_array(probe_trim, field)
        if not np.all(np.isfinite(y)):
            warnings.append(f"probe.{field} contains non-finite values")
            continue
        amplitude, mean, phase_lag, reason = _phase_lag_metric(
            t=t, y=y, omega=omega, forcing_phase=forcing_phase,
            forcing_amplitude=forcing_amplitude,
            forcing_is_flat=forcing_is_flat,
            n_cycles=n_cycles, min_cycles=min_cycles, min_samples=min_samples,
        )
        probe_name = _strip_pressure(field)
        base["probes"][probe_name] = {
            "pressure_mean": mean,
            "pressure_amplitude": amplitude,
            "pressure_phase_lag_vs_q_rad": phase_lag,
            "warning": reason,
        }
        if phase_lag is None and reason:
            warnings.append(f"probe.{probe_name}: phase lag unavailable ({reason})")

    return base


def _strip_pressure(field: str) -> str:
    """``inlet_side_pressure`` -> ``inlet_side``."""
    if field.endswith("_pressure"):
        return field[: -len("_pressure")]
    return field


def fit_sinusoid(t: np.ndarray, y: np.ndarray, frequency_hz: float):
    """Convenience utility for the unit tests.

    Returns (mean, amplitude, phase_rad).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    omega = 2.0 * math.pi * float(frequency_hz)
    basis = np.column_stack([np.sin(omega * t), np.cos(omega * t), np.ones_like(t)])
    coeff = _safe_lstsq(basis, y)
    amplitude, phase = _amplitude_phase_from_coeffs(float(coeff[0]), float(coeff[1]))
    return float(coeff[2]), amplitude, float(phase)
