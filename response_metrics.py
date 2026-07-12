"""Time-aligned response estimators for periodic effective-area studies.

Each signal is fitted against its own timestamps.  The phase convention is

    phase_lag = wrap(forcing_phase - response_phase),

so a response delayed in time has a positive phase lag.  Periodic amplitude
and phase are reported only when the same cycle, sample-count, sampling-density
and flat-signal guards are satisfied; raw fit values remain available for
diagnosis but must not enter surrogates.
"""
import math
from typing import Iterable, List, Optional

import numpy as np


DEFAULT_QOI_KEYS = (
    "exit_mach", "max_mach", "pressure_recovery", "tpr", "shock_x",
    "mdot_prescribed", "mdot_exit", "mass_defect", "thrust",
)


def wrap_phase(angle_rad: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    wrapped = (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped == -math.pi else wrapped


def _rows_to_array(rows: Iterable[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _drop_transient_by_time(rows: List[dict], discard_fraction: float) -> List[dict]:
    """Drop a leading fraction of physical duration, not row count."""
    rows = list(rows)
    if not rows or discard_fraction <= 0.0:
        return rows
    if discard_fraction >= 1.0:
        return []
    times = _rows_to_array(rows, "time")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0.0):
        return []
    cutoff = times[0] + float(discard_fraction) * (times[-1] - times[0])
    return [row for row in rows if float(row["time"]) >= cutoff]


def _safe_lstsq(basis: np.ndarray, y: np.ndarray):
    coeff, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)
    return coeff


def _amplitude_phase_from_coeffs(coeff_sin: float, coeff_cos: float):
    amplitude = float(math.hypot(coeff_sin, coeff_cos))
    phase = 0.0 if amplitude == 0.0 else float(math.atan2(coeff_cos, coeff_sin))
    return amplitude, phase


def _is_flat(values: np.ndarray, rel_tol: float = 1.0e-9) -> bool:
    if values.size == 0:
        return True
    spread = float(np.max(values) - np.min(values))
    scale = max(float(np.max(np.abs(values))), 1.0)
    return spread <= rel_tol * scale


def _support_check(forcing_amplitude, n_cycles, n_samples, min_cycles,
                   min_samples, forcing_flat, response_flat,
                   min_samples_per_cycle=4.0):
    if forcing_flat or forcing_amplitude <= 1.0e-12:
        return False, "epsilon=0 or numerically flat forcing"
    if response_flat:
        return False, "response time history is numerically flat"
    if n_cycles < min_cycles:
        return False, f"insufficient cycles: {n_cycles:.2f} < {min_cycles}"
    if n_samples < min_samples:
        return False, f"insufficient samples: {n_samples} < {min_samples}"
    if n_cycles > 0.0 and n_samples / n_cycles < min_samples_per_cycle:
        density = n_samples / n_cycles
        return False, (
            f"insufficient sampling density: {density:.1f} samples/cycle < "
            f"{min_samples_per_cycle} (aliasing risk)"
        )
    return True, ""


def _periodic_fit(t, y, omega):
    """Fit sin/cos + intercept + centered linear drift and report quality."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t_center = float(np.mean(t))
    tau = t - t_center
    basis = np.column_stack([
        np.sin(omega * t), np.cos(omega * t), np.ones_like(t), tau,
    ])
    coeff = _safe_lstsq(basis, y)
    fitted = basis @ coeff
    residual = y - fitted
    amplitude, phase = _amplitude_phase_from_coeffs(coeff[0], coeff[1])
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    centered = y - np.mean(y)
    ss_total = float(np.sum(centered**2))
    ss_residual = float(np.sum(residual**2))
    r_squared = (1.0 - ss_residual / ss_total
                 if ss_total > 1.0e-30 else float(ss_residual <= 1.0e-30))
    duration = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
    drift_span = abs(float(coeff[3]) * duration)
    drift_fraction = drift_span / max(amplitude, 1.0e-30)
    snr = amplitude / max(residual_rms, 1.0e-30)
    return {
        "mean": float(coeff[2]),
        "raw_amplitude": amplitude,
        "phase_rad": phase,
        "slope_per_s": float(coeff[3]),
        "quality": {
            "r_squared": float(r_squared),
            "residual_rms": residual_rms,
            "drift_fraction": float(drift_fraction),
            "snr": float(snr),
        },
    }


def _fit_response(rows, field, omega, frequency_hz, forcing_fit,
                  min_cycles, min_samples):
    t = _rows_to_array(rows, "time")
    y = _rows_to_array(rows, field)
    fit = _periodic_fit(t, y, omega)
    duration = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
    cycles = float(frequency_hz) * duration
    response_flat = (_is_flat(y) or fit["raw_amplitude"]
                     <= 1.0e-14 * max(abs(fit["mean"]), 1.0))
    supported, reason = _support_check(
        forcing_fit["raw_amplitude"], cycles, len(t), min_cycles,
        min_samples, forcing_fit["flat"], response_flat,
    )
    fit["quality"].update({
        "supported": bool(supported),
        "reason": reason,
        "n_samples": int(len(t)),
        "n_cycles": cycles,
        "duration": duration,
    })
    fit["amplitude"] = fit["raw_amplitude"] if supported else None
    fit["phase_lag_vs_q_rad"] = (
        wrap_phase(forcing_fit["phase_rad"] - fit["phase_rad"])
        if supported else None
    )
    fit["warning"] = reason
    return fit


def _probe_pressure_field_names(rows):
    if not rows:
        return []
    return [key for key in rows[0] if key.endswith("_pressure")]


def extract_response_metrics(qoi_rows: List[dict],
                             forcing_rows: List[dict],
                             probe_rows: Optional[List[dict]] = None,
                             frequency_hz: float = 0.0,
                             discard_fraction: float = 0.25,
                             qoi_keys: Iterable[str] = DEFAULT_QOI_KEYS,
                             min_cycles: float = 1.0,
                             min_samples: int = 8) -> dict:
    """Extract drift-aware periodic metrics using each series' own times.

    A positive ``phase_lag_vs_q_rad`` means the response is delayed relative
    to the forcing.  Unsupported periodic amplitudes and phases are ``None``;
    ``raw_amplitude`` and the quality block remain for diagnostics.
    """
    qoi_keys = list(qoi_keys)
    warnings = []
    notes = []
    base = {
        "transient_discard_fraction": float(discard_fraction),
        "transient_cut_method": "physical_time",
        "n_samples_after_transient": 0,
        "n_cycles_after_transient": 0.0,
        "forcing": {
            "mean": None, "amplitude": None, "raw_amplitude": None,
            "phase_rad": None, "quality": {"supported": False},
        },
        "qoi": {
            key: {
                "mean": None, "amplitude": None, "raw_amplitude": None,
                "phase_lag_vs_q_rad": None, "warning": "",
                "quality": {"supported": False},
            } for key in qoi_keys
        },
        "probes": {},
        "warnings": warnings,
        "notes": notes,
        "stability": {"qoi_finite": True, "forcing_finite": True},
    }
    if not qoi_rows or not forcing_rows:
        warnings.append("no time-history samples provided")
        return base

    qoi_trim = _drop_transient_by_time(qoi_rows, discard_fraction)
    forcing_trim = _drop_transient_by_time(forcing_rows, discard_fraction)
    probe_trim = _drop_transient_by_time(probe_rows or [], discard_fraction)
    base["n_samples_after_transient"] = len(qoi_trim)
    if len(qoi_trim) >= 2 and frequency_hz > 0.0:
        t_q = _rows_to_array(qoi_trim, "time")
        base["n_cycles_after_transient"] = float(frequency_hz) * (t_q[-1] - t_q[0])
    if len(qoi_trim) < 2 or len(forcing_trim) < 2:
        warnings.append("post-transient window is too short for fitting")
        return base

    t_force = _rows_to_array(forcing_trim, "time")
    q = _rows_to_array(forcing_trim, "q")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(t_force)):
        base["stability"]["forcing_finite"] = False
        warnings.append("forcing history contains non-finite values")
        return base

    if frequency_hz <= 0.0:
        base["forcing"] = {
            "mean": float(np.mean(q)), "amplitude": None,
            "raw_amplitude": float(np.std(q)), "phase_rad": None,
            "quality": {"supported": False,
                        "reason": "frequency_hz <= 0"},
        }
        notes.append("frequency_hz <= 0: periodic amplitude/phase undefined")
        for key in qoi_keys:
            if key not in qoi_trim[0]:
                continue
            y = _rows_to_array(qoi_trim, key)
            base["qoi"][key].update({
                "mean": float(np.mean(y)), "raw_amplitude": float(np.std(y)),
                "warning": "periodic response undefined for non-periodic forcing",
                "quality": {"supported": False,
                            "reason": "frequency_hz <= 0"},
            })
        return base

    omega = 2.0 * math.pi * float(frequency_hz)
    forcing_fit = _periodic_fit(t_force, q, omega)
    force_duration = float(t_force[-1] - t_force[0])
    force_cycles = float(frequency_hz) * force_duration
    forcing_flat = _is_flat(q)
    forcing_supported, forcing_reason = _support_check(
        forcing_fit["raw_amplitude"], force_cycles, len(t_force), min_cycles,
        min_samples, forcing_flat, False,
    )
    forcing_fit["flat"] = forcing_flat
    forcing_fit["amplitude"] = forcing_fit["raw_amplitude"]
    forcing_fit["quality"].update({
        "supported": bool(forcing_supported), "reason": forcing_reason,
        "n_samples": int(len(t_force)), "n_cycles": force_cycles,
        "duration": force_duration,
    })
    base["forcing"] = {key: value for key, value in forcing_fit.items()
                       if key != "flat"}
    if forcing_flat:
        notes.append("epsilon=0 detected; periodic response metrics are unsupported")

    for key in qoi_keys:
        if key not in qoi_trim[0]:
            continue
        y = _rows_to_array(qoi_trim, key)
        t = _rows_to_array(qoi_trim, "time")
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(t)):
            base["stability"]["qoi_finite"] = False
            base["qoi"][key]["warning"] = "non-finite values present"
            warnings.append(f"qoi.{key} contains non-finite values")
            continue
        entry = _fit_response(
            qoi_trim, key, omega, frequency_hz, forcing_fit,
            min_cycles, min_samples,
        )
        base["qoi"][key] = entry
        if not entry["quality"]["supported"]:
            warnings.append(f"qoi.{key}: response unavailable ({entry['warning']})")

    for field in _probe_pressure_field_names(probe_trim):
        y = _rows_to_array(probe_trim, field)
        t = _rows_to_array(probe_trim, "time")
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(t)):
            warnings.append(f"probe.{field} contains non-finite values")
            continue
        entry = _fit_response(
            probe_trim, field, omega, frequency_hz, forcing_fit,
            min_cycles, min_samples,
        )
        name = _strip_pressure(field)
        base["probes"][name] = {
            "pressure_mean": entry["mean"],
            "pressure_amplitude": entry["amplitude"],
            "pressure_raw_amplitude": entry["raw_amplitude"],
            "pressure_phase_lag_vs_q_rad": entry["phase_lag_vs_q_rad"],
            "slope_per_s": entry["slope_per_s"],
            "quality": entry["quality"],
            "warning": entry["warning"],
        }
        if not entry["quality"]["supported"]:
            warnings.append(f"probe.{name}: response unavailable ({entry['warning']})")
    return base


def _strip_pressure(field: str) -> str:
    return field[:-len("_pressure")] if field.endswith("_pressure") else field


def fit_sinusoid(t: np.ndarray, y: np.ndarray, frequency_hz: float):
    """Return (mean, amplitude, phase) from the drift-aware periodic fit."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    fit = _periodic_fit(t, y, 2.0 * math.pi * float(frequency_hz))
    return fit["mean"], fit["raw_amplitude"], fit["phase_rad"]
