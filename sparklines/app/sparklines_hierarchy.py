"""Archive fetch and composite hierarchy helpers for the sparkline notebook."""

from __future__ import annotations

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests
import yaml
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.auto import tqdm


ARCHIVER_URL = "http://lcls-archapp.slac.stanford.edu/retrieval/data/getData.json"
ARCHIVE_TIMEOUT_SECONDS = 20.0
ARCHIVE_MAX_WORKERS = 8
LOCAL_TIMEZONE = ZoneInfo("America/Los_Angeles")
DEFAULT_MONITOR_SPECS_PATH = Path(__file__).with_name("monitor_pvs.yaml")
DEFAULT_PV_GROUPS_PATH = Path(__file__).with_name("pv_groups.yaml")


def _load_default_monitor_specs(path: Path | None = None) -> dict:
    specs_path = Path(path) if path is not None else DEFAULT_MONITOR_SPECS_PATH
    with specs_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {specs_path}")
    monitor_specs = payload.get("monitor_pvs", payload)
    if not isinstance(monitor_specs, dict):
        raise ValueError(f"Expected 'monitor_pvs' to be a mapping in {specs_path}")
    return monitor_specs


DEFAULT_MONITOR_SPECS = _load_default_monitor_specs()


def load_pv_groups(path: Path | None = None) -> dict:
    groups_path = Path(path) if path is not None else DEFAULT_PV_GROUPS_PATH
    with groups_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {groups_path}")
    return payload


def _extract_time_and_values(data: dict):
    sec = np.asarray(data["secondsPastEpoch"], dtype=float)
    nsec = np.asarray(data.get("nanoseconds", data.get("nanosecond", 0)), dtype=float)
    if nsec.ndim == 0:
        t = sec + float(nsec) * 1e-9
    else:
        t = sec + nsec * 1e-9
    y = np.asarray(data["values"], dtype=float)
    keep = np.isfinite(t) & np.isfinite(y)
    return t[keep], y[keep]


def _series_varies_in_window(t, y, window_start, window_end, change_tol=0.0) -> bool:
    in_window = (t >= window_start) & (t <= window_end)
    y_window = y[in_window]
    if y_window.size < 2:
        return False
    return float(np.max(y_window) - np.min(y_window)) > float(change_tol)


def _normalize_threshold(value, default=1.0) -> float:
    if value in (None, 0):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _normalize_optional_positive_float(value) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _event_times_from_value_changes(t, y, min_delta=None):
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if t_arr.shape != y_arr.shape:
        raise ValueError(
            "time/value length mismatch: "
            f"len(t)={t_arr.size} len(y)={y_arr.size}"
        )
    if t_arr.size == 0:
        return np.array([], dtype=float)

    mask = np.isfinite(t_arr) & np.isfinite(y_arr)
    t_arr = t_arr[mask]
    y_arr = y_arr[mask]
    if t_arr.size == 0:
        return np.array([], dtype=float)

    order = np.argsort(t_arr, kind="mergesort")
    t_sorted = t_arr[order]
    y_sorted = y_arr[order]

    if min_delta is None:
        return t_sorted

    min_delta = float(min_delta)
    if not np.isfinite(min_delta) or min_delta <= 0:
        return t_sorted

    events = []
    last_setpoint = float(y_sorted[0])
    for ti, yi in zip(t_sorted[1:], y_sorted[1:]):
        yi = float(yi)
        if abs(yi - last_setpoint) >= min_delta:
            events.append(float(ti))
            last_setpoint = yi
    return np.asarray(events, dtype=float)


def _detect_tuning_periods(t, timeout=300):
    t_arr = np.asarray(t, dtype=float)
    if t_arr.size == 0:
        return []

    t_arr = np.sort(t_arr[np.isfinite(t_arr)])
    if t_arr.size == 0:
        return []

    periods = []
    t_on = float(t_arr[0])
    last_event = float(t_arr[0])
    for ti in t_arr[1:]:
        ti = float(ti)
        if ti - last_event <= timeout:
            last_event = ti
            continue
        periods.append((t_on, last_event))
        t_on = ti
        last_event = ti

    periods.append((t_on, last_event))
    return periods


def _series_from_time_values(template: dict, t: np.ndarray, values: np.ndarray) -> dict:
    seconds = np.floor(t).astype(np.int64)
    nanoseconds = np.rint((t - seconds) * 1e9).astype(np.int64)
    rollover = nanoseconds == 1000000000
    seconds[rollover] += 1
    nanoseconds[rollover] = 0

    out = dict(template)
    out["secondsPastEpoch"] = seconds
    out["nanoseconds"] = nanoseconds
    out["values"] = np.asarray(values, dtype=float)
    out["severity"] = np.zeros_like(seconds, dtype=np.int64)
    out["status"] = np.zeros_like(seconds, dtype=np.int64)
    return out


def _compress_measurement_series_for_composite(data: dict, deadband=None, timeout=300):
    t, values = _extract_time_and_values(data)
    if t.size == 0:
        return data

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    values = values[order]

    event_times = _event_times_from_value_changes(t, values, min_delta=deadband)
    periods = _detect_tuning_periods(event_times, timeout=timeout)
    if not periods:
        return _series_from_time_values(data, t[:1], values[:1])

    keep_indices = {0}
    for t_on, t_off in periods:
        period_mask = (t >= t_on) & (t <= t_off)
        period_indices = np.flatnonzero(period_mask)
        if period_indices.size == 0:
            continue
        first_idx = int(period_indices[0])
        last_idx = int(period_indices[-1])
        period_values = values[period_indices]
        keep_indices.update(
            {
                first_idx,
                last_idx,
                int(period_indices[np.argmin(period_values)]),
                int(period_indices[np.argmax(period_values)]),
            }
        )

    keep = np.asarray(sorted(keep_indices), dtype=int)
    return _series_from_time_values(data, t[keep], values[keep])


def _normalize_beam_paths(value) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        raw_entries = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_entries = []
        for entry in value:
            if isinstance(entry, str):
                raw_entries.extend(entry.split(","))
    else:
        return ()

    normalized = []
    for entry in raw_entries:
        token = str(entry).strip().upper()
        if token and token not in normalized:
            normalized.append(token)
    return tuple(normalized)


def _normalize_monitor_specs(monitor_specs) -> dict[str, dict]:
    if monitor_specs is None:
        return {}

    if isinstance(monitor_specs, dict):
        entries = monitor_specs.items()
    elif isinstance(monitor_specs, list):
        entries = []
        for item in monitor_specs:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            if label:
                entries.append((label, item))
    else:
        return {}

    normalized = {}
    for label, spec in entries:
        if not isinstance(spec, dict):
            continue

        pv_name = str(spec.get("pv_name", "")).strip()
        if not pv_name:
            continue

        try:
            value_scale = float(spec.get("value_scale", 1.0))
        except (TypeError, ValueError):
            value_scale = 1.0

        subsample = spec.get("subsample", True)
        if isinstance(subsample, str):
            subsample = subsample.strip().lower() not in {"0", "false", "no", "off"}
        else:
            subsample = bool(subsample)

        normalized[str(label)] = {
            "pv_name": pv_name,
            "beam_paths": _normalize_beam_paths(
                spec.get("Beam_Path", spec.get("beam_paths"))
            ),
            "value_scale": value_scale,
            "subsample": subsample,
        }

    return normalized


def _parse_group_hierarchy(pv_groups_dict: dict) -> dict:
    groups = pv_groups_dict.get("groups", [])
    if not isinstance(groups, list):
        raise ValueError('Expected top-level key "groups" to be a list.')

    parsed = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_name = str(group.get("group_name", "")).strip()
        if not group_name:
            continue

        group_threshold = group.get("threshold", None)
        group_beam_paths = _normalize_beam_paths(group.get("Beam_Path"))
        subgroups = group.get("subgroups", {}) or {}
        subgroup_nodes = {}

        for subgroup_name, subgroup in subgroups.items():
            if not isinstance(subgroup, dict):
                continue

            inherited_subgroup_threshold = subgroup.get("threshold", group_threshold)
            inherited_subgroup_measurement = _normalize_bool(
                subgroup.get("measurement", False),
                default=False,
            )
            inherited_subgroup_measurement_deadband = _normalize_optional_positive_float(
                subgroup.get("measurement_deadband")
            )
            subgroup_beam_paths = _normalize_beam_paths(
                subgroup.get("Beam_Path", group_beam_paths)
            )
            pv_specs = []
            for entry in subgroup.get("pv", []) or []:
                if isinstance(entry, dict):
                    pv_name = str(entry.get("pv_name", "")).strip()
                    threshold = entry.get("threshold", inherited_subgroup_threshold)
                    measurement = _normalize_bool(
                        entry.get("measurement", inherited_subgroup_measurement),
                        default=inherited_subgroup_measurement,
                    )
                    measurement_deadband = _normalize_optional_positive_float(
                        entry.get(
                            "measurement_deadband",
                            inherited_subgroup_measurement_deadband,
                        )
                    )
                    beam_paths = _normalize_beam_paths(
                        entry.get("Beam_Path", subgroup_beam_paths)
                    )
                elif isinstance(entry, str):
                    pv_name = entry.strip()
                    threshold = inherited_subgroup_threshold
                    measurement = inherited_subgroup_measurement
                    measurement_deadband = inherited_subgroup_measurement_deadband
                    beam_paths = subgroup_beam_paths
                else:
                    continue

                if not pv_name:
                    continue

                pv_specs.append(
                    {
                        "pv_name": pv_name,
                        "threshold": _normalize_threshold(threshold, default=1.0),
                        "measurement": measurement,
                        "measurement_deadband": measurement_deadband,
                        "beam_paths": beam_paths,
                    }
                )

            if pv_specs:
                combined_subgroup_beam_paths = list(subgroup_beam_paths)
                for spec in pv_specs:
                    for beam_path in spec.get("beam_paths", ()):
                        if beam_path not in combined_subgroup_beam_paths:
                            combined_subgroup_beam_paths.append(beam_path)
                subgroup_nodes[str(subgroup_name)] = {
                    "threshold": _normalize_threshold(
                        inherited_subgroup_threshold, default=1.0
                    ),
                    "beam_paths": tuple(combined_subgroup_beam_paths),
                    "pv_specs": pv_specs,
                }

        if subgroup_nodes:
            combined_group_beam_paths = list(group_beam_paths)
            for subgroup in subgroup_nodes.values():
                for beam_path in subgroup.get("beam_paths", ()):
                    if beam_path not in combined_group_beam_paths:
                        combined_group_beam_paths.append(beam_path)
            parsed[group_name] = {
                "threshold": _normalize_threshold(group_threshold, default=1.0),
                "beam_paths": tuple(combined_group_beam_paths),
                "subgroups": subgroup_nodes,
            }

    return parsed


def _build_composite_from_series(
    node_name: str,
    series_by_name: list[tuple[str, dict]],
    threshold_by_name: dict | None = None,
):
    threshold_by_name = threshold_by_name or {}

    normalized_series = []
    t0_candidates = []
    for series_name, data in series_by_name:
        t, values = _extract_time_and_values(data)
        if t.size == 0:
            continue

        order = np.argsort(t)
        t = t[order]
        values = values[order]

        threshold = _normalize_threshold(
            threshold_by_name.get(series_name, 1.0), default=1.0
        )
        normalized_series.append((series_name, t, values / threshold))
        t0_candidates.append(float(t[0]))

    if not normalized_series:
        return None

    t0 = max(t0_candidates)
    baseline_values = {}
    current_values = {}
    changes = {}

    for series_name, t, values in normalized_series:
        start_idx = int(np.searchsorted(t, t0, side="right") - 1)
        if start_idx < 0:
            start_idx = 0

        baseline = float(values[start_idx])
        baseline_values[series_name] = baseline
        current_values[series_name] = baseline

        for ts, value in zip(t[start_idx + 1 :], values[start_idx + 1 :]):
            changes.setdefault(float(ts), []).append((series_name, float(value)))

    composite_times = [t0]
    composite_values = [0.0]

    for ts in sorted(changes):
        for series_name, value in changes[ts]:
            current_values[series_name] = value

        deltas = [
            abs(current_values[name] - baseline_values[name])
            for name in baseline_values
        ]
        composite_value = sum(deltas) / len(deltas) if deltas else 0.0
        if composite_value == composite_values[-1]:
            continue
        composite_times.append(ts)
        composite_values.append(composite_value)

    composite_times = np.asarray(composite_times, dtype=float)
    seconds = np.floor(composite_times).astype(np.int64)
    nanoseconds = np.rint((composite_times - seconds) * 1e9).astype(np.int64)
    rollover = nanoseconds == 1000000000
    seconds[rollover] += 1
    nanoseconds[rollover] = 0

    return {
        "name": node_name,
        "secondsPastEpoch": seconds,
        "nanoseconds": nanoseconds,
        "values": np.asarray(composite_values, dtype=float),
        "severity": np.zeros_like(seconds, dtype=np.int64),
        "status": np.zeros_like(seconds, dtype=np.int64),
    }


def _unwrap_archive_payload(raw):
    if raw is None:
        return None

    if isinstance(raw, dict):
        value = raw.get("value")
        if isinstance(value, dict) and "value" in value:
            return value["value"]

    return raw


def _normalize_archive_batch(pv_names: list[str], raw_batch) -> dict[str, object | None]:
    normalized = {pv_name: None for pv_name in pv_names}
    if raw_batch is None:
        return normalized

    if isinstance(raw_batch, list):
        for idx, item in enumerate(raw_batch):
            pv_name = None
            if isinstance(item, dict):
                candidate = str(item.get("pvName", "")).strip()
                if candidate in normalized:
                    pv_name = candidate

            if pv_name is None and idx < len(pv_names):
                pv_name = pv_names[idx]

            if pv_name is not None:
                normalized[pv_name] = _unwrap_archive_payload(item)

        return normalized

    if len(pv_names) == 1:
        normalized[pv_names[0]] = _unwrap_archive_payload(raw_batch)
        return normalized

    if isinstance(raw_batch, dict):
        candidate = str(raw_batch.get("pvName", "")).strip()
        if candidate in normalized:
            normalized[candidate] = _unwrap_archive_payload(raw_batch)

    return normalized


class ArchiveRequestError(RuntimeError):
    """Raised when the archive appliance response is structurally invalid."""


def _format_archive_time(
    value: dt.datetime | str | None, local_timezone=LOCAL_TIMEZONE
) -> str | None:
    if value is None or isinstance(value, str):
        return value

    if value.tzinfo is None:
        value = value.replace(tzinfo=local_timezone)

    return value.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
    retry=retry_if_exception_type(
        (requests.RequestException, ArchiveRequestError, ValueError, KeyError, TypeError)
    ),
)
def _fetch_single_pv_direct(
    pv_name: str,
    start: dt.datetime,
    end: dt.datetime,
    timeout: float,
    *,
    archiver_url=ARCHIVER_URL,
    local_timezone=LOCAL_TIMEZONE,
):
    response = requests.get(
        archiver_url,
        params={
            "pv": pv_name,
            "from": _format_archive_time(start, local_timezone=local_timezone),
            "to": _format_archive_time(end, local_timezone=local_timezone),
        },
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise ArchiveRequestError(f"archive returned empty payload for PV {pv_name}")

    first = payload[0]
    meta = first.get("meta") or {}
    data = first.get("data")
    if not isinstance(data, list):
        raise ArchiveRequestError(f"archive payload missing data array for PV {pv_name}")

    return pv_name, {
        "name": meta.get("name", pv_name),
        "secondsPastEpoch": np.array([datum["secs"] for datum in data]),
        "values": np.array([datum["val"] for datum in data]),
        "nanoseconds": np.array([datum["nanos"] for datum in data]),
        "severity": np.array([datum["severity"] for datum in data]),
        "status": np.array([datum["status"] for datum in data]),
    }


def _fetch_archive_batch(
    pv_names: list[str],
    start: dt.datetime,
    end: dt.datetime,
    *,
    archiver_url=ARCHIVER_URL,
    archive_timeout_seconds=ARCHIVE_TIMEOUT_SECONDS,
    archive_max_workers=ARCHIVE_MAX_WORKERS,
):
    if not pv_names:
        return {}, {}, {}

    raw_by_pv = {pv_name: None for pv_name in pv_names}
    errors = {}
    fetch_timings = {}
    worker_count = max(1, min(len(pv_names), int(archive_max_workers)))
    batch_started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_by_pv = {
            executor.submit(
                _fetch_single_pv_timed,
                pv_name,
                start,
                end,
                archive_timeout_seconds,
                archiver_url=archiver_url,
            ): pv_name
            for pv_name in pv_names
        }
        with tqdm(total=len(pv_names), desc="Loading PVs from archiver", unit="pv") as progress:
            for future in as_completed(future_by_pv):
                pv_name = future_by_pv[future]
                try:
                    result = future.result()
                except Exception as exc:
                    errors[pv_name] = f"archive error: {exc}"
                else:
                    fetch_timings[pv_name] = float(result["fetch_seconds"])
                    if result["error"] is not None:
                        errors[pv_name] = result["error"]
                    else:
                        raw_by_pv[pv_name] = _unwrap_archive_payload(result["raw"])
                finally:
                    progress.update(1)

    return raw_by_pv, errors, {
        "per_pv_fetch_seconds": fetch_timings,
        "fetch_wall_seconds": time.perf_counter() - batch_started,
    }


def _fetch_single_pv_timed(
    pv_name: str,
    start: dt.datetime,
    end: dt.datetime,
    timeout: float,
    *,
    archiver_url=ARCHIVER_URL,
    local_timezone=LOCAL_TIMEZONE,
):
    fetch_started = time.perf_counter()
    try:
        _, raw = _fetch_single_pv_direct(
            pv_name,
            start,
            end,
            timeout,
            archiver_url=archiver_url,
            local_timezone=local_timezone,
        )
    except Exception as exc:
        return {
            "pv_name": pv_name,
            "raw": None,
            "error": f"archive error: {exc}",
            "fetch_seconds": time.perf_counter() - fetch_started,
        }

    return {
        "pv_name": pv_name,
        "raw": raw,
        "error": None,
        "fetch_seconds": time.perf_counter() - fetch_started,
    }


def build_composite_hierarchy(
    pv_groups_dict: dict,
    start: dt.datetime,
    end: dt.datetime,
    *,
    monitor_specs=None,
    archiver_url=ARCHIVER_URL,
    archive_timeout_seconds=ARCHIVE_TIMEOUT_SECONDS,
    archive_max_workers=ARCHIVE_MAX_WORKERS,
) -> dict:
    build_started = time.perf_counter()
    parsed = _parse_group_hierarchy(pv_groups_dict)
    raw_monitor_specs = (
        monitor_specs if monitor_specs is not None else DEFAULT_MONITOR_SPECS
    )
    monitor_specs = _normalize_monitor_specs(raw_monitor_specs)

    all_pvs = {
        spec["pv_name"]
        for group in parsed.values()
        for subgroup in group["subgroups"].values()
        for spec in subgroup["pv_specs"]
    }
    requested_pvs = all_pvs | {spec["pv_name"] for spec in monitor_specs.values()}

    pv_cache = {}
    skipped_pvs = {}
    ordered_pvs = sorted(requested_pvs)

    raw_by_pv, fetch_errors, fetch_timing = _fetch_archive_batch(
        ordered_pvs,
        start,
        end,
        archiver_url=archiver_url,
        archive_timeout_seconds=archive_timeout_seconds,
        archive_max_workers=archive_max_workers,
    )
    skipped_pvs.update(fetch_errors)

    for pv_name in ordered_pvs:
        if pv_name in skipped_pvs:
            continue

        raw = raw_by_pv.get(pv_name)
        if raw is None:
            skipped_pvs[pv_name] = "archive returned None"
            continue

        try:
            data = dict(raw)
        except Exception as exc:
            skipped_pvs[pv_name] = f"invalid archive payload: {exc}"
            continue

        if "secondsPastEpoch" not in data or "values" not in data:
            skipped_pvs[pv_name] = "missing required keys"
            continue

        data["name"] = pv_name
        pv_cache[pv_name] = data

    hierarchy = {
        "groups": {},
        "pv_cache": pv_cache,
        "skipped_pvs": skipped_pvs,
        "monitor_pvs": {},
        "timing": {},
    }

    for label, spec in monitor_specs.items():
        pv_name = spec["pv_name"]
        if pv_name not in pv_cache:
            continue
        monitor_entry = dict(pv_cache[pv_name])
        monitor_entry["name"] = pv_name
        monitor_entry["label"] = label
        monitor_entry["beam_paths"] = spec["beam_paths"]
        monitor_entry["value_scale"] = spec.get("value_scale", 1.0)
        monitor_entry["subsample"] = spec.get("subsample", True)
        hierarchy["monitor_pvs"][label] = monitor_entry

    for group_name, group in parsed.items():
        subgroup_nodes = {}
        subgroup_series = []

        for subgroup_name, subgroup in group["subgroups"].items():
            pv_series = []
            threshold_map = {}
            pv_data = []

            for spec in subgroup["pv_specs"]:
                pv_name = spec["pv_name"]
                if pv_name not in pv_cache:
                    continue
                pv_entry = dict(pv_cache[pv_name])
                pv_entry["beam_paths"] = spec.get(
                    "beam_paths", subgroup.get("beam_paths", ())
                )
                pv_entry["measurement"] = bool(spec.get("measurement", False))
                pv_entry["measurement_deadband"] = spec.get("measurement_deadband")
                composite_source = (
                    _compress_measurement_series_for_composite(
                        pv_entry,
                        deadband=pv_entry.get("measurement_deadband"),
                    )
                    if pv_entry["measurement"]
                    else pv_entry
                )
                pv_series.append((pv_name, composite_source))
                pv_data.append(pv_entry)
                threshold_map[pv_name] = spec["threshold"]

            if not pv_series:
                continue

            subgroup_composite = _build_composite_from_series(
                subgroup_name,
                pv_series,
                threshold_by_name=threshold_map,
            )
            if subgroup_composite is None:
                continue

            subgroup_composite["beam_paths"] = subgroup.get("beam_paths", ())
            subgroup_nodes[subgroup_name] = {
                "threshold": subgroup["threshold"],
                "beam_paths": subgroup.get("beam_paths", ()),
                "pv_specs": subgroup["pv_specs"],
                "pv_data": pv_data,
                "composite": subgroup_composite,
            }
            subgroup_series.append((subgroup_name, subgroup_composite))

        if not subgroup_nodes:
            continue

        group_composite = _build_composite_from_series(
            group_name,
            subgroup_series,
            threshold_by_name={name: 1.0 for name, _ in subgroup_series},
        )

        if group_composite is not None:
            group_composite["beam_paths"] = group.get("beam_paths", ())

        hierarchy["groups"][group_name] = {
            "threshold": group["threshold"],
            "beam_paths": group.get("beam_paths", ()),
            "subgroups": subgroup_nodes,
            "composite": group_composite,
        }

    hierarchy["timing"] = {
        **fetch_timing,
        "build_wall_seconds": time.perf_counter() - build_started,
    }

    return hierarchy


def build_default_composite_hierarchy(
    start: dt.datetime,
    end: dt.datetime,
    *,
    pv_groups_path: Path | None = None,
    monitor_specs=None,
    archiver_url=ARCHIVER_URL,
    archive_timeout_seconds=ARCHIVE_TIMEOUT_SECONDS,
    archive_max_workers=ARCHIVE_MAX_WORKERS,
) -> dict:
    """Build the hierarchy using local app YAML defaults."""
    pv_groups_dict = load_pv_groups(pv_groups_path)
    return build_composite_hierarchy(
        pv_groups_dict,
        start,
        end,
        monitor_specs=monitor_specs,
        archiver_url=archiver_url,
        archive_timeout_seconds=archive_timeout_seconds,
        archive_max_workers=archive_max_workers,
    )
