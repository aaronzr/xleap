"""Diagnostic probes for archive-appliance connectivity and data availability.

Helpers used by the ``laser_position_1h_archive`` and ``undulator_probe``
notebooks to inspect what the archiver returns for individual PVs. Includes a
timeout-resilient bisect probe that halves the time window whenever a request
fails, converging on the smallest working window per PV.
"""

from __future__ import annotations

import datetime as dt
import time
from collections import deque

import requests

from sparklines_v2.archive.fetch import format_archive_time, unwrap_archive_payload
from sparklines_v2.archive.hierarchy import (
    iter_laser_position_measurements,
    load_group_specs,
)
from sparklines_v2.beamlines import ARCHIVER_URL, LOCAL_TIMEZONE


__all__ = [
    "DEFAULT_PROBE_MAX_DEPTH",
    "DEFAULT_PROBE_MIN_WINDOW",
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "base_and_1h_candidates",
    "bisect_until_success",
    "fetch_archive_window",
    "find_pv_result",
    "iter_laser_position_measurements",
    "load_group_specs",
    "probe_archive_pv",
    "run_undulator_probe",
    "split_window",
    "summarize_probe",
]


DEFAULT_PROBE_TIMEOUT_SECONDS = 8.0
DEFAULT_PROBE_MIN_WINDOW = dt.timedelta(minutes=5)
DEFAULT_PROBE_MAX_DEPTH = 12


def base_and_1h_candidates(pv_name: str) -> list[dict[str, str]]:
    """Return ``base`` and ``1H`` candidate variants for one configured PV name."""
    base = pv_name[:-2] if pv_name.endswith("1H") else pv_name
    sampled = f"{base}1H"
    return [
        {"configured_pv": pv_name, "candidate_kind": "base", "pv_name": base},
        {"configured_pv": pv_name, "candidate_kind": "1H", "pv_name": sampled},
    ]


def probe_archive_pv(
    pv_name: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    timeout: float = 20.0,
    archiver_url: str = ARCHIVER_URL,
    local_timezone=LOCAL_TIMEZONE,
) -> dict:
    """Fetch one PV over a window and report availability, size, and timing."""
    params = {
        "pv": pv_name,
        "from": format_archive_time(start, local_timezone=local_timezone),
        "to": format_archive_time(end, local_timezone=local_timezone),
    }
    request_start = time.monotonic()
    response = requests.get(archiver_url, params=params, timeout=timeout)
    elapsed_s = time.monotonic() - request_start
    response.raise_for_status()

    payload = response.json()
    if isinstance(payload, list):
        first = payload[0] if payload else None
    else:
        first = payload
    first = unwrap_archive_payload(first) if isinstance(first, dict) else first

    if not first:
        return {
            "ok": False,
            "point_count": 0,
            "elapsed_s": elapsed_s,
            "error": "empty payload",
        }

    data = first.get("data") if isinstance(first, dict) else None
    if data is None:
        return {
            "ok": False,
            "point_count": 0,
            "elapsed_s": elapsed_s,
            "error": "payload missing data array",
        }

    seconds = [
        item.get("secs")
        for item in data
        if isinstance(item, dict) and item.get("secs") is not None
    ]
    return {
        "ok": bool(data),
        "point_count": len(data),
        "elapsed_s": elapsed_s,
        "first_time": (
            dt.datetime.fromtimestamp(min(seconds), tz=local_timezone)
            if seconds
            else None
        ),
        "last_time": (
            dt.datetime.fromtimestamp(max(seconds), tz=local_timezone)
            if seconds
            else None
        ),
        "error": "" if data else "no data points in window",
    }


def fetch_archive_window(
    pv_name: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    archiver_url: str = ARCHIVER_URL,
    local_timezone=LOCAL_TIMEZONE,
) -> dict:
    """Fetch a raw archive window and report its size, byte count, and wall time."""
    params = {
        "pv": pv_name,
        "from": format_archive_time(start, local_timezone=local_timezone),
        "to": format_archive_time(end, local_timezone=local_timezone),
    }
    started = time.perf_counter()
    response = requests.get(archiver_url, params=params, timeout=timeout)
    elapsed = time.perf_counter() - started
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"archive returned empty payload for PV {pv_name}")

    data = payload[0].get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"archive payload missing data array for PV {pv_name}")

    return {
        "pv_name": pv_name,
        "start": start,
        "end": end,
        "elapsed_seconds": elapsed,
        "sample_count": len(data),
        "response_bytes": len(response.content),
    }


def split_window(
    start: dt.datetime, end: dt.datetime
) -> tuple[dt.datetime, dt.datetime]:
    """Return the midpoint of ``[start, end]`` as ``(start, mid)``."""
    return start, start + (end - start) / 2


def bisect_until_success(
    pv_name: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    min_window: dt.timedelta = DEFAULT_PROBE_MIN_WINDOW,
    max_depth: int = DEFAULT_PROBE_MAX_DEPTH,
    archiver_url: str = ARCHIVER_URL,
    local_timezone=LOCAL_TIMEZONE,
) -> dict:
    """Bisect the window whenever a fetch fails, converging on the smallest working window."""
    attempts = []
    queue = deque([(start, end, 0)])

    while queue:
        win_start, win_end, depth = queue.popleft()
        duration = win_end - win_start
        attempt = {
            "pv_name": pv_name,
            "depth": depth,
            "start": win_start,
            "end": win_end,
            "duration": duration,
        }

        try:
            payload = fetch_archive_window(
                pv_name,
                win_start,
                win_end,
                timeout=timeout,
                archiver_url=archiver_url,
                local_timezone=local_timezone,
            )
        except Exception as exc:
            attempt["success"] = False
            attempt["error_type"] = type(exc).__name__
            attempt["error"] = str(exc)
            attempts.append(attempt)

            if depth < max_depth and duration > min_window:
                left_start, mid = split_window(win_start, win_end)
                queue.append((left_start, mid, depth + 1))
                queue.append((mid, win_end, depth + 1))
            continue

        attempt.update(payload)
        attempt["success"] = True
        attempts.append(attempt)

    successes = [attempt for attempt in attempts if attempt["success"]]
    failures = [attempt for attempt in attempts if not attempt["success"]]

    return {
        "pv_name": pv_name,
        "attempts": attempts,
        "successes": successes,
        "failures": failures,
        "first_success": successes[0] if successes else None,
        "smallest_success": (
            min(successes, key=lambda item: item["duration"]) if successes else None
        ),
        "smallest_failure": (
            min(failures, key=lambda item: item["duration"]) if failures else None
        ),
    }


def run_undulator_probe(
    start: dt.datetime,
    end: dt.datetime,
    *,
    target_subgroups: list[str] | None = None,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    min_window: dt.timedelta = DEFAULT_PROBE_MIN_WINDOW,
    max_depth: int = DEFAULT_PROBE_MAX_DEPTH,
    group_name: str = "undulator",
) -> dict[str, list[dict]]:
    """Bisect-probe every PV in the named group's subgroups."""
    specs = load_group_specs(group_name=group_name)
    results = {}
    for subgroup_name, pv_names in specs.items():
        if target_subgroups is not None and subgroup_name not in target_subgroups:
            continue
        results[subgroup_name] = [
            bisect_until_success(
                pv_name,
                start,
                end,
                timeout=timeout,
                min_window=min_window,
                max_depth=max_depth,
            )
            for pv_name in pv_names
        ]
    return results


def summarize_probe(results: dict[str, list[dict]]) -> list[dict]:
    """Flatten a probe result mapping into one summary row per PV."""
    summary = []
    for subgroup_name, pv_results in results.items():
        for pv_result in pv_results:
            first_success = pv_result["first_success"]
            smallest_success = pv_result["smallest_success"]
            smallest_failure = pv_result["smallest_failure"]
            summary.append(
                {
                    "subgroup": subgroup_name,
                    "pv_name": pv_result["pv_name"],
                    "attempt_count": len(pv_result["attempts"]),
                    "success_count": len(pv_result["successes"]),
                    "failure_count": len(pv_result["failures"]),
                    "first_success_window": (
                        str(first_success["duration"]) if first_success else None
                    ),
                    "first_success_samples": (
                        first_success["sample_count"] if first_success else None
                    ),
                    "smallest_success_window": (
                        str(smallest_success["duration"])
                        if smallest_success
                        else None
                    ),
                    "smallest_failure_window": (
                        str(smallest_failure["duration"])
                        if smallest_failure
                        else None
                    ),
                    "last_error": (
                        pv_result["failures"][-1]["error"]
                        if pv_result["failures"]
                        else None
                    ),
                }
            )
    return summary


def find_pv_result(results: dict[str, list[dict]], pv_name: str) -> dict:
    """Look up the per-PV bisect result by PV name across subgroups."""
    for pv_results in results.values():
        for pv_result in pv_results:
            if pv_result["pv_name"] == pv_name:
                return pv_result
    raise KeyError(pv_name)
