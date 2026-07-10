"""Low-level archive appliance fetch helpers.

Structural HTTP helpers used to talk to the LCLS archive appliance. Kept
separate from ``hierarchy.py`` so probe utilities and other callers can pull
raw archive data without dragging in composite-building logic.
"""

from __future__ import annotations

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.auto import tqdm

from sparklines_v2.beamlines import (
    ARCHIVER_URL,
    ARCHIVE_MAX_WORKERS,
    ARCHIVE_TIMEOUT_SECONDS,
    LOCAL_TIMEZONE,
)


class ArchiveRequestError(RuntimeError):
    """Raised when the archive appliance response is structurally invalid."""


def format_archive_time(
    value: dt.datetime | str | None, local_timezone=LOCAL_TIMEZONE
) -> str | None:
    """Convert a datetime to the archive appliance's ISO-8601 UTC format."""
    if value is None or isinstance(value, str):
        return value

    if value.tzinfo is None:
        value = value.replace(tzinfo=local_timezone)

    return value.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def unwrap_archive_payload(raw):
    """Peel a doubly-wrapped ``{"value": {"value": ...}}`` archive payload."""
    if raw is None:
        return None

    if isinstance(raw, dict):
        value = raw.get("value")
        if isinstance(value, dict) and "value" in value:
            return value["value"]

    return raw


def normalize_archive_batch(pv_names: list[str], raw_batch) -> dict[str, object | None]:
    """Normalize an archive batch response into ``{pv_name: payload | None}``."""
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
                normalized[pv_name] = unwrap_archive_payload(item)

        return normalized

    if len(pv_names) == 1:
        normalized[pv_names[0]] = unwrap_archive_payload(raw_batch)
        return normalized

    if isinstance(raw_batch, dict):
        candidate = str(raw_batch.get("pvName", "")).strip()
        if candidate in normalized:
            normalized[candidate] = unwrap_archive_payload(raw_batch)

    return normalized


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
    retry=retry_if_exception_type(
        (requests.RequestException, ArchiveRequestError, ValueError, KeyError, TypeError)
    ),
)
def fetch_single_pv_direct(
    pv_name: str,
    start: dt.datetime,
    end: dt.datetime,
    timeout: float,
    *,
    archiver_url=ARCHIVER_URL,
    local_timezone=LOCAL_TIMEZONE,
):
    """Fetch one PV from the archive; retries the request up to three times."""
    response = requests.get(
        archiver_url,
        params={
            "pv": pv_name,
            "from": format_archive_time(start, local_timezone=local_timezone),
            "to": format_archive_time(end, local_timezone=local_timezone),
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


def fetch_single_pv_timed(
    pv_name: str,
    start: dt.datetime,
    end: dt.datetime,
    timeout: float,
    *,
    archiver_url=ARCHIVER_URL,
    local_timezone=LOCAL_TIMEZONE,
):
    """Wrap ``fetch_single_pv_direct`` with per-call wall-time measurement."""
    fetch_started = time.perf_counter()
    try:
        _, raw = fetch_single_pv_direct(
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


def fetch_archive_batch(
    pv_names: list[str],
    start: dt.datetime,
    end: dt.datetime,
    *,
    archiver_url=ARCHIVER_URL,
    archive_timeout_seconds=ARCHIVE_TIMEOUT_SECONDS,
    archive_max_workers=ARCHIVE_MAX_WORKERS,
):
    """Fetch a batch of PVs concurrently and return raw payloads, errors, timings."""
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
                fetch_single_pv_timed,
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
                        raw_by_pv[pv_name] = unwrap_archive_payload(result["raw"])
                finally:
                    progress.update(1)

    return raw_by_pv, errors, {
        "per_pv_fetch_seconds": fetch_timings,
        "fetch_wall_seconds": time.perf_counter() - batch_started,
    }
