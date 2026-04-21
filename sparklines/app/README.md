# MEME Sparklines Desktop App

This folder contains a PyQt5 desktop wrapper around the Matplotlib-based
sparkline hierarchy viewer.

## What It Does

- Loads the real hierarchy from `app/pv_groups.yaml`
- Loads monitor PV definitions from `app/monitor_pvs.yaml`
- Embeds the existing Matplotlib viewer inside a `PyQt5` main window
- Includes the standard Matplotlib navigation toolbar
- Loads archive data in a background Qt thread so the UI does not freeze during fetch/build
- Writes draw timing logs to `app/sparklines_draw_report.txt`

## Run

From the repo root:

```bash
uv run python app/main.py
```

You can also override the initial time range:

```bash
uv run python app/main.py --start "2026-03-30 22:00:00" --end "2026-03-31 06:00:00"
```

Or use a default rolling window:

```bash
uv run python app/main.py --hours 12
```

## Notes

- The app uses the same hierarchy builder and Matplotlib viewer as the notebook workflow.
- Monitor quantile series are cached inside the viewer so redraws can reuse previously computed monitor bands.
