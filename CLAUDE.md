# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository analyzes NYC subway data from [subwaydata.nyc](https://subwaydata.nyc), focusing on M and F train performance at Roosevelt Island. The M train began serving Roosevelt Island on December 8, 2025.

## Commands

```bash
# Run analysis scripts (use venv python)
./venv/bin/python analyze_mf_trains.py          # M/F comparison at Roosevelt Island
./venv/bin/python analyze_headways_over_20.py   # System-wide headway analysis

# Download data (macOS date command syntax)
./download_data.sh       # Nov 2025 - Jan 2026
./download_sept_oct.sh   # Sep - Oct 2025

# Extract downloaded archives
cd data && for f in *.tar.xz; do tar -xf "$f"; done
```

## Data Source

Data files are downloaded from `https://subwaydata.nyc/data/subwaydatanyc_YYYY-MM-DD_csv.tar.xz`

Each archive contains two CSVs:
- `*_trips.csv`: `trip_uid, trip_id, route_id, direction_id, start_time, vehicle_id, ...`
- `*_stop_times.csv`: `trip_uid, stop_id, track, arrival_time, departure_time, last_observed, ...`

Timestamps are Unix epoch seconds. Join on `trip_uid` to associate arrivals with routes.

## Key Constants

- Roosevelt Island stops: `B06N` (northbound), `B06S` (southbound)
- M service at Roosevelt Island: weekdays 5 AM - 9 PM only (started Dec 8, 2025)
- F serves Roosevelt Island all times including nights/weekends
- MTA line colors: M = `#0039A6` (blue), F = `#FF6B00` (orange)

## Architecture

Analysis scripts follow a common pattern:
1. Load CSVs from `data/` directory using glob patterns
2. Join stop_times with trips on `trip_uid`
3. Calculate headways as time diff between consecutive arrivals (grouped by route/direction/stop)
4. Filter invalid headways (< 1 min or > 120 min are likely data errors or overnight gaps)
5. Generate matplotlib visualizations saved as PNG files
