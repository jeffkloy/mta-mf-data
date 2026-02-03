# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository analyzes NYC subway data from [subwaydata.nyc](https://subwaydata.nyc), focusing on M and F train performance at Roosevelt Island. The M train began serving Roosevelt Island on December 8, 2025.

**Data range**: Aug 1, 2025 - Jan 19, 2026 (172 days)

## Commands

```bash
# Run analysis scripts
python3 analyze_mf_trains.py          # M/F comparison at Roosevelt Island
python3 analyze_headways_over_20.py   # System-wide headway analysis
python3 average_headways.py           # Headway averages by time period/direction

# Download data (macOS date command syntax)
./download_data.sh          # Aug 1, 2025 - Jan 20, 2026

# Extract downloaded archives
cd data && for f in *.tar.xz; do tar -xf "$f"; done
```

## Data Files

### Raw Data (from subwaydata.nyc)
- `data/*_trips.csv`: `trip_uid, trip_id, route_id, direction_id, start_time, vehicle_id, ...`
- `data/*_stop_times.csv`: `trip_uid, stop_id, track, arrival_time, departure_time, last_observed, ...`

Downloaded from `https://subwaydata.nyc/data/subwaydatanyc_YYYY-MM-DD_csv.tar.xz`

Timestamps are Unix epoch seconds. Join on `trip_uid` to associate arrivals with routes.

### Excel Exports
- `raw_data.xlsx`: All M/F train data (3.5M stop_times rows across 4 sheets, 100K trips) - tracked via Git LFS
- `raw_data_ri.xlsx`: Roosevelt Island only (51K rows each for stop_times and trips)
- `F_M Switch - Impact on Roosevelt Island Headways.xlsx`: Before/after analysis results

## Key Constants

- Roosevelt Island stops: `B06N` (northbound), `B06S` (southbound)
- M service at Roosevelt Island: weekdays 5 AM - 9 PM only (started Dec 8, 2025)
- F serves Roosevelt Island all times including nights/weekends
- MTA line colors: M = `#0039A6` (blue), F = `#FF6B00` (orange)

## Analysis Methodology

1. Load CSVs from `data/` directory using glob patterns
2. Join stop_times with trips on `trip_uid`
3. Calculate headways as time diff between consecutive arrivals (grouped by route/direction/stop)
4. Filter invalid headways (< 1 min or > 60 min are likely data errors or overnight gaps)
5. Split by time periods: Late Night (12am-6am), AM Rush (6am-10am), Midday (10am-3pm), PM Rush (3pm-8pm), Evening (8pm-12am)
6. Compare before (Aug 1 - Dec 7, 2025) vs after (Dec 8, 2025 - Jan 19, 2026) M service switch
