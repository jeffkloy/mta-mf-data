#!/usr/bin/env python3
"""
Calculate average headways by time period.
"""

import pandas as pd
import numpy as np
import glob

ROOSEVELT_ISLAND_STOPS = ['B06N', 'B06S']
M_SERVICE_START_DATE = '2025-12-08'

TIME_PERIODS = [
    ('12am - 5:59am', 0, 5),
    ('6am - 9:59am', 6, 9),
    ('10am - 2:59pm', 10, 14),
    ('3pm - 7:59pm', 15, 19),
    ('8pm - 11:59pm', 20, 23),
]

def load_data(data_dir='data'):
    """Load all stop_times and trips data from CSV files."""
    stop_times_files = sorted(glob.glob(f'{data_dir}/*_stop_times.csv'))
    trips_files = sorted(glob.glob(f'{data_dir}/*_trips.csv'))

    all_stop_times = []
    all_trips = []

    for st_file, tr_file in zip(stop_times_files, trips_files):
        date_str = st_file.split('/')[-1].split('_')[1]

        st_df = pd.read_csv(st_file)
        st_df = st_df[st_df['stop_id'].isin(ROOSEVELT_ISLAND_STOPS)]
        st_df['date'] = date_str

        tr_df = pd.read_csv(tr_file)
        tr_df = tr_df[tr_df['route_id'].isin(['M', 'F'])]
        tr_df['date'] = date_str

        all_stop_times.append(st_df)
        all_trips.append(tr_df)

    stop_times = pd.concat(all_stop_times, ignore_index=True)
    trips = pd.concat(all_trips, ignore_index=True)

    return stop_times, trips

def get_time_period(hour):
    """Return the time period label for a given hour."""
    for label, start, end in TIME_PERIODS:
        if start <= hour <= end:
            return label
    return None

def calculate_headways(df):
    """Calculate headways from a dataframe of arrivals."""
    df = df.sort_values(['direction_id', 'stop_id', 'arrival_datetime'])

    headways = []
    for (direction, stop), group in df.groupby(['direction_id', 'stop_id']):
        group = group.sort_values('arrival_datetime')
        hw = group['arrival_datetime'].diff().dt.total_seconds() / 60
        hours = group['arrival_datetime'].dt.hour

        for (idx, val), hour in zip(hw.items(), hours):
            if pd.notna(val) and 0 < val <= 60:  # Skip headways > 60 min
                headways.append({
                    'headway_minutes': val,
                    'hour': hour,
                    'time_period': get_time_period(hour)
                })

    return pd.DataFrame(headways)

def calculate_headways_by_direction(df):
    """Calculate headways from a dataframe, separated by direction."""
    df = df.sort_values(['stop_id', 'arrival_datetime'])

    headways = []
    for stop, group in df.groupby('stop_id'):
        group = group.sort_values('arrival_datetime')
        hw = group['arrival_datetime'].diff().dt.total_seconds() / 60
        hours = group['arrival_datetime'].dt.hour

        direction = 'Northbound' if stop == 'B06N' else 'Southbound'

        for (idx, val), hour in zip(hw.items(), hours):
            if pd.notna(val) and 0 < val <= 60:  # Skip headways > 60 min
                headways.append({
                    'headway_minutes': val,
                    'hour': hour,
                    'time_period': get_time_period(hour),
                    'direction': direction
                })

    return pd.DataFrame(headways)

def print_time_period_table(headways_df):
    """Print a table with time periods as rows and directions as columns."""
    if len(headways_df) == 0:
        print("  No data")
        return

    # Overall stats
    print(f"\n  OVERALL: {len(headways_df):,} obs, Mean: {headways_df['headway_minutes'].mean():.2f}m, Median: {headways_df['headway_minutes'].median():.2f}m")

    # Table header
    print(f"\n  {'Period':<20} {'Northbound':>20} {'Southbound':>20} {'Combined':>20}")
    print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*20}")

    for period_label, _, _ in TIME_PERIODS:
        period_data = headways_df[headways_df['time_period'] == period_label]

        if len(period_data) == 0:
            print(f"  {period_label:<20} {'--':>20} {'--':>20} {'--':>20}")
            continue

        nb_data = period_data[period_data['direction'] == 'Northbound']
        sb_data = period_data[period_data['direction'] == 'Southbound']

        nb_str = f"{nb_data['headway_minutes'].mean():.2f}m (n={len(nb_data):,})" if len(nb_data) > 0 else "--"
        sb_str = f"{sb_data['headway_minutes'].mean():.2f}m (n={len(sb_data):,})" if len(sb_data) > 0 else "--"
        comb_str = f"{period_data['headway_minutes'].mean():.2f}m (n={len(period_data):,})"

        print(f"  {period_label:<20} {nb_str:>20} {sb_str:>20} {comb_str:>20}")

def main():
    print("Loading data...")
    stop_times, trips = load_data()

    merged = stop_times.merge(trips[['trip_uid', 'route_id', 'direction_id']], on='trip_uid', how='inner')
    merged['date_parsed'] = pd.to_datetime(merged['date'])
    merged['arrival_datetime'] = pd.to_datetime(merged['arrival_time'], unit='s')
    merged['hour'] = merged['arrival_datetime'].dt.hour

    m_start = pd.to_datetime(M_SERVICE_START_DATE)

    # F BEFORE Dec 8: all times
    f_before = merged[(merged['route_id'] == 'F') & (merged['date_parsed'] < m_start)]

    # F AFTER Dec 8: 9 PM to 5 AM only (hours 21-23, 0-4)
    f_after = merged[(merged['route_id'] == 'F') &
                     (merged['date_parsed'] >= m_start) &
                     ((merged['hour'] >= 21) | (merged['hour'] <= 4))]

    # M AFTER Dec 8: 5 AM to 9 PM (hours 5-20)
    m_after = merged[(merged['route_id'] == 'M') &
                     (merged['date_parsed'] >= m_start) &
                     (merged['hour'] >= 5) & (merged['hour'] <= 20)]

    datasets = [
        ('F (before 12/8) - All times', f_before),
        ('F (after 12/8) - 9PM-5AM only', f_after),
        ('M (after 12/8) - 5AM-9PM only', m_after),
    ]

    print("\n" + "="*80)
    print("AVERAGE HEADWAYS AT ROOSEVELT ISLAND BY TIME PERIOD AND DIRECTION")
    print("(Headways > 60 min excluded)")
    print("Data range: Aug 1, 2025 - Jan 19, 2026")
    print("="*80)

    for dataset_name, df in datasets:
        print(f"\n{'─'*80}")
        print(f"{dataset_name}")
        print(f"{'─'*80}")

        headways_df = calculate_headways_by_direction(df)

        print_time_period_table(headways_df)

if __name__ == "__main__":
    main()
