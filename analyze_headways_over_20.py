#!/usr/bin/env python3
"""
Analyze headways > 20 minutes for M, F, and all other subway lines.
Creates a line graph showing daily counts of 20+ minute headways.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import glob

DATA_DIR = 'data'

def load_all_data():
    """Load all stop_times and trips data from CSV files."""
    stop_times_files = sorted(glob.glob(f'{DATA_DIR}/*_stop_times.csv'))
    trips_files = sorted(glob.glob(f'{DATA_DIR}/*_trips.csv'))

    print(f"Found {len(stop_times_files)} stop_times files and {len(trips_files)} trips files")

    all_data = []

    for st_file, tr_file in zip(stop_times_files, trips_files):
        # Extract date from filename
        date_str = st_file.split('/')[-1].split('_')[1]

        # Load trips to get route_id
        tr_df = pd.read_csv(tr_file, usecols=['trip_uid', 'route_id', 'direction_id'])

        # Load stop times
        st_df = pd.read_csv(st_file, usecols=['trip_uid', 'stop_id', 'arrival_time'])

        # Merge to get route for each arrival
        merged = st_df.merge(tr_df, on='trip_uid', how='inner')
        merged['date'] = date_str

        all_data.append(merged)

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total records: {len(combined):,}")

    return combined

def calculate_headways(df):
    """Calculate headways for each stop/route/direction combination."""
    df = df.copy()

    # Convert arrival_time to datetime
    df['arrival_datetime'] = pd.to_datetime(df['arrival_time'], unit='s')

    # Sort by route, direction, stop, and arrival time
    df = df.sort_values(['route_id', 'direction_id', 'stop_id', 'arrival_datetime'])

    # Calculate headway within each group
    df['headway_minutes'] = df.groupby(['route_id', 'direction_id', 'stop_id', 'date'])['arrival_datetime'].diff().dt.total_seconds() / 60

    # Filter out invalid headways (overnight gaps, data errors)
    # Keep headways between 1 minute and 120 minutes
    df = df[(df['headway_minutes'] > 1) & (df['headway_minutes'] < 120)]

    return df

def categorize_lines(route_id):
    """Categorize lines into M, F, or Other."""
    if route_id == 'M':
        return 'M'
    elif route_id == 'F':
        return 'F'
    else:
        return 'Other Lines'

def analyze_long_headways(df):
    """Count headways > 20 minutes by category and date."""
    # Filter for headways > 20 minutes
    long_headways = df[df['headway_minutes'] > 20].copy()

    # Categorize by line
    long_headways['category'] = long_headways['route_id'].apply(categorize_lines)

    # Count by date and category
    daily_counts = long_headways.groupby(['date', 'category']).size().unstack(fill_value=0)

    # Ensure all categories exist
    for cat in ['M', 'F', 'Other Lines']:
        if cat not in daily_counts.columns:
            daily_counts[cat] = 0

    # Sort by date
    daily_counts.index = pd.to_datetime(daily_counts.index)
    daily_counts = daily_counts.sort_index()

    return daily_counts

def plot_headways_over_20(daily_counts, output_file='headways_over_20_chart.png'):
    """Create a line graph of headways > 20 minutes with dual y-axes."""
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Define colors for each category
    colors = {
        'M': '#0039A6',      # M train blue
        'F': '#FF6B00',      # F train orange
        'Other Lines': '#888888'  # Gray for others
    }

    # Plot M and F on primary axis (left)
    lines = []
    for category in ['M', 'F']:
        if category in daily_counts.columns:
            line, = ax1.plot(daily_counts.index, daily_counts[category],
                           label=category, color=colors[category],
                           linewidth=2.5, alpha=0.9)
            lines.append(line)

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('M & F Line: Daily Count of Headways > 20 Minutes', fontsize=12, color='#333333')
    ax1.tick_params(axis='y', labelcolor='#333333')

    # Create secondary y-axis for Other Lines
    ax2 = ax1.twinx()
    if 'Other Lines' in daily_counts.columns:
        line, = ax2.plot(daily_counts.index, daily_counts['Other Lines'],
                        label='Other Lines', color=colors['Other Lines'],
                        linewidth=1.5, alpha=0.6, linestyle='--')
        lines.append(line)

    ax2.set_ylabel('Other Lines: Daily Count of Headways > 20 Minutes', fontsize=12, color='#888888')
    ax2.tick_params(axis='y', labelcolor='#888888')

    # Title
    ax1.set_title('NYC Subway Headways Over 20 Minutes\n(M Line, F Line, and All Other Lines Combined)',
                  fontsize=14, fontweight='bold')

    # Combined legend
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11)

    # Date formatting
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Grid (only on primary axis)
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_file}")

def print_summary(df, daily_counts):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY: Headways > 20 Minutes")
    print("="*70)

    # Filter for > 20 min headways
    long_headways = df[df['headway_minutes'] > 20].copy()
    long_headways['category'] = long_headways['route_id'].apply(categorize_lines)

    # Overall stats
    print(f"\nDate range: {daily_counts.index.min().strftime('%Y-%m-%d')} to {daily_counts.index.max().strftime('%Y-%m-%d')}")
    print(f"Total days: {len(daily_counts)}")

    for category in ['M', 'F', 'Other Lines']:
        cat_data = long_headways[long_headways['category'] == category]
        if len(cat_data) > 0:
            daily = daily_counts[category] if category in daily_counts.columns else pd.Series([0])
            print(f"\n{category}:")
            print(f"  Total occurrences: {len(cat_data):,}")
            print(f"  Daily average: {daily.mean():.1f}")
            print(f"  Daily max: {daily.max():.0f}")
            print(f"  Average headway (when >20min): {cat_data['headway_minutes'].mean():.1f} min")

    # All lines combined
    print("\n" + "-"*40)
    print("COMPARISON:")
    print("-"*40)

    for category in ['M', 'F']:
        if category in daily_counts.columns:
            daily_avg = daily_counts[category].mean()
            print(f"{category} line: avg {daily_avg:.1f} occurrences/day of 20+ min headways")

    if 'Other Lines' in daily_counts.columns:
        other_avg = daily_counts['Other Lines'].mean()
        print(f"Other lines combined: avg {other_avg:.1f} occurrences/day of 20+ min headways")

def main():
    print("Loading data...")
    df = load_all_data()

    print("\nCalculating headways...")
    df = calculate_headways(df)

    print("\nAnalyzing headways > 20 minutes...")
    daily_counts = analyze_long_headways(df)

    print("\nGenerating plot...")
    plot_headways_over_20(daily_counts)

    print_summary(df, daily_counts)

    print("\n" + "="*70)
    print("Analysis complete! Generated: headways_over_20_chart.png")
    print("="*70)

if __name__ == "__main__":
    main()
