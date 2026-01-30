#!/usr/bin/env python3
"""
Analyze M and F train frequency and delays at Roosevelt Island (Stop ID: B06)
Note: M train service to Roosevelt Island started December 8, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import glob
from collections import defaultdict

# Roosevelt Island stop IDs (Northbound and Southbound)
ROOSEVELT_ISLAND_STOPS = ['B06N', 'B06S']

# M train service started at Roosevelt Island on this date
M_SERVICE_START_DATE = '2025-12-08'

# M train only runs at Roosevelt Island during weekday daytime (Mon-Fri, before ~9:30 PM)
# F train serves Roosevelt Island on weekends and weekday nights after ~9:30 PM
M_SERVICE_END_HOUR = 21  # 9 PM - M stops serving after this
M_SERVICE_START_HOUR = 5  # 5 AM - M starts serving around this time

def load_data(data_dir='data'):
    """Load all stop_times and trips data from CSV files."""
    stop_times_files = sorted(glob.glob(f'{data_dir}/*_stop_times.csv'))
    trips_files = sorted(glob.glob(f'{data_dir}/*_trips.csv'))

    print(f"Found {len(stop_times_files)} stop_times files and {len(trips_files)} trips files")

    all_stop_times = []
    all_trips = []

    for st_file, tr_file in zip(stop_times_files, trips_files):
        # Extract date from filename
        date_str = st_file.split('_')[1]

        # Load stop times for Roosevelt Island only
        st_df = pd.read_csv(st_file)
        st_df = st_df[st_df['stop_id'].isin(ROOSEVELT_ISLAND_STOPS)]
        st_df['date'] = date_str

        # Load trips for M and F trains only
        tr_df = pd.read_csv(tr_file)
        tr_df = tr_df[tr_df['route_id'].isin(['M', 'F'])]
        tr_df['date'] = date_str

        all_stop_times.append(st_df)
        all_trips.append(tr_df)

    stop_times = pd.concat(all_stop_times, ignore_index=True)
    trips = pd.concat(all_trips, ignore_index=True)

    print(f"Total stop times at Roosevelt Island: {len(stop_times)}")
    print(f"Total M/F trips: {len(trips)}")

    return stop_times, trips

def merge_and_filter(stop_times, trips):
    """Merge stop times with trips to get M/F trains at Roosevelt Island."""
    # Merge on trip_uid
    merged = stop_times.merge(trips[['trip_uid', 'route_id', 'direction_id', 'start_time']],
                               on='trip_uid', how='inner')

    # Filter out M train data before service started at Roosevelt Island
    m_start = pd.to_datetime(M_SERVICE_START_DATE)
    merged['date_parsed'] = pd.to_datetime(merged['date'])
    pre_filter_count = len(merged)
    merged = merged[~((merged['route_id'] == 'M') & (merged['date_parsed'] < m_start))]
    post_filter_count = len(merged)

    print(f"M/F trains at Roosevelt Island: {post_filter_count}")
    print(f"  (Filtered out {pre_filter_count - post_filter_count} M train records before {M_SERVICE_START_DATE})")
    print(f"Route breakdown:")
    print(merged['route_id'].value_counts())

    return merged

def add_service_period_info(df):
    """Add columns to identify when M train serves Roosevelt Island vs F-only periods."""
    df = df.copy()

    # Convert arrival_time to datetime if not already
    if 'arrival_datetime' not in df.columns:
        df['arrival_datetime'] = pd.to_datetime(df['arrival_time'], unit='s')

    # Get day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['arrival_datetime'].dt.dayofweek
    df['hour'] = df['arrival_datetime'].dt.hour

    # M train serves Roosevelt Island: Weekdays (Mon-Fri), roughly 5 AM - 9:30 PM
    # F train serves Roosevelt Island: Weekends + Weekday nights after ~9:30 PM
    df['is_weekday'] = df['day_of_week'] < 5  # Mon-Fri
    df['is_m_service_hours'] = (df['hour'] >= M_SERVICE_START_HOUR) & (df['hour'] < M_SERVICE_END_HOUR)
    df['m_service_period'] = df['is_weekday'] & df['is_m_service_hours']

    # Service period labels
    df['service_period'] = df.apply(
        lambda row: 'Weekday Daytime (M+F)' if row['m_service_period']
        else ('Weekend' if not row['is_weekday'] else 'Weekday Night (F only)'),
        axis=1
    )

    return df

def calculate_frequency(df):
    """Calculate train frequency (headway) in minutes."""
    df = df.copy()

    # Convert arrival_time to datetime
    df['arrival_datetime'] = pd.to_datetime(df['arrival_time'], unit='s')
    df['date_parsed'] = pd.to_datetime(df['date'])

    # Add service period info
    df = add_service_period_info(df)

    # Sort by route, direction, and arrival time
    df = df.sort_values(['route_id', 'direction_id', 'arrival_datetime'])

    # Calculate headway (time since previous train)
    freq_data = []

    for (route, direction, stop), group in df.groupby(['route_id', 'direction_id', 'stop_id']):
        group = group.sort_values('arrival_datetime')
        headways = group['arrival_datetime'].diff().dt.total_seconds() / 60  # Convert to minutes

        for idx, row in group.iterrows():
            headway = headways.loc[idx] if idx in headways.index else np.nan
            # Filter out unreasonable headways (> 2 hours likely overnight gap)
            if pd.notna(headway) and 0 < headway < 120:
                freq_data.append({
                    'route_id': route,
                    'direction_id': direction,
                    'stop_id': stop,
                    'date': row['date'],
                    'arrival_datetime': row['arrival_datetime'],
                    'headway_minutes': headway,
                    'hour': row['arrival_datetime'].hour,
                    'day_of_week': row['day_of_week'],
                    'is_weekday': row['is_weekday'],
                    'm_service_period': row['m_service_period'],
                    'service_period': row['service_period']
                })

    return pd.DataFrame(freq_data)

def calculate_delays(df):
    """
    Calculate delays based on the difference between scheduled arrival and actual observation.
    arrival_time: scheduled arrival (Unix timestamp)
    last_observed: when the train was actually observed at the stop
    """
    df = df.copy()

    # Convert to datetime
    df['scheduled_arrival'] = pd.to_datetime(df['arrival_time'], unit='s')
    df['actual_observation'] = pd.to_datetime(df['last_observed'], unit='s')

    # Add service period info
    df['arrival_datetime'] = df['scheduled_arrival']
    df = add_service_period_info(df)

    # Calculate delay in minutes (positive = late, negative = early)
    df['delay_minutes'] = (df['actual_observation'] - df['scheduled_arrival']).dt.total_seconds() / 60

    # Filter out extreme values (likely data errors)
    df = df[(df['delay_minutes'] > -30) & (df['delay_minutes'] < 60)]

    return df

def plot_frequency_comparison(freq_df, output_dir='.'):
    """Create frequency comparison plots for M vs F trains."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('M vs F Train Frequency at Roosevelt Island\n(Sep 2025 - Jan 2026; M service started Dec 8, 2025)', fontsize=14, fontweight='bold')

    # 1. Average headway by route
    ax1 = axes[0, 0]
    avg_headway = freq_df.groupby('route_id')['headway_minutes'].mean()
    colors = ['#FF6B00' if r == 'F' else '#0039A6' for r in avg_headway.index]
    bars = ax1.bar(avg_headway.index, avg_headway.values, color=colors, edgecolor='black')
    ax1.set_ylabel('Average Headway (minutes)')
    ax1.set_xlabel('Train Line')
    ax1.set_title('Average Headway by Line')
    for bar, val in zip(bars, avg_headway.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    # 2. Headway by hour of day
    ax2 = axes[0, 1]
    for route in ['F', 'M']:
        route_data = freq_df[freq_df['route_id'] == route]
        hourly = route_data.groupby('hour')['headway_minutes'].mean()
        color = '#FF6B00' if route == 'F' else '#0039A6'
        ax2.plot(hourly.index, hourly.values, marker='o', label=route, color=color, linewidth=2)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average Headway (minutes)')
    ax2.set_title('Headway by Hour of Day')
    ax2.legend()
    ax2.set_xticks(range(0, 24, 3))
    ax2.grid(True, alpha=0.3)

    # 3. Daily average headway over time
    ax3 = axes[1, 0]
    for route in ['F', 'M']:
        route_data = freq_df[freq_df['route_id'] == route]
        daily = route_data.groupby('date')['headway_minutes'].mean()
        dates = pd.to_datetime(daily.index)
        color = '#FF6B00' if route == 'F' else '#0039A6'
        ax3.plot(dates, daily.values, label=route, color=color, alpha=0.7, linewidth=1.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Average Headway (minutes)')
    ax3.set_title('Daily Average Headway Over Time')
    ax3.legend()
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    # 4. Headway distribution
    ax4 = axes[1, 1]
    for route in ['F', 'M']:
        route_data = freq_df[freq_df['route_id'] == route]['headway_minutes']
        color = '#FF6B00' if route == 'F' else '#0039A6'
        ax4.hist(route_data, bins=30, alpha=0.5, label=route, color=color, edgecolor='black')
    ax4.set_xlabel('Headway (minutes)')
    ax4.set_ylabel('Count')
    ax4.set_title('Headway Distribution')
    ax4.legend()
    ax4.set_xlim(0, 30)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/frequency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved frequency_comparison.png")

def plot_delay_comparison(delay_df, output_dir='.'):
    """Create delay comparison plots for M vs F trains."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('M vs F Train Delays at Roosevelt Island\n(Sep 2025 - Jan 2026; M service started Dec 8, 2025)', fontsize=14, fontweight='bold')

    # 1. Average delay by route
    ax1 = axes[0, 0]
    avg_delay = delay_df.groupby('route_id')['delay_minutes'].mean()
    colors = ['#FF6B00' if r == 'F' else '#0039A6' for r in avg_delay.index]
    bars = ax1.bar(avg_delay.index, avg_delay.values, color=colors, edgecolor='black')
    ax1.set_ylabel('Average Delay (minutes)')
    ax1.set_xlabel('Train Line')
    ax1.set_title('Average Delay by Line')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, avg_delay.values):
        y_pos = bar.get_height() + 0.1 if val >= 0 else bar.get_height() - 0.3
        ax1.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:.1f}', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')

    # 2. Delay by hour of day
    ax2 = axes[0, 1]
    delay_df['hour'] = delay_df['scheduled_arrival'].dt.hour
    for route in ['F', 'M']:
        route_data = delay_df[delay_df['route_id'] == route]
        hourly = route_data.groupby('hour')['delay_minutes'].mean()
        color = '#FF6B00' if route == 'F' else '#0039A6'
        ax2.plot(hourly.index, hourly.values, marker='o', label=route, color=color, linewidth=2)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average Delay (minutes)')
    ax2.set_title('Delay by Hour of Day')
    ax2.legend()
    ax2.set_xticks(range(0, 24, 3))
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # 3. Daily average delay over time
    ax3 = axes[1, 0]
    for route in ['F', 'M']:
        route_data = delay_df[delay_df['route_id'] == route]
        daily = route_data.groupby('date')['delay_minutes'].mean()
        dates = pd.to_datetime(daily.index)
        color = '#FF6B00' if route == 'F' else '#0039A6'
        ax3.plot(dates, daily.values, label=route, color=color, alpha=0.7, linewidth=1.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Average Delay (minutes)')
    ax3.set_title('Daily Average Delay Over Time')
    ax3.legend()
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    # 4. Delay distribution
    ax4 = axes[1, 1]
    for route in ['F', 'M']:
        route_data = delay_df[delay_df['route_id'] == route]['delay_minutes']
        color = '#FF6B00' if route == 'F' else '#0039A6'
        ax4.hist(route_data, bins=50, alpha=0.5, label=route, color=color, edgecolor='black')
    ax4.set_xlabel('Delay (minutes)')
    ax4.set_ylabel('Count')
    ax4.set_title('Delay Distribution (negative = early)')
    ax4.legend()
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/delay_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved delay_comparison.png")

def plot_overlapping_period_comparison(freq_df, delay_df, output_dir='.'):
    """Create focused comparison for the period when both M and F serve Roosevelt Island.

    IMPORTANT: M train only serves Roosevelt Island during weekday daytime (Mon-Fri, ~5AM-9PM).
    F train serves all times, including weekends and weekday nights.
    This comparison focuses on weekday daytime when BOTH trains run (apples-to-apples).
    """
    m_start = pd.to_datetime(M_SERVICE_START_DATE)

    # Filter for overlapping period only (Dec 8+)
    freq_overlap = freq_df[pd.to_datetime(freq_df['date']) >= m_start].copy()
    delay_overlap = delay_df[pd.to_datetime(delay_df['date']) >= m_start].copy()

    # APPLES-TO-APPLES: Only compare during weekday daytime when both M and F run
    freq_weekday_day = freq_overlap[freq_overlap['m_service_period'] == True].copy()
    delay_weekday_day = delay_overlap[delay_overlap['m_service_period'] == True].copy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('M vs F Train Comparison at Roosevelt Island\n(Weekday Daytime Only: Mon-Fri 5AM-9PM, Dec 8 2025 - Jan 20 2026)',
                 fontsize=14, fontweight='bold')

    # 1. Average headway comparison (weekday daytime only)
    ax1 = axes[0, 0]
    avg_headway = freq_weekday_day.groupby('route_id')['headway_minutes'].mean()
    colors = ['#FF6B00' if r == 'F' else '#0039A6' for r in avg_headway.index]
    bars = ax1.bar(avg_headway.index, avg_headway.values, color=colors, edgecolor='black')
    ax1.set_ylabel('Average Headway (minutes)')
    ax1.set_title('Avg Headway (Weekday Daytime)')
    for bar, val in zip(bars, avg_headway.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    # 2. Average delay comparison (weekday daytime only)
    ax2 = axes[0, 1]
    avg_delay = delay_weekday_day.groupby('route_id')['delay_minutes'].mean()
    colors = ['#FF6B00' if r == 'F' else '#0039A6' for r in avg_delay.index]
    bars = ax2.bar(avg_delay.index, avg_delay.values, color=colors, edgecolor='black')
    ax2.set_ylabel('Average Delay (minutes)')
    ax2.set_title('Avg Delay (Weekday Daytime)')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, avg_delay.values):
        y_pos = bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.1
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')

    # 3. Train counts (weekday daytime only)
    ax3 = axes[0, 2]
    train_counts = freq_weekday_day.groupby('route_id').size()
    colors = ['#FF6B00' if r == 'F' else '#0039A6' for r in train_counts.index]
    bars = ax3.bar(train_counts.index, train_counts.values, color=colors, edgecolor='black')
    ax3.set_ylabel('Number of Trains')
    ax3.set_title('Train Count (Weekday Daytime)')
    for bar, val in zip(bars, train_counts.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom', fontweight='bold')

    # 4. Headway by hour (weekday daytime - both lines)
    ax4 = axes[1, 0]
    for route in ['F', 'M']:
        route_data = freq_weekday_day[freq_weekday_day['route_id'] == route]
        hourly = route_data.groupby('hour')['headway_minutes'].mean()
        color = '#FF6B00' if route == 'F' else '#0039A6'
        ax4.plot(hourly.index, hourly.values, marker='o', label=route, color=color, linewidth=2)
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Average Headway (minutes)')
    ax4.set_title('Headway by Hour (Weekday)')
    ax4.legend()
    ax4.set_xticks(range(5, 22, 2))
    ax4.axvline(x=M_SERVICE_END_HOUR, color='red', linestyle=':', alpha=0.5, label='M service ends')
    ax4.grid(True, alpha=0.3)

    # 5. Delay by hour (weekday daytime - both lines)
    ax5 = axes[1, 1]
    for route in ['F', 'M']:
        route_data = delay_weekday_day[delay_weekday_day['route_id'] == route]
        hourly = route_data.groupby('hour')['delay_minutes'].mean()
        color = '#FF6B00' if route == 'F' else '#0039A6'
        ax5.plot(hourly.index, hourly.values, marker='o', label=route, color=color, linewidth=2)
    ax5.set_xlabel('Hour of Day')
    ax5.set_ylabel('Average Delay (minutes)')
    ax5.set_title('Delay by Hour (Weekday)')
    ax5.legend()
    ax5.set_xticks(range(5, 22, 2))
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax5.grid(True, alpha=0.3)

    # 6. Reliability comparison (weekday daytime only)
    ax6 = axes[1, 2]
    reliability = {}
    for route in ['F', 'M']:
        route_data = delay_weekday_day[delay_weekday_day['route_id'] == route]['delay_minutes']
        if len(route_data) > 0:
            reliability[route] = (abs(route_data) <= 2).mean() * 100
    colors = ['#FF6B00' if r == 'F' else '#0039A6' for r in reliability.keys()]
    bars = ax6.bar(reliability.keys(), reliability.values(), color=colors, edgecolor='black')
    ax6.set_ylabel('% On Time (within 2 min)')
    ax6.set_title('Reliability (Weekday Daytime)')
    ax6.set_ylim(95, 100)
    for bar, val in zip(bars, reliability.values()):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/weekday_daytime_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved weekday_daytime_comparison.png")

    # Also create a service period breakdown chart
    plot_service_period_breakdown(freq_overlap, delay_overlap, output_dir)


def plot_service_period_breakdown(freq_df, delay_df, output_dir='.'):
    """Show how F train performs across different service periods (weekday day vs night vs weekend)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('F Train Performance by Service Period at Roosevelt Island\n(Dec 8, 2025 - Jan 20, 2026)',
                 fontsize=14, fontweight='bold')

    # Filter F train only
    freq_f = freq_df[freq_df['route_id'] == 'F'].copy()
    delay_f = delay_df[delay_df['route_id'] == 'F'].copy()

    # Define colors for service periods
    period_colors = {
        'Weekday Daytime (M+F)': '#4CAF50',  # Green - both trains
        'Weekday Night (F only)': '#FF9800',  # Orange - F only
        'Weekend': '#2196F3'  # Blue - F only (weekend)
    }

    # 1. F train headway by service period
    ax1 = axes[0, 0]
    period_headway = freq_f.groupby('service_period')['headway_minutes'].mean()
    colors = [period_colors.get(p, 'gray') for p in period_headway.index]
    bars = ax1.bar(range(len(period_headway)), period_headway.values, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(period_headway)))
    ax1.set_xticklabels([p.replace(' (M+F)', '\n(M+F)').replace(' (F only)', '\n(F only)') for p in period_headway.index], fontsize=9)
    ax1.set_ylabel('Average Headway (minutes)')
    ax1.set_title('F Train Headway by Service Period')
    for bar, val in zip(bars, period_headway.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    # 2. F train delay by service period
    ax2 = axes[0, 1]
    period_delay = delay_f.groupby('service_period')['delay_minutes'].mean()
    colors = [period_colors.get(p, 'gray') for p in period_delay.index]
    bars = ax2.bar(range(len(period_delay)), period_delay.values, color=colors, edgecolor='black')
    ax2.set_xticks(range(len(period_delay)))
    ax2.set_xticklabels([p.replace(' (M+F)', '\n(M+F)').replace(' (F only)', '\n(F only)') for p in period_delay.index], fontsize=9)
    ax2.set_ylabel('Average Delay (minutes)')
    ax2.set_title('F Train Delay by Service Period')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, period_delay.values):
        y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.05
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')

    # 3. Train counts by service period (both M and F)
    ax3 = axes[1, 0]
    period_counts = freq_df.groupby(['service_period', 'route_id']).size().unstack(fill_value=0)
    x = np.arange(len(period_counts.index))
    width = 0.35
    if 'F' in period_counts.columns:
        ax3.bar(x - width/2, period_counts['F'], width, label='F', color='#FF6B00', edgecolor='black')
    if 'M' in period_counts.columns:
        ax3.bar(x + width/2, period_counts['M'], width, label='M', color='#0039A6', edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels([p.replace(' (M+F)', '\n(M+F)').replace(' (F only)', '\n(F only)') for p in period_counts.index], fontsize=9)
    ax3.set_ylabel('Number of Trains')
    ax3.set_title('Train Counts by Service Period')
    ax3.legend()

    # 4. Combined frequency: effective headway when both trains run vs F only
    ax4 = axes[1, 1]

    # Calculate combined headway during weekday daytime (when M+F both run)
    weekday_day_data = freq_df[freq_df['m_service_period'] == True].copy()
    if len(weekday_day_data) > 0:
        # Combined: sort all arrivals and calculate time between consecutive trains (any line)
        weekday_day_data = weekday_day_data.sort_values('arrival_datetime')
        combined_headways = weekday_day_data.groupby(['date', 'direction_id']).apply(
            lambda g: g.sort_values('arrival_datetime')['arrival_datetime'].diff().dt.total_seconds() / 60
        ).dropna()
        combined_headways = combined_headways[(combined_headways > 0) & (combined_headways < 60)]
        combined_avg = combined_headways.mean()

        # F-only periods
        f_only_data = freq_df[(freq_df['m_service_period'] == False) & (freq_df['route_id'] == 'F')]
        f_only_avg = f_only_data['headway_minutes'].mean() if len(f_only_data) > 0 else 0

        labels = ['Weekday Daytime\n(M+F Combined)', 'Nights/Weekends\n(F Only)']
        values = [combined_avg, f_only_avg]
        colors = ['#4CAF50', '#FF6B00']
        bars = ax4.bar(labels, values, color=colors, edgecolor='black')
        ax4.set_ylabel('Effective Headway (minutes)')
        ax4.set_title('Effective Wait Time: M+F vs F Only')
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/service_period_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved service_period_breakdown.png")

def print_summary_stats(freq_df, delay_df):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS: M vs F Trains at Roosevelt Island")
    print("="*70)

    # Full period stats
    print("\n--- FULL PERIOD STATS ---")
    print(f"F Train: Sep 2025 - Jan 2026 (full dataset)")
    print(f"M Train: Dec 8, 2025 - Jan 2026 (service started Dec 8)")

    print("\n--- FREQUENCY (Headway in minutes) ---")
    for route in ['F', 'M']:
        route_data = freq_df[freq_df['route_id'] == route]['headway_minutes']
        if len(route_data) > 0:
            print(f"\n{route} Train:")
            print(f"  Mean headway:   {route_data.mean():.1f} min")
            print(f"  Median headway: {route_data.median():.1f} min")
            print(f"  Std deviation:  {route_data.std():.1f} min")
            print(f"  Min headway:    {route_data.min():.1f} min")
            print(f"  Max headway:    {route_data.max():.1f} min")
            print(f"  Total observations: {len(route_data):,}")

    print("\n--- DELAYS (in minutes, positive = late) ---")
    for route in ['F', 'M']:
        route_data = delay_df[delay_df['route_id'] == route]['delay_minutes']
        if len(route_data) > 0:
            print(f"\n{route} Train:")
            print(f"  Mean delay:     {route_data.mean():.2f} min")
            print(f"  Median delay:   {route_data.median():.2f} min")
            print(f"  Std deviation:  {route_data.std():.2f} min")
            print(f"  % on time (within 2 min): {(abs(route_data) <= 2).mean()*100:.1f}%")
            print(f"  % delayed (> 5 min):      {(route_data > 5).mean()*100:.1f}%")
            print(f"  Total observations: {len(route_data):,}")

    # Overlapping period comparison
    m_start = pd.to_datetime(M_SERVICE_START_DATE)
    freq_overlap = freq_df[pd.to_datetime(freq_df['date']) >= m_start]
    delay_overlap = delay_df[pd.to_datetime(delay_df['date']) >= m_start]

    print("\n" + "="*70)
    print("APPLES-TO-APPLES: WEEKDAY DAYTIME COMPARISON (Dec 8 - Jan 20)")
    print("(Mon-Fri, 5AM-9PM when BOTH M and F serve Roosevelt Island)")
    print("="*70)

    # Filter for weekday daytime only
    freq_weekday = freq_overlap[freq_overlap['m_service_period'] == True]
    delay_weekday = delay_overlap[delay_overlap['m_service_period'] == True]

    print("\n--- FREQUENCY (Weekday Daytime) ---")
    for route in ['F', 'M']:
        route_data = freq_weekday[freq_weekday['route_id'] == route]['headway_minutes']
        if len(route_data) > 0:
            print(f"{route} Train: Mean={route_data.mean():.1f}min, Median={route_data.median():.1f}min, Std={route_data.std():.1f}min (n={len(route_data):,})")

    print("\n--- DELAYS (Weekday Daytime) ---")
    for route in ['F', 'M']:
        route_data = delay_weekday[delay_weekday['route_id'] == route]['delay_minutes']
        if len(route_data) > 0:
            on_time_pct = (abs(route_data) <= 2).mean() * 100
            print(f"{route} Train: Mean={route_data.mean():.2f}min, On-time={on_time_pct:.1f}% (n={len(route_data):,})")

    # Combined effective headway
    print("\n--- COMBINED SERVICE (Weekday Daytime) ---")
    weekday_day_all = freq_weekday.sort_values('arrival_datetime')
    combined_headways = weekday_day_all.groupby(['date', 'direction_id']).apply(
        lambda g: g.sort_values('arrival_datetime')['arrival_datetime'].diff().dt.total_seconds() / 60
    ).dropna()
    combined_headways = combined_headways[(combined_headways > 0) & (combined_headways < 60)]
    if len(combined_headways) > 0:
        print(f"Effective combined headway (M+F): {combined_headways.mean():.1f} min average")

    print("\n" + "="*70)
    print("F TRAIN ONLY PERIODS")
    print("(Weekday nights after 9PM + Weekends)")
    print("="*70)

    # F-only periods
    freq_f_only = freq_overlap[(freq_overlap['m_service_period'] == False) & (freq_overlap['route_id'] == 'F')]
    delay_f_only = delay_overlap[(delay_overlap['m_service_period'] == False) & (delay_overlap['route_id'] == 'F')]

    if len(freq_f_only) > 0:
        print(f"\nF Train (nights/weekends): Mean={freq_f_only['headway_minutes'].mean():.1f}min, Median={freq_f_only['headway_minutes'].median():.1f}min (n={len(freq_f_only):,})")
    if len(delay_f_only) > 0:
        on_time_pct = (abs(delay_f_only['delay_minutes']) <= 2).mean() * 100
        print(f"F Train delays: Mean={delay_f_only['delay_minutes'].mean():.2f}min, On-time={on_time_pct:.1f}%")

def main():
    print("Loading data...")
    stop_times, trips = load_data()

    print("\nMerging M/F trains with Roosevelt Island stops...")
    merged = merge_and_filter(stop_times, trips)

    if len(merged) == 0:
        print("ERROR: No M/F trains found at Roosevelt Island. Checking data...")
        print("Stop IDs in stop_times:", stop_times['stop_id'].unique()[:20])
        print("Route IDs in trips:", trips['route_id'].unique())
        return

    print("\nCalculating frequency...")
    freq_df = calculate_frequency(merged)

    print("\nCalculating delays...")
    delay_df = calculate_delays(merged)

    print("\nGenerating plots...")
    plot_frequency_comparison(freq_df)
    plot_delay_comparison(delay_df)
    plot_overlapping_period_comparison(freq_df, delay_df)

    print_summary_stats(freq_df, delay_df)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("Generated files:")
    print("  - frequency_comparison.png (full period, all times)")
    print("  - delay_comparison.png (full period, all times)")
    print("  - weekday_daytime_comparison.png (M vs F, Mon-Fri 5AM-9PM)")
    print("  - service_period_breakdown.png (F train by service period)")
    print("="*70)

if __name__ == "__main__":
    main()
