import pandas as pd
import numpy as np
import os

def extract_24h_count(df):
    # Load the data

    # Ensure 'date' is datetime (in case it's not)
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Group by id, day, and location_name, then count
    counts = df.groupby(['id', 'date', 'location']).size().reset_index(name='count')

    # Pivot to get location_name as columns
    pivot_df = counts.pivot_table(
        index=['id', 'date'],
        columns='location',
        values='count',
        fill_value=0
    ).reset_index()

    # Optional: flatten column names if needed (e.g., after pivot)
    pivot_df.columns.name = None  # remove hierarchical column name


    # Expected location columns
    expected_locations = ['back-door', 'bathroom', 'bedroom', 'fridge-door',
                          'front-door', 'hallway', 'kitchen', 'lounge']

    # Ensure all location columns are present
    for loc in expected_locations:
        if loc not in pivot_df.columns:
            pivot_df[loc] = 0


    return pivot_df  # if you want to inspect it in code


def extract_24h_contextual(df):

    df['date'] = pd.to_datetime(df['date']).dt.date
    counts = df.groupby(['id', 'date', 'location']).size().reset_index(name='count')
    pivot_df = counts.pivot_table(
        index=['id', 'date'],
        columns='location',
        values='count',
        fill_value=0
    ).reset_index()
    pivot_df.columns.name = None

    # Expected location columns
    expected_locations = ['back-door', 'bathroom', 'bedroom', 'fridge-door',
                          'front-door', 'hallway', 'kitchen', 'lounge']

    # Ensure all location columns are present
    for loc in expected_locations:
        if loc not in pivot_df.columns:
            pivot_df[loc] = 0

    # Private and public locations
    private = ['bedroom', 'bathroom']
    public = ['kitchen', 'hallway', 'lounge']

    # Activity features
    pivot_df['total-events'] = pivot_df[expected_locations].sum(axis=1)
    pivot_df['unique-locations'] = pivot_df[expected_locations].gt(0).sum(axis=1)
    pivot_df['active-location-ratio'] = pivot_df['unique-locations'] / len(expected_locations)

    private_sum = pivot_df[private].sum(axis=1)
    public_sum = pivot_df[public].sum(axis=1)
    pivot_df['private-to-public-ratio'] = np.where(
        public_sum == 0,
        1.0,
        (private_sum / public_sum).round(4)
    )

    # Entropy function
    def entropy(row):
        probs = row[row > 0] / row.sum()
        return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0

    pivot_df['location-entropy'] = pivot_df[expected_locations].apply(entropy, axis=1)
    pivot_df['location-dominance-ratio'] = pivot_df[expected_locations].max(axis=1) / (pivot_df['total-events'] + 1e-5)

    # Sort original df for transition features
    df_sorted = df.sort_values(by=['id', 'date', 'date-time'])
    df_sorted['prev-location'] = df_sorted.groupby(['id', 'date'])['location'].shift(1)
    df_sorted['prev2-location'] = df_sorted.groupby(['id', 'date'])['location'].shift(2)
    df_sorted['back-and-forth'] = ((df_sorted['prev2-location'] == df_sorted['location']) &
                                   (df_sorted['location'] != df_sorted['prev-location'])).astype(int)

    # Aggregate transition features
    backforth_counts = df_sorted.groupby(['id', 'date'])['back-and-forth'].sum().reset_index()
    transition_counts = df_sorted.groupby(['id', 'date']).apply(
        lambda x: (x['location'] != x['prev-location']).sum()
    ).reset_index(name='num-transitions')

    pivot_df = pivot_df.merge(backforth_counts, on=['id', 'date'], how='left')
    pivot_df = pivot_df.merge(transition_counts, on=['id', 'date'], how='left')
    pivot_df.rename(columns={'back-and-forth': 'back-and-forth-count'}, inplace=True)

    # Fill NaNs with 0 and convert to int
    pivot_df['back-and-forth-count'] = pivot_df['back-and-forth-count'].fillna(0).astype(int)
    pivot_df['num-transitions'] = pivot_df['num-transitions'].fillna(0).astype(int)

    # # Restlessness Flag
    # pivot_df['restlessness-flag'] = np.where(
    #     (pivot_df['num-transitions'] >= 10) |
    #     (pivot_df['back-and-forth-count'] >= 3) |
    #     (pivot_df['location-entropy'] >= 1.5),
    #     1, 0
    # )

    # Format: floats to 4 decimal points, integers as int
    for col in pivot_df.select_dtypes(include='float').columns:
        pivot_df[col] = pivot_df[col].round(4)

    for col in pivot_df.select_dtypes(include='number').columns:
        if (pivot_df[col] == pivot_df[col].astype(int)).all():
            pivot_df[col] = pivot_df[col].astype(int)


    pivot_df = pivot_df.drop(columns=expected_locations)


    return pivot_df


def extract_24h_statistical(df):
    expected_locations = ['back-door', 'bathroom', 'bedroom', 'fridge-door',
                          'front-door', 'hallway', 'kitchen', 'lounge']
    expected_stats = ['sum', 'mean', 'max', 'std']

    # Ensure datetime format
    df['date'] = pd.to_datetime(df['date-time']).dt.date
    df['hour'] = df['date-time'].dt.hour

    # Count events per hour
    hourly_counts = df.groupby(['id', 'date', 'hour', 'location']).size().reset_index(name='count')

    # Aggregate stats per location per day
    stats_df = hourly_counts.groupby(['id', 'date', 'location'])['count'].agg(expected_stats).reset_index()

    # Reshape long
    stats_long = stats_df.melt(
        id_vars=['id', 'date', 'location'],
        value_vars=expected_stats,
        var_name='stat',
        value_name='value'
    )

    # Create column names like bathroom-count-mean
    stats_long['feature'] = stats_long['location'] + '-count-' + stats_long['stat']

    # Pivot to wide format
    stats_pivot = stats_long.pivot_table(
        index=['id', 'date'],
        columns='feature',
        values='value'
    ).reset_index()

    stats_pivot.columns.name = None

    # Ensure all expected features are present — fill missing ones with NaN
    all_expected_features = [loc + '-count-' + stat for loc in expected_locations for stat in expected_stats]
    for feature in all_expected_features:
        if feature not in stats_pivot.columns:
            stats_pivot[feature] = pd.NA


    # Sort columns (optional for consistency)
    feature_cols = sorted([col for col in stats_pivot.columns if col not in ['id', 'date']])
    stats_pivot = stats_pivot[['id', 'date'] + feature_cols]

    # Round float columns
    for col in stats_pivot.select_dtypes(include='float').columns:
        stats_pivot[col] = stats_pivot[col].round(4)

    # Convert to Int64 if all values are integer-like
    for col in stats_pivot.select_dtypes(include='number').columns:
        col_data = stats_pivot[col].dropna()
        if col_data.apply(lambda x: float(x).is_integer()).all():
            stats_pivot[col] = stats_pivot[col].astype('Int64')  # nullable int

    return stats_pivot


def extract_24h_physiology(df):

    # Convert to datetime if needed
    df['date-time'] = pd.to_datetime(df['date-time'])
    df['date'] = df['date-time'].dt.date

    # List of expected device types (measurement names)
    types = [
        'body-temperature', 'systolic-blood-pressure', 'diastolic-blood-pressure',
        'heart-rate', 'body-weight', 'muscle-mass', 'total-body-water', 'skin-temperature'
    ]

    # Group and compute mean value of each measurement in each quarter
    df_grouped = df.groupby(['id', 'date', 'type'])['value'].mean().reset_index()

    # Pivot so each device_type becomes a column
    pivot_df = df_grouped.pivot_table(
        index=['id', 'date'],
        columns='type',
        values='value'
    ).reset_index()

    # Rename columns (flatten multi-index if needed)
    pivot_df.columns.name = None

    # Ensure all required feature columns are present
    for feature in types:
        if feature not in pivot_df.columns:
            pivot_df[feature] = pd.NA  # Fill missing columns

    # Fill missing values with -1
    # pivot_df[types] = pivot_df[types].fillna(-1)

    # Round all floats to 4 decimal points
    pivot_df[types] = pivot_df[types].round(4)

    return pivot_df


def extract_24h_labels(df):

    # Ensure 'date' is datetime
    df['date'] = pd.to_datetime(df['date-time']).dt.date

    # Desired types
    expected_types = ['blood-pressure', 'agitation', 'body-water', 'pulse', 'weight', 'body-temperature-label']

    # Count occurrences of each type per id, day, quarter
    counts = df.groupby(['id', 'date', 'type']).size().reset_index(name='count')

    # Pivot to make type columns
    pivot_df = counts.pivot_table(
        index=['id', 'date'],
        columns='type',
        values='count',
        fill_value=0
    ).reset_index()

    pivot_df.columns.name = None  # Flatten column names

    # Ensure all expected type columns exist
    for col in expected_types:
        if col not in pivot_df.columns:
            pivot_df[col] = 0

    # Order the columns: id, day, quarter, then expected types
    ordered_cols = ['id', 'date'] + expected_types
    pivot_df = pivot_df[ordered_cols]

    return pivot_df


def add_agitation_next_24h(df):

    df['date'] = pd.to_datetime(df['date'])

    df_shifted = df.shift(-1)

    agitation_next_list = []
    agitation_four_list = []

    for i in range(len(df)):
        row = df.loc[i]
        next_row = df_shifted.loc[i]

        agitation_next = -10
        agitation_four = -10

        if not pd.isna(next_row['id']):
            same_id = row['id'] == next_row['id']
            next_day = row['date'] + pd.Timedelta(days=1) == next_row['date']

            if same_id and next_day:
                agitation_next = next_row['agitation']
                curr_ag = row['agitation']
                next_ag = agitation_next

                if curr_ag in [0, -1] and next_ag in [0, -1]:
                    agitation_four = 0
                elif curr_ag in [0, -1] and next_ag == 1:
                    agitation_four = 1
                elif curr_ag == 1 and next_ag in [0, -1]:
                    agitation_four = 2
                elif curr_ag == 1 and next_ag == 1:
                    agitation_four = 3

        agitation_next_list.append(agitation_next)
        agitation_four_list.append(agitation_four)

    df['agitation-next'] = agitation_next_list
    df['agitation-four'] = agitation_four_list

    return df


def extract_12h_count(df):
    # Ensure datetime is in proper format
    df['date-time'] = pd.to_datetime(df['date-time'])
    df['date'] = df['date-time'].dt.date
    df['hour'] = df['date-time'].dt.hour

    # Define 12h bins
    df['12h'] = pd.cut(
        df['hour'],
        bins=[0, 12, 24],
        labels=['00-12', '12-24'],
        right=False
    )

    # Group by id, date, 12h period, and location
    counts = df.groupby(['id', 'date', '12h', 'location']).size().reset_index(name='count')

    # Pivot to get location columns
    pivot_df = counts.pivot_table(
        index=['id', 'date', '12h'],
        columns='location',
        values='count',
        fill_value=0
    ).reset_index()

    # Remove multi-level column name
    pivot_df.columns.name = None

    # Expected location columns
    expected_locations = ['back-door', 'bathroom', 'bedroom', 'fridge-door',
                          'front-door', 'hallway', 'kitchen', 'lounge']

    # Ensure all expected location columns are present
    for loc in expected_locations:
        if loc not in pivot_df.columns:
            pivot_df[loc] = 0

    # Sort columns for consistency
    pivot_df = pivot_df[['id', 'date', '12h'] + sorted(expected_locations)]

    return pivot_df


def extract_12h_contextual(df):
    expected_locations = ['back-door', 'bathroom', 'bedroom', 'fridge-door',
                          'front-door', 'hallway', 'kitchen', 'lounge']
    private = ['bedroom', 'bathroom']
    public = ['kitchen', 'hallway', 'lounge']

    # Preprocessing
    df['date-time'] = pd.to_datetime(df['date-time'])
    df['date'] = df['date-time'].dt.date
    df['hour'] = df['date-time'].dt.hour

    df['12h'] = pd.cut(
        df['hour'],
        bins=[0, 12, 24],
        labels=['00-12', '12-24'],
        right=False
        )

    # Count events per location per 12h
    counts = df.groupby(['id', 'date', '12h', 'location']).size().reset_index(name='count')
    pivot_df = counts.pivot_table(index=['id', 'date', '12h'], columns='location', values='count', fill_value=0).reset_index()
    pivot_df.columns.name = None

    # Ensure all expected locations are present
    for loc in expected_locations:
        if loc not in pivot_df.columns:
            pivot_df[loc] = 0

    # Contextual features
    pivot_df['total-events'] = pivot_df[expected_locations].sum(axis=1)
    pivot_df['unique-locations'] = pivot_df[expected_locations].gt(0).sum(axis=1)
    pivot_df['active-location-ratio'] = pivot_df['unique-locations'] / len(expected_locations)

    private_sum = pivot_df[private].sum(axis=1)
    public_sum = pivot_df[public].sum(axis=1)
    pivot_df['private-to-public-ratio'] = np.where(
        public_sum == 0,
        1.0,
        (private_sum / public_sum).round(4)
    )

    # Entropy
    def entropy(row):
        probs = row[row > 0] / row.sum()
        return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0

    pivot_df['location-entropy'] = pivot_df[expected_locations].apply(entropy, axis=1)
    pivot_df['location-dominance-ratio'] = pivot_df[expected_locations].max(axis=1) / (pivot_df['total-events'] + 1e-5)

    # Transitions: need to compute from raw df
    df_sorted = df.sort_values(by=['id', 'date', '12h', 'date-time'])
    df_sorted['prev-location'] = df_sorted.groupby(['id', 'date', '12h'])['location'].shift(1)
    df_sorted['prev2-location'] = df_sorted.groupby(['id', 'date', '12h'])['location'].shift(2)
    df_sorted['back-and-forth'] = ((df_sorted['prev2-location'] == df_sorted['location']) &
                                   (df_sorted['location'] != df_sorted['prev-location'])).astype(int)

    # Aggregate transitions
    backforth_counts = df_sorted.groupby(['id', 'date', '12h'])['back-and-forth'].sum().reset_index()
    transition_counts = df_sorted.groupby(['id', 'date', '12h']).apply(
        lambda x: (x['location'] != x['prev-location']).sum()
    ).reset_index(name='num-transitions')

    # Merge transition features
    pivot_df = pivot_df.merge(backforth_counts, on=['id', 'date', '12h'], how='left')
    pivot_df = pivot_df.merge(transition_counts, on=['id', 'date', '12h'], how='left')
    pivot_df.rename(columns={'back-and-forth': 'back-and-forth-count'}, inplace=True)

    # Fill NaNs
    pivot_df['back-and-forth-count'] = pivot_df['back-and-forth-count'].fillna(0).astype(int)
    pivot_df['num-transitions'] = pivot_df['num-transitions'].fillna(0).astype(int)

    # Round floats
    for col in pivot_df.select_dtypes(include='float').columns:
        pivot_df[col] = pivot_df[col].round(4)

    # Convert int-like floats to Int64
    for col in pivot_df.select_dtypes(include='number').columns:
        if pivot_df[col].dropna().apply(lambda x: float(x).is_integer()).all():
            pivot_df[col] = pivot_df[col].astype('Int64')

    # Drop raw location columns
    pivot_df = pivot_df.drop(columns=expected_locations)

    return pivot_df


def extract_12h_statistical(df):
    expected_locations = ['back-door', 'bathroom', 'bedroom', 'fridge-door',
                          'front-door', 'hallway', 'kitchen', 'lounge']
    expected_stats = ['sum', 'mean', 'max', 'std']

    # Ensure datetime format
    df['date-time'] = pd.to_datetime(df['date-time'])
    df['date'] = df['date-time'].dt.date
    df['hour'] = df['date-time'].dt.hour

    # Create 12-hour bin
    df['12h'] = pd.cut(
        df['hour'],
        bins=[0, 12, 24],
        labels=['00-12', '12-24'],
        right=False)

    # Count events per hour
    hourly_counts = df.groupby(['id', 'date', '12h', 'hour', 'location']).size().reset_index(name='count')

    # Aggregate stats per location per id-date-12h
    stats_df = hourly_counts.groupby(['id', 'date', '12h', 'location'])['count'].agg(expected_stats).reset_index()

    # Melt to long format
    stats_long = stats_df.melt(
        id_vars=['id', 'date', '12h', 'location'],
        value_vars=expected_stats,
        var_name='stat',
        value_name='value'
    )

    # Create feature names
    stats_long['feature'] = stats_long['location'] + '-count-' + stats_long['stat']

    # Pivot to wide format
    stats_pivot = stats_long.pivot_table(
        index=['id', 'date', '12h'],
        columns='feature',
        values='value'
    ).reset_index()

    stats_pivot.columns.name = None

    # Ensure all expected features are present
    all_expected_features = [loc + '-count-' + stat for loc in expected_locations for stat in expected_stats]
    for feature in all_expected_features:
        if feature not in stats_pivot.columns:
            stats_pivot[feature] = pd.NA  # use NaN to represent missing stats

    # Sort columns
    feature_cols = sorted([col for col in stats_pivot.columns if col not in ['id', 'date', '12h']])
    stats_pivot = stats_pivot[['id', 'date', '12h'] + feature_cols]

    # Round float columns
    for col in stats_pivot.select_dtypes(include='float').columns:
        stats_pivot[col] = stats_pivot[col].round(4)

    # Convert to Int64 if values are all integer-like
    for col in stats_pivot.select_dtypes(include='number').columns:
        col_data = stats_pivot[col].dropna()
        if col_data.apply(lambda x: float(x).is_integer()).all():
            stats_pivot[col] = stats_pivot[col].astype('Int64')  # nullable integer

    return stats_pivot


def extract_12h_physiology(df):
    # Convert to datetime if needed
    df['date-time'] = pd.to_datetime(df['date-time'])
    df['date'] = df['date-time'].dt.date
    df['hour'] = df['date-time'].dt.hour

    # Create 12h bin
    df['12h'] = pd.cut(df['hour'], bins=[0, 12, 24], labels=['00-12', '12-24'], right=False)

    # List of expected types
    types = [
        'body-temperature', 'systolic-blood-pressure', 'diastolic-blood-pressure',
        'heart-rate', 'body-weight', 'muscle-mass', 'total-body-water', 'skin-temperature'
    ]

    # Group and compute mean per 12h period
    df_grouped = df.groupby(['id', 'date', '12h', 'type'])['value'].mean().reset_index()

    # Pivot so each type becomes a column
    pivot_df = df_grouped.pivot_table(
        index=['id', 'date', '12h'],
        columns='type',
        values='value'
    ).reset_index()

    # Flatten column names
    pivot_df.columns.name = None

    # Ensure all expected types are present
    for feature in types:
        if feature not in pivot_df.columns:
            pivot_df[feature] = pd.NA  # Use NA (can be -1 later if needed)

    # Round all floats to 4 decimal points
    pivot_df[types] = pivot_df[types].round(4)

    return pivot_df


def extract_12h_labels(df):
    # Convert to datetime and extract date and hour
    df['date-time'] = pd.to_datetime(df['date-time'])
    df['date'] = df['date-time'].dt.date
    df['hour'] = df['date-time'].dt.hour

    # Define 12-hour bins
    df['12h'] = pd.cut(df['hour'], bins=[0, 12, 24], labels=['00-12', '12-24'], right=False)

    # Desired label types
    expected_types = ['blood-pressure', 'agitation', 'body-water', 'pulse', 'weight', 'body-temperature-label']

    # Count occurrences per id, date, 12h period, and type
    counts = df.groupby(['id', 'date', '12h', 'type']).size().reset_index(name='count')

    # Pivot to make type columns
    pivot_df = counts.pivot_table(
        index=['id', 'date', '12h'],
        columns='type',
        values='count',
        fill_value=0
    ).reset_index()

    # Flatten column names
    pivot_df.columns.name = None

    # Ensure all expected type columns exist
    for col in expected_types:
        if col not in pivot_df.columns:
            pivot_df[col] = 0

    # Order the columns: id, date, 12h, then expected types
    ordered_cols = ['id', 'date', '12h'] + expected_types
    pivot_df = pivot_df[ordered_cols]

    return pivot_df


def add_agitation_next_12h(df):

    df['date'] = pd.to_datetime(df['date'])

    # Define half-day periods
    half_day_order = ['00-12', '12-24']
    half_day_index = {h: i for i, h in enumerate(half_day_order)}

    df_shifted = df.shift(-1)

    agitation_next_list = []
    agitation_four_list = []

    for i in range(len(df)):
        row = df.loc[i]
        next_row = df_shifted.loc[i]

        # Default values
        agitation_next = -10
        agitation_four = -10

        if not pd.isna(next_row['id']):
            same_id = row['id'] == next_row['id']
            curr_day = row['date']
            next_day = next_row['date']

            try:
                curr_h = half_day_index[row['12h']]
                next_h = half_day_index[next_row['12h']]
            except:
                curr_h, next_h = -1, -2  # force fail

            # Define valid half-day transition
            is_next_half_same_day = curr_day == next_day and next_h == curr_h + 1
            is_next_day_rollover = curr_day + pd.Timedelta(days=1) == next_day and curr_h == 1 and next_h == 0

            if same_id and (is_next_half_same_day or is_next_day_rollover):
                agitation_next = next_row['agitation']
                curr_ag = row['agitation']
                next_ag = agitation_next

                if curr_ag in [0, -1] and next_ag in [0, -1]:
                    agitation_four = 0
                elif curr_ag in [0, -1] and next_ag == 1:
                    agitation_four = 1
                elif curr_ag == 1 and next_ag in [0, -1]:
                    agitation_four = 2
                elif curr_ag == 1 and next_ag == 1:
                    agitation_four = 3

        agitation_next_list.append(agitation_next)
        agitation_four_list.append(agitation_four)

    df['agitation-next'] = agitation_next_list
    df['agitation-four'] = agitation_four_list

    return df


def extract_6h_count(df):
    # Ensure datetime is in proper format
    df['date-time'] = pd.to_datetime(df['date-time'])
    df['date'] = df['date-time'].dt.date
    df['hour'] = df['date-time'].dt.hour

    # Define 6-hour bins
    df['6h'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['00-06', '06-12', '12-18', '18-24'],
        right=False
    )

    # Group by id, date, 6h period, and location
    counts = df.groupby(['id', 'date', '6h', 'location']).size().reset_index(name='count')

    # Pivot to get location columns
    pivot_df = counts.pivot_table(
        index=['id', 'date', '6h'],
        columns='location',
        values='count',
        fill_value=0
    ).reset_index()

    # Remove multi-level column name
    pivot_df.columns.name = None

    # Expected location columns
    expected_locations = ['back-door', 'bathroom', 'bedroom', 'fridge-door',
                          'front-door', 'hallway', 'kitchen', 'lounge']

    # Ensure all expected location columns are present
    for loc in expected_locations:
        if loc not in pivot_df.columns:
            pivot_df[loc] = 0

    # Sort columns for consistency
    pivot_df = pivot_df[['id', 'date', '6h'] + sorted(expected_locations)]

    return pivot_df


def extract_6h_contextual(df):

    expected_locations = ['back-door', 'bathroom', 'bedroom', 'fridge-door',
                          'front-door', 'hallway', 'kitchen', 'lounge']
    
    private = ['bedroom', 'bathroom']
    public = ['kitchen', 'hallway', 'lounge']

    # Preprocessing
    df['date-time'] = pd.to_datetime(df['date-time'])
    df['date'] = df['date-time'].dt.date
    df['hour'] = df['date-time'].dt.hour

    # Define 6h bins
    df['6h'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['00-06', '06-12', '12-18', '18-24'],
        right=False
    )

    # Count events per location per 6h
    counts = df.groupby(['id', 'date', '6h', 'location']).size().reset_index(name='count')
    pivot_df = counts.pivot_table(index=['id', 'date', '6h'], columns='location', values='count', fill_value=0).reset_index()
    pivot_df.columns.name = None

    # Ensure all expected locations are present
    for loc in expected_locations:
        if loc not in pivot_df.columns:
            pivot_df[loc] = 0

    # Contextual features
    pivot_df['total-events'] = pivot_df[expected_locations].sum(axis=1)
    pivot_df['unique-locations'] = pivot_df[expected_locations].gt(0).sum(axis=1)
    pivot_df['active-location-ratio'] = pivot_df['unique-locations'] / len(expected_locations)

    private_sum = pivot_df[private].sum(axis=1)
    public_sum = pivot_df[public].sum(axis=1)
    pivot_df['private-to-public-ratio'] = np.where(
        public_sum == 0,
        1.0,
        (private_sum / public_sum).round(4)
    )

    # Entropy
    def entropy(row):
        probs = row[row > 0] / row.sum()
        return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0

    pivot_df['location-entropy'] = pivot_df[expected_locations].apply(entropy, axis=1)
    pivot_df['location-dominance-ratio'] = pivot_df[expected_locations].max(axis=1) / (pivot_df['total-events'] + 1e-5)

    # Transitions: compute from raw df
    df_sorted = df.sort_values(by=['id', 'date', '6h', 'date-time'])
    df_sorted['prev-location'] = df_sorted.groupby(['id', 'date', '6h'])['location'].shift(1)
    df_sorted['prev2-location'] = df_sorted.groupby(['id', 'date', '6h'])['location'].shift(2)
    df_sorted['back-and-forth'] = ((df_sorted['prev2-location'] == df_sorted['location']) &
                                   (df_sorted['location'] != df_sorted['prev-location'])).astype(int)

    # Aggregate transitions
    backforth_counts = df_sorted.groupby(['id', 'date', '6h'])['back-and-forth'].sum().reset_index()
    transition_counts = df_sorted.groupby(['id', 'date', '6h']).apply(
        lambda x: (x['location'] != x['prev-location']).sum()
    ).reset_index(name='num-transitions')

    # Merge transition features
    pivot_df = pivot_df.merge(backforth_counts, on=['id', 'date', '6h'], how='left')
    pivot_df = pivot_df.merge(transition_counts, on=['id', 'date', '6h'], how='left')
    pivot_df.rename(columns={'back-and-forth': 'back-and-forth-count'}, inplace=True)

    # Fill NaNs
    pivot_df['back-and-forth-count'] = pivot_df['back-and-forth-count'].fillna(0).astype(int)
    pivot_df['num-transitions'] = pivot_df['num-transitions'].fillna(0).astype(int)

    # Round floats
    for col in pivot_df.select_dtypes(include='float').columns:
        pivot_df[col] = pivot_df[col].round(4)

    # Convert int-like floats to Int64
    for col in pivot_df.select_dtypes(include='number').columns:
        if pivot_df[col].dropna().apply(lambda x: float(x).is_integer()).all():
            pivot_df[col] = pivot_df[col].astype('Int64')

    # Drop raw location columns
    pivot_df = pivot_df.drop(columns=expected_locations)

    return pivot_df


def extract_6h_statistical(df):
    expected_locations = ['back-door', 'bathroom', 'bedroom', 'fridge-door',
                          'front-door', 'hallway', 'kitchen', 'lounge']
    expected_stats = ['sum', 'mean', 'max', 'std']

    # Ensure datetime format
    df['date-time'] = pd.to_datetime(df['date-time'])
    df['date'] = df['date-time'].dt.date
    df['hour'] = df['date-time'].dt.hour

    # Create 6-hour bin
    df['6h'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['00-06', '06-12', '12-18', '18-24'],
        right=False
    )

    # Count events per hour
    hourly_counts = df.groupby(['id', 'date', '6h', 'hour', 'location']).size().reset_index(name='count')

    # Aggregate stats per location per id-date-6h
    stats_df = hourly_counts.groupby(['id', 'date', '6h', 'location'])['count'].agg(expected_stats).reset_index()

    # Melt to long format
    stats_long = stats_df.melt(
        id_vars=['id', 'date', '6h', 'location'],
        value_vars=expected_stats,
        var_name='stat',
        value_name='value'
    )

    # Create feature names like kitchen-count-mean
    stats_long['feature'] = stats_long['location'] + '-count-' + stats_long['stat']

    # Pivot to wide format
    stats_pivot = stats_long.pivot_table(
        index=['id', 'date', '6h'],
        columns='feature',
        values='value'
    ).reset_index()

    stats_pivot.columns.name = None

    # Ensure all expected features are present
    all_expected_features = [loc + '-count-' + stat for loc in expected_locations for stat in expected_stats]
    for feature in all_expected_features:
        if feature not in stats_pivot.columns:
            stats_pivot[feature] = pd.NA

    # Sort columns for consistency
    feature_cols = sorted([col for col in stats_pivot.columns if col not in ['id', 'date', '6h']])
    stats_pivot = stats_pivot[['id', 'date', '6h'] + feature_cols]

    # Round float columns
    for col in stats_pivot.select_dtypes(include='float').columns:
        stats_pivot[col] = stats_pivot[col].round(4)

    # Convert int-like columns to Int64
    for col in stats_pivot.select_dtypes(include='number').columns:
        if stats_pivot[col].dropna().apply(lambda x: float(x).is_integer()).all():
            stats_pivot[col] = stats_pivot[col].astype('Int64')

    return stats_pivot


def extract_6h_physiology(df):

    # Convert to datetime if needed
    df['date-time'] = pd.to_datetime(df['date-time'])
    df['date'] = df['date-time'].dt.date
    df['hour'] = df['date-time'].dt.hour

    # Create 6-hour bin
    df['6h'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['00-06', '06-12', '12-18', '18-24'],
        right=False
    )

    # List of expected types
    types = [
        'body-temperature', 'systolic-blood-pressure', 'diastolic-blood-pressure',
        'heart-rate', 'body-weight', 'muscle-mass', 'total-body-water', 'skin-temperature'
    ]

    # Group and compute mean per 6h period
    df_grouped = df.groupby(['id', 'date', '6h', 'type'])['value'].mean().reset_index()

    # Pivot so each type becomes a column
    pivot_df = df_grouped.pivot_table(
        index=['id', 'date', '6h'],
        columns='type',
        values='value'
    ).reset_index()

    # Flatten column names
    pivot_df.columns.name = None

    # Ensure all expected types are present
    for feature in types:
        if feature not in pivot_df.columns:
            pivot_df[feature] = pd.NA

    # Round all floats to 4 decimal points
    pivot_df[types] = pivot_df[types].round(4)

    return pivot_df


def extract_6h_labels(df):
    # Convert to datetime and extract date and hour
    df['date-time'] = pd.to_datetime(df['date-time'])
    df['date'] = df['date-time'].dt.date
    df['hour'] = df['date-time'].dt.hour

    # Define 6-hour bins
    df['6h'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['00-06', '06-12', '12-18', '18-24'],
        right=False
    )

    # Desired label types
    expected_types = ['blood-pressure', 'agitation', 'body-water', 'pulse', 'weight', 'body-temperature-label']

    # Count occurrences per id, date, 6h period, and type
    counts = df.groupby(['id', 'date', '6h', 'type']).size().reset_index(name='count')

    # Pivot to make type columns
    pivot_df = counts.pivot_table(
        index=['id', 'date', '6h'],
        columns='type',
        values='count',
        fill_value=0
    ).reset_index()

    # Flatten column names
    pivot_df.columns.name = None

    # Ensure all expected type columns exist
    for col in expected_types:
        if col not in pivot_df.columns:
            pivot_df[col] = 0

    # Order the columns: id, date, 6h, then expected types
    ordered_cols = ['id', 'date', '6h'] + expected_types
    pivot_df = pivot_df[ordered_cols]

    return pivot_df


def add_agitation_next_6h(df):

    df['date'] = pd.to_datetime(df['date'])

    # Define ordered quarters
    quarter_order = ['00-06', '06-12', '12-18', '18-24']
    df_shifted = df.shift(-1)

    agitation_next_list = []
    agitation_four_list = []

    for i in range(len(df)):
        row = df.loc[i]
        next_row = df_shifted.loc[i]

        # Defaults
        agitation_next = -10
        agitation_four = -10

        if not pd.isna(next_row['id']):
            same_id = row['id'] == next_row['id']
            same_day = row['date'] == next_row['date']
            next_day = row['date'] + pd.Timedelta(days=1) == next_row['date']

            try:
                curr_q = quarter_order.index(row['6h'])
                next_q = quarter_order.index(next_row['6h'])
            except:
                curr_q, next_q = -1, -2  # force non-matching

            is_consecutive_quarter = (
                (same_day and next_q == curr_q + 1) or
                (curr_q == 3 and next_day and next_q == 0)
            )

            # Apply only if same ID and next quarter
            if same_id and is_consecutive_quarter:
                agitation_next = next_row['agitation']
                curr_ag = row['agitation']
                next_ag = agitation_next

                if curr_ag in [0, -1] and next_ag in [0, -1]:
                    agitation_four = 0
                elif curr_ag in [0, -1] and next_ag == 1:
                    agitation_four = 1
                elif curr_ag == 1 and next_ag in [0, -1]:
                    agitation_four = 2
                elif curr_ag == 1 and next_ag == 1:
                    agitation_four = 3

        agitation_next_list.append(agitation_next)
        agitation_four_list.append(agitation_four)

    df['agitation-next'] = agitation_next_list
    df['agitation-four'] = agitation_four_list

    return df


def hierarchical_imputation(df):
    
    df = df.copy()
    value_columns = [col for col in df.columns if col not in ['id', 'date']]
    for index, row in df.iterrows():
        for col in value_columns:
            if pd.isna(row[col]):
                # Step 1: mean of same date (excluding current row)
                same_date_rows = df[(df['date'] == row['date']) & (df.index != index)][col]
                same_date_mean = same_date_rows[same_date_rows.notna()].mean()

                if not np.isnan(same_date_mean):
                    df.at[index, col] = round(same_date_mean, 4)
                    continue

                # Step 2: mean of same id (excluding current row)
                same_id_rows = df[(df['id'] == row['id']) & (df.index != index)][col]
                same_id_mean = same_id_rows[same_id_rows.notna()].mean()

                if not np.isnan(same_id_mean):
                    df.at[index, col] = round(same_id_mean, 4)
                    continue

                # Step 3: global mean
                global_mean = df[col][df[col].notna()].mean()
                df.at[index, col] = round(global_mean, 4)

    return df


def hierarchical_imputation_columns_to_exclude(df, columns_to_exclude=None):
    df = df.copy()

    if columns_to_exclude is None:
        columns_to_exclude = []

    # Define columns to impute (exclude id, date, and user-specified ones)
    value_columns = [col for col in df.columns if col not in ['id', 'date'] + columns_to_exclude]

    for index, row in df.iterrows():
        for col in value_columns:
            if pd.isna(row[col]):
                # Step 1: mean of same date (excluding current row)
                same_date_rows = df[(df['date'] == row['date']) & (df.index != index)][col]
                same_date_mean = same_date_rows[same_date_rows.notna()].mean()

                if not np.isnan(same_date_mean):
                    df.at[index, col] = round(same_date_mean, 4)
                    continue

                # Step 2: mean of same id (excluding current row)
                same_id_rows = df[(df['id'] == row['id']) & (df.index != index)][col]
                same_id_mean = same_id_rows[same_id_rows.notna()].mean()

                if not np.isnan(same_id_mean):
                    df.at[index, col] = round(same_id_mean, 4)
                    continue

                # Step 3: global mean
                global_mean = df[col][df[col].notna()].mean()
                df.at[index, col] = round(global_mean, 4)

    return df


def segment_dataframe(df, n, columns_indices):
    
    segments = []
    labels = []
    participants = []

    df = df.sort_values(by=['id', 'date', '6h']).reset_index(drop=True)

    for i in range(n - 1, len(df) - 1):  # stop at len(df) - 1 to have a next row
        segment = df.iloc[i - n + 1: i + 1]
        next_row = df.iloc[i + 1]

        # Check if all rows in segment (and next_row) have the same id
        if segment['id'].nunique() != 1 or segment['id'].iloc[0] != next_row['id']:
            continue

        # Get the label from the next row's agitation value
        labels.append(int(next_row['agitation']))
        
        
        participants.append(next_row['id'])

        # Drop non-feature columns
        segment = segment.drop(columns=columns_indices + ['agitation', 'agitation-next', 'agitation-four'], errors='ignore')
        segments.append(segment.reset_index(drop=True).to_numpy())

    return segments, labels, participants


def combine_single_row_csvs(input_folder, output_file):

    combined_data = []

    # List all CSV files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)

            # Ensure the file has only one row
            if df.shape[0] != 1:
                raise ValueError(f"{filename} has more than one row.")

            # Add filename as a new column
            row = df.iloc[0].copy()
            row['model'] = filename.split('.')[0]
            combined_data.append(row)

    # Combine all rows into a single DataFrame
    combined_df = pd.DataFrame(combined_data)

    # Move 'filename' to the first column
    cols = ['model'] + [col for col in combined_df.columns if col != 'model']
    combined_df = combined_df[cols]

    combined_df.sort_values(by='model', ascending=True, inplace=True)

    # Save to output CSV
    combined_df.to_csv(output_file, index=False)


