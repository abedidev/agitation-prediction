import pandas as pd


def count_location_per_day(df):
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

    return pivot_df  # if you want to inspect it in code


def count_location_per_half_day(df):
    # Ensure 'date' is datetime
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Create half-day bins
    df['12h'] = pd.cut(
        df['date-time'].dt.hour,
        bins=[0, 12, 24],
        labels=['00-12', '12-24'],
        right=False
    )

    # Group by id, day, half_day, and location_name, then count
    counts = df.groupby(['id', 'date', '12h', 'location']).size().reset_index(name='count')

    # Pivot to get location_name as columns
    pivot_df = counts.pivot_table(
        index=['id', 'date', '12h'],
        columns='location',
        values='count',
        fill_value=0
    ).reset_index()

    # Remove hierarchical column name
    pivot_df.columns.name = None

    return pivot_df



def count_location_per_quarter(df):
    # Load the data

    # Ensure 'date' is datetime (in case it's not)
    df['date'] = pd.to_datetime(df['date']).dt.date

    df['6h'] = pd.cut(
        df['date-time'].dt.hour,
        bins=[0, 6, 12, 18, 24],
        labels=['00-06', '06-12', '12-18', '18-24'],
        right=False
    )

    # Group by id, day, and location_name, then count
    counts = df.groupby(['id', 'date', '6h', 'location']).size().reset_index(name='count')

    # Pivot to get location_name as columns
    pivot_df = counts.pivot_table(
        index=['id', 'date', '6h'],
        columns='location',
        values='count',
        fill_value=0
    ).reset_index()

    # Optional: flatten column names if needed (e.g., after pivot)
    pivot_df.columns.name = None  # remove hierarchical column name

    return pivot_df  # if you want to inspect it in code


def count_location_per_hour(df):
    # Load the data

    # Ensure 'date' is datetime (in case it's not)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['hour'] = pd.to_datetime(df['date-time']).dt.hour


    # Group by id, day, and location_name, then count
    counts = df.groupby(['id', 'date', 'hour', 'location']).size().reset_index(name='count')

    # Pivot to get location_name as columns
    pivot_df = counts.pivot_table(
        index=['id', 'date', 'hour'],
        columns='location',
        values='count',
        fill_value=0
    ).reset_index()

    # Optional: flatten column names if needed (e.g., after pivot)
    pivot_df.columns.name = None  # remove hierarchical column name

    return pivot_df  # if you want to inspect it in code


import pandas as pd
import numpy as np


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def extract_activity_features(df):
    # Ensure datetime types
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['6h'] = pd.cut(
        df['date-time'].dt.hour,
        bins=[0, 6, 12, 18, 24],
        labels=['00-06', '06-12', '12-18', '18-24'],
        right=False
    )

    # Standardize location names
    df['location'] = df['location'].replace({
        'Back Door': 'back-door',
        'Bathroom': 'bathroom',
        'Bedroom': 'bedroom',
        'Fridge Door': 'fridge-door',
        'Front Door': 'front-door',
        'Hallway': 'hallway',
        'Kitchen': 'kitchen',
        'Lounge': 'lounge'
    })

    # Count events per location per quarter
    counts = df.groupby(['id', 'date', '6h', 'location']).size().reset_index(name='count')
    pivot_df = counts.pivot_table(
        index=['id', 'date', '6h'],
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
        (private_sum / public_sum).round(2)
    )

    # Entropy function
    def entropy(row):
        probs = row[row > 0] / row.sum()
        return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0

    pivot_df['location-entropy'] = pivot_df[expected_locations].apply(entropy, axis=1)
    # pivot_df['dominant-location'] = pivot_df[expected_locations].idxmax(axis=1)
    pivot_df['location-dominance-ratio'] = pivot_df[expected_locations].max(axis=1) / (pivot_df['total-events'] + 1e-5)

    # Sort original df for transition features
    df_sorted = df.sort_values(by=['id', 'date', '6h', 'date-time'])
    df_sorted['prev-location'] = df_sorted.groupby(['id', 'date', '6h'])['location'].shift(1)
    df_sorted['prev2-location'] = df_sorted.groupby(['id', 'date', '6h'])['location'].shift(2)
    df_sorted['back-and-forth'] = ((df_sorted['prev2-location'] == df_sorted['location']) &
                                   (df_sorted['location'] != df_sorted['prev-location'])).astype(int)

    # Aggregate transition features
    backforth_counts = df_sorted.groupby(['id', 'date', '6h'])['back-and-forth'].sum().reset_index()
    transition_counts = df_sorted.groupby(['id', 'date', '6h']).apply(
        lambda x: (x['location'] != x['prev-location']).sum()
    ).reset_index(name='num-transitions')

    pivot_df = pivot_df.merge(backforth_counts, on=['id', 'date', '6h'], how='left')
    pivot_df = pivot_df.merge(transition_counts, on=['id', 'date', '6h'], how='left')
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
        pivot_df[col] = pivot_df[col].round(2)

    for col in pivot_df.select_dtypes(include='number').columns:
        if (pivot_df[col] == pivot_df[col].astype(int)).all():
            pivot_df[col] = pivot_df[col].astype(int)

    return pivot_df


def extract_hourly_stats_per_quarter_paper(df):

    # Preprocess date columns
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['hour'] = df['date-time'].dt.hour
    df['6h'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['00-06', '06-12', '12-18', '18-24'],
        right=False
    )

    # Standardize location names
    df['location'] = df['location'].replace({
        'Back Door': 'back-door',
        'Bathroom': 'bathroom',
        'Bedroom': 'bedroom',
        'Fridge Door': 'fridge-door',
        'Front Door': 'front-door',
        'Hallway': 'hallway',
        'Kitchen': 'kitchen',
        'Lounge': 'lounge'
    })

    # Step 1: count events per patient-day-hour-location
    hourly_counts = df.groupby(['id', 'date', '6h', 'hour', 'location']).size().reset_index(name='count')

    # Step 2: compute sum, mean, max, std per quarter
    stats_df = hourly_counts.groupby(['id', 'date', '6h', 'location'])['count'].agg(['sum', 'mean', 'max', 'std']).reset_index()

    # Step 3: pivot the statistics into feature columns
    stats_long = stats_df.melt(
        id_vars=['id', 'date', '6h', 'location'],
        value_vars=['sum', 'mean', 'max', 'std'],
        var_name='stat',
        value_name='value'
    )

    # Create column names like bathroom-count-sum
    stats_long['feature'] = stats_long['location'] + '-count-' + stats_long['stat']
    stats_pivot = stats_long.pivot_table(
        index=['id', 'date', '6h'],
        columns='feature',
        values='value'
    ).reset_index()

    stats_pivot.columns.name = None

    # Rename id to id
    stats_pivot = stats_pivot.rename(columns={'id': 'id'})

    # Format numbers: round floats, make integers clean
    for col in stats_pivot.select_dtypes(include='float').columns:
        stats_pivot[col] = stats_pivot[col].round(2)
    for col in stats_pivot.select_dtypes(include='number').columns:
        if (stats_pivot[col] == stats_pivot[col].astype(int)).all():
            stats_pivot[col] = stats_pivot[col].astype(int)

    return stats_pivot


def vitals_to_quarter(df):
    # Convert to datetime if needed
    df['date-time'] = pd.to_datetime(df['date-time'])
    df['date'] = df['date-time'].dt.date
    df['hour'] = df['date-time'].dt.hour

    # Define quarter
    df['6h'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['00-06', '06-12', '12-18', '18-24'],
        right=False
    )

    # List of expected device types (measurement names)
    features = [
        'body-temperature', 'systolic-blood-pressure', 'diastolic-blood-pressure',
        'heart-rate', 'body-weight', 'muscle-mass', 'total-body-water', 'skin-temperature'
    ]

    # Group and compute mean value of each measurement in each quarter
    df_grouped = df.groupby(['id', 'date', '6h', 'type'])['value'].mean().reset_index()

    # Pivot so each device_type becomes a column
    pivot_df = df_grouped.pivot_table(
        index=['id', 'date', '6h'],
        columns='type',
        values='value'
    ).reset_index()

    # Rename columns (flatten multi-index if needed)
    pivot_df.columns.name = None

    # Ensure all required feature columns are present
    for feature in features:
        if feature not in pivot_df.columns:
            pivot_df[feature] = -1  # Fill missing columns

    # Fill missing values with -1
    pivot_df[features] = pivot_df[features].fillna(-1)

    # Round all floats to 4 decimal points
    pivot_df[features] = pivot_df[features].round(2)

    return pivot_df


def count_type_per_quarter(df):


    # Ensure 'date' is datetime
    df['date'] = pd.to_datetime(df['date-time']).dt.date

    # Define quarter bins
    df['6h'] = pd.cut(
        df['date-time'].dt.hour,
        bins=[0, 6, 12, 18, 24],
        labels=['00-06', '06-12', '12-18', '18-24'],
        right=False
    )

    # Desired types
    expected_types = ['blood-pressure', 'agitation', 'body-water', 'pulse', 'weight', 'body-temperature-label']

    # Count occurrences of each type per id, day, quarter
    counts = df.groupby(['id', 'date', '6h', 'type']).size().reset_index(name='count')

    # Pivot to make type columns
    pivot_df = counts.pivot_table(
        index=['id', 'date', '6h'],
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
    ordered_cols = ['id', 'date', '6h'] + expected_types
    pivot_df = pivot_df[ordered_cols]

    return pivot_df


def merge_activity_physiology(activity_df, physiology_df):
    """
    Merges activity and physiology data on 'id', 'date', and '6h'.
    If a day in physiology_df is missing some quarters, it duplicates any available quarter's data
    to ensure all four quarters ('00-06', '06-12', '12-18', '18-24') are present.

    Parameters:
        activity_df (pd.DataFrame): Activity data with ['id', 'date', '6h', ...activity columns...]
        physiology_df (pd.DataFrame): Physiology data with ['id', 'date', '6h', ...physiology columns...]

    Returns:
        pd.DataFrame: Merged DataFrame with all quarters and combined activity + physiology features.
    """
    # Standardize column names
    # activity_df.columns = ['id', 'date', '6h'] + [f"act_{col}" for col in activity_df.columns[3:]]
    # physiology_df.columns = ['id', 'date', '6h'] + [f"phys_{col}" for col in physiology_df.columns[3:]]

    # Define expected quarters
    all_quarters = ['00-06', '06-12', '12-18', '18-24']

    # Fill missing quarters in physiology_df
    expanded_rows = []
    for (pid, day), group in physiology_df.groupby(['id', 'date']):
        existing_quarters = set(group['6h'])
        for quarter in all_quarters:
            if quarter in existing_quarters:
                expanded_rows.append(group[group['6h'] == quarter])
            else:
                # Use any available row from the day to fill the missing quarter
                row_to_copy = group.iloc[0:1].copy()
                row_to_copy['6h'] = quarter
                expanded_rows.append(row_to_copy)

    physiology_expanded = pd.concat(expanded_rows, ignore_index=True)

    # Merge with activity data
    merged_df = pd.merge(activity_df, physiology_expanded, on=['id', 'date', '6h'], how='left')

    return merged_df


def merge_activity_physiology_half(activity_df, physiology_df):
    """
    Merges activity and physiology data on 'id', 'date', and '6h'.
    If a day in physiology_df is missing some quarters, it duplicates any available quarter's data
    to ensure all four quarters ('00-06', '06-12', '12-18', '18-24') are present.

    Parameters:
        activity_df (pd.DataFrame): Activity data with ['id', 'date', '6h', ...activity columns...]
        physiology_df (pd.DataFrame): Physiology data with ['id', 'date', '6h', ...physiology columns...]

    Returns:
        pd.DataFrame: Merged DataFrame with all quarters and combined activity + physiology features.
    """
    # Standardize column names
    # activity_df.columns = ['id', 'date', '6h'] + [f"act_{col}" for col in activity_df.columns[3:]]
    # physiology_df.columns = ['id', 'date', '6h'] + [f"phys_{col}" for col in physiology_df.columns[3:]]

    # Define expected quarters
    all_quarters = ['00-12', '12-24']

    # Fill missing quarters in physiology_df
    expanded_rows = []
    for (pid, day), group in physiology_df.groupby(['id', 'date']):
        existing_quarters = set(group['12h'])
        for quarter in all_quarters:
            if quarter in existing_quarters:
                expanded_rows.append(group[group['12h'] == quarter])
            else:
                # Use any available row from the day to fill the missing quarter
                row_to_copy = group.iloc[0:1].copy()
                row_to_copy['12h'] = quarter
                expanded_rows.append(row_to_copy)

    physiology_expanded = pd.concat(expanded_rows, ignore_index=True)

    # Merge with activity data
    merged_df = pd.merge(activity_df, physiology_expanded, on=['id', 'date', '12h'], how='left')

    return merged_df


def merge_merged_label(merged_df, labels_df):

    all_quarters = ['00-06', '06-12', '12-18', '18-24']

    # Fill missing quarters in physiology_df
    expanded_rows = []
    for (pid, day), group in labels_df.groupby(['id', 'date']):
        existing_quarters = set(group['6h'])
        for quarter in all_quarters:
            if quarter in existing_quarters:
                expanded_rows.append(group[group['6h'] == quarter])
            else:
                # Use any available row from the day to fill the missing quarter
                row_to_copy = group.iloc[0:1].copy()
                row_to_copy['6h'] = quarter
                expanded_rows.append(row_to_copy)

    physiology_expanded = pd.concat(expanded_rows, ignore_index=True)

    # Merge with activity data
    merged_df = pd.merge(merged_df, physiology_expanded, on=['id', 'date', '6h'], how='left')

    return merged_df



def merge_merged_label_half(merged_df, labels_df):

    all_quarters = ['00-12', '12-24']

    # Fill missing quarters in physiology_df
    expanded_rows = []
    for (pid, day), group in labels_df.groupby(['id', 'date']):
        existing_quarters = set(group['12h'])
        for quarter in all_quarters:
            if quarter in existing_quarters:
                expanded_rows.append(group[group['12h'] == quarter])
            else:
                # Use any available row from the day to fill the missing quarter
                row_to_copy = group.iloc[0:1].copy()
                row_to_copy['12h'] = quarter
                expanded_rows.append(row_to_copy)

    physiology_expanded = pd.concat(expanded_rows, ignore_index=True)

    # Merge with activity data
    merged_df = pd.merge(merged_df, physiology_expanded, on=['id', 'date', '12h'], how='left')

    return merged_df


def add_agitation_next_column(input_df):

    df = input_df.copy()
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


def vitals_to_day(df):
    # Convert to datetime if needed
    df['date-time'] = pd.to_datetime(df['date-time'])
    df['date'] = df['date-time'].dt.date
    # df['hour'] = df['date-time'].dt.hour

    # Define quarter
    # df['6h'] = pd.cut(
    #     df['hour'],
    #     bins=[0, 6, 12, 18, 24],
    #     labels=['00-06', '06-12', '12-18', '18-24'],
    #     right=False
    # )

    # List of expected device types (measurement names)
    features = [
        'body-temperature', 'systolic-blood-pressure', 'diastolic-blood-pressure',
        'heart-rate', 'body-weight', 'muscle-mass', 'total-body-water', 'skin-temperature'
    ]

    # Group and compute mean value of each measurement in each quarter
    # df_grouped = df.groupby(['id', 'date', '6h', 'type'])['value'].mean().reset_index()
    df_grouped = df.groupby(['id', 'date', 'type'])['value'].mean().reset_index()

    # Pivot so each device_type becomes a column
    pivot_df = df_grouped.pivot_table(
        # index=['id', 'date', '6h'],
        index=['id', 'date'],
        columns='type',
        values='value'
    ).reset_index()

    # Rename columns (flatten multi-index if needed)
    pivot_df.columns.name = None

    # Ensure all required feature columns are present
    for feature in features:
        if feature not in pivot_df.columns:
            pivot_df[feature] = -1  # Fill missing columns

    # Fill missing values with -1
    pivot_df[features] = pivot_df[features].fillna(-1)

    # Round all floats to 4 decimal points
    pivot_df[features] = pivot_df[features].round(2)

    return pivot_df




def count_type_per_day(df):


    # Ensure 'date' is datetime
    df['date'] = pd.to_datetime(df['date-time']).dt.date

    # Define quarter bins
    # df['6h'] = pd.cut(
    #     df['date-time'].dt.hour,
    #     bins=[0, 6, 12, 18, 24],
    #     labels=['00-06', '06-12', '12-18', '18-24'],
    #     right=False
    # )

    # Desired types
    expected_types = ['blood-pressure', 'agitation', 'body-water', 'pulse', 'weight', 'body-temperature-label']

    # Count occurrences of each type per id, day, quarter
    # counts = df.groupby(['id', 'date', '6h', 'type']).size().reset_index(name='count')
    counts = df.groupby(['id', 'date', 'type']).size().reset_index(name='count')

    # Pivot to make type columns
    pivot_df = counts.pivot_table(
        # index=['id', 'date', '6h'],
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
    # ordered_cols = ['id', 'date', '6h'] + expected_types
    ordered_cols = ['id', 'date'] + expected_types
    pivot_df = pivot_df[ordered_cols]

    return pivot_df

def merge_activity_physiology_day(activity_df, physiology_df):
    """
    Merges activity and physiology data on 'id', 'date', and '6h'.
    If a day in physiology_df is missing some quarters, it duplicates any available quarter's data
    to ensure all four quarters ('00-06', '06-12', '12-18', '18-24') are present.

    Parameters:
        activity_df (pd.DataFrame): Activity data with ['id', 'date', '6h', ...activity columns...]
        physiology_df (pd.DataFrame): Physiology data with ['id', 'date', '6h', ...physiology columns...]

    Returns:
        pd.DataFrame: Merged DataFrame with all quarters and combined activity + physiology features.
    """
    # Standardize column names
    # activity_df.columns = ['id', 'date', '6h'] + [f"act_{col}" for col in activity_df.columns[3:]]
    # physiology_df.columns = ['id', 'date', '6h'] + [f"phys_{col}" for col in physiology_df.columns[3:]]

    # Define expected quarters
    # all_quarters = ['00-06', '06-12', '12-18', '18-24']
    #
    # # Fill missing quarters in physiology_df
    # expanded_rows = []
    # for (pid, day), group in physiology_df.groupby(['id', 'date']):
    #     existing_quarters = set(group['6h'])
    #     for quarter in all_quarters:
    #         if quarter in existing_quarters:
    #             expanded_rows.append(group[group['6h'] == quarter])
    #         else:
    #             # Use any available row from the day to fill the missing quarter
    #             row_to_copy = group.iloc[0:1].copy()
    #             row_to_copy['6h'] = quarter
    #             expanded_rows.append(row_to_copy)
    #
    # physiology_expanded = pd.concat(expanded_rows, ignore_index=True)

    # Merge with activity data
    merged_df = pd.merge(activity_df, physiology_df, on=['id', 'date'], how='outer')

    return merged_df



def merge_merged_label_day(merged_df, labels_df):

    # all_quarters = ['00-06', '06-12', '12-18', '18-24']

    # Fill missing quarters in physiology_df
    # expanded_rows = []
    # for (pid, day), group in labels_df.groupby(['id', 'date']):
    #     existing_quarters = set(group['6h'])
    #     for quarter in all_quarters:
    #         if quarter in existing_quarters:
    #             expanded_rows.append(group[group['6h'] == quarter])
    #         else:
    #             # Use any available row from the day to fill the missing quarter
    #             row_to_copy = group.iloc[0:1].copy()
    #             row_to_copy['6h'] = quarter
    #             expanded_rows.append(row_to_copy)

    # physiology_expanded = pd.concat(labels_df, ignore_index=True)

    # Merge with activity data
    merged_df = pd.merge(merged_df, labels_df, on=['id', 'date'], how='left')

    return merged_df


def add_agitation_next_column_day(input_df):
    df = input_df.copy()
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


def vitals_to_half(df):
    # Convert to datetime if needed
    df['date-time'] = pd.to_datetime(df['date-time'])
    df['date'] = df['date-time'].dt.date
    df['hour'] = df['date-time'].dt.hour

    # Define quarter
    df['12h'] = pd.cut(
        df['date-time'].dt.hour,
        bins=[0, 12, 24],
        labels=['00-12', '12-24'],
        right=False
    )

    # List of expected device types (measurement names)
    features = [
        'body-temperature', 'systolic-blood-pressure', 'diastolic-blood-pressure',
        'heart-rate', 'body-weight', 'muscle-mass', 'total-body-water', 'skin-temperature'
    ]

    # Group and compute mean value of each measurement in each quarter
    df_grouped = df.groupby(['id', 'date', '12h', 'type'])['value'].mean().reset_index()

    # Pivot so each device_type becomes a column
    pivot_df = df_grouped.pivot_table(
        index=['id', 'date', '12h'],
        columns='type',
        values='value'
    ).reset_index()

    # Rename columns (flatten multi-index if needed)
    pivot_df.columns.name = None

    # Ensure all required feature columns are present
    for feature in features:
        if feature not in pivot_df.columns:
            pivot_df[feature] = -1  # Fill missing columns

    # Fill missing values with -1
    pivot_df[features] = pivot_df[features].fillna(-1)

    # Round all floats to 4 decimal points
    pivot_df[features] = pivot_df[features].round(2)

    return pivot_df


def count_type_per_half(df):


    # Ensure 'date' is datetime
    df['date'] = pd.to_datetime(df['date-time']).dt.date

    # Define quarter bins
    df['12h'] = pd.cut(
        df['date-time'].dt.hour,
        bins=[0, 12, 24],
        labels=['00-12', '12-24'],
        right=False
    )

    # Desired types
    expected_types = ['blood-pressure', 'agitation', 'body-water', 'pulse', 'weight', 'body-temperature-label']

    # Count occurrences of each type per id, day, quarter
    counts = df.groupby(['id', 'date', '12h', 'type']).size().reset_index(name='count')

    # Pivot to make type columns
    pivot_df = counts.pivot_table(
        index=['id', 'date', '12h'],
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
    ordered_cols = ['id', 'date', '12h'] + expected_types
    pivot_df = pivot_df[ordered_cols]

    return pivot_df


def add_agitation_next_column_half(input_df):
    df = input_df.copy()
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