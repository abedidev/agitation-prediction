import os.path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from utils import count_location_per_day, count_location_per_hour, \
    count_location_per_half_day, extract_activity_features, extract_hourly_stats_per_quarter_paper, vitals_to_quarter, \
    merge_activity_physiology, count_type_per_quarter, count_location_per_quarter, merge_merged_label, \
    add_agitation_next_column, vitals_to_day, count_type_per_day, merge_activity_physiology_day, merge_merged_label_day, \
    add_agitation_next_column_day

activity = pd.read_csv('Activity.csv')
physiology = pd.read_csv('Physiology.csv')
sleep = pd.read_csv('Sleep.csv')
labels = pd.read_csv('Labels.csv')

# Standardize location names
activity['location_name'] = activity['location_name'].replace({
    'Back Door': 'back-door',
    'Bathroom': 'bathroom',
    'Bedroom': 'bedroom',
    'Fridge Door': 'fridge-door',
    'Front Door': 'front-door',
    'Hallway': 'hallway',
    'Kitchen': 'kitchen',
    'Lounge': 'lounge'
})


physiology['device_type'] = physiology['device_type'].replace({
    'Body Temperature': 'body-temperature',
    'Systolic blood pressure': 'systolic-blood-pressure',
    'Diastolic blood pressure': 'diastolic-blood-pressure',
    'Heart rate': 'heart-rate',
    'Body weight': 'body-weight',
    'O/E - muscle mass': 'muscle-mass',
    'Total body water': 'total-body-water',
    'Skin Temperature': 'skin-temperature'
})


labels['type'] = labels['type'].replace({
    'Blood pressure': 'blood-pressure',
    'Agitation': 'agitation',
    'Body water': 'body-water',
    'Pulse': 'pulse',
    'Weight': 'weight',
    'Body temperature': 'body-temperature-label'
})

physiology = physiology.drop(['unit'], axis=1)

physiology.to_csv('-.csv', index=False)


activity = activity.rename(columns={
    'patient_id': 'id',
    'date': 'date-time',
    'location_name': 'location'})

physiology = physiology.rename(columns={
    'patient_id': 'id',
    'date': 'date-time',
    'device_type': 'type'})

labels = labels.rename(columns={
    'patient_id': 'id',
    'date': 'date-time'})
sleep = sleep.rename(columns={
    'patient_id': 'id',
    'date': 'date-time'})


print(activity.shape,
      physiology.shape,
      sleep.shape,
      labels.shape)

print(activity.columns)
print(physiology.columns)
print(sleep.columns)
print(labels.columns)

# print(activity.date)
# print(physiology.date)
# print(sleep.date)
# print(labels.date)

# print(activity.location_name.unique())
# print(physiology.device_type.unique())
# print(sleep.state.unique())
# print(labels.type.unique())

print(activity.id.unique().shape)
print(physiology.id.unique().shape)
print(sleep.id.unique().shape)
print(labels.id.unique().shape)

# df = activity
# df['date-time'] = pd.to_datetime(df['date-time'])
# df['year'] = df['date-time'].dt.year
# df['hour'] = df['date-time'].dt.hour

count = 0

destination = '/home/ali/PycharmProjects/tihm/activity'
for idx, (id, group) in enumerate(activity.sort_values('id').groupby('id')):

    group['date-time'] = pd.to_datetime(group['date-time'])
    group = group.sort_values('date-time')

    group['date'] = group['date-time'].dt.date

    # group['quarter'] = pd.cut(
    #     group['date-time'].dt.hour,
    #     bins=[0, 6, 12, 18, 24],
    #     labels=['00-06', '06-12', '12-18', '18-24'],
    #     right=False
    # )

    # group.to_csv(os.path.join(destination, str(idx).zfill(2) + '-' + id + '.csv'), index=False)
    group.to_csv(os.path.join(destination, id + '.csv'), index=False)


    group_count_day = count_location_per_day(group)
    # group_count_day.to_csv(os.path.join(destination + '-count-day', str(idx).zfill(2) + '-' + id + '.csv'), index=False)
    group_count_day.to_csv(os.path.join(destination + '-count-day', id + '.csv'), index=False)

    group_count_half = count_location_per_half_day(group)
    # group_count_half.to_csv(os.path.join(destination + '-count-half', str(idx).zfill(2) + '-' + id + '.csv'), index=False)
    group_count_half.to_csv(os.path.join(destination + '-count-half', id + '.csv'), index=False)

    group_count_quarter = count_location_per_quarter(group)
    # group_count_quarter.to_csv(os.path.join(destination + '-count-quarter', str(idx).zfill(2) + '-' + id + '.csv'), index=False)
    group_count_quarter.to_csv(os.path.join(destination + '-count-quarter', id + '.csv'), index=False)

    count += group_count_quarter.shape[0]

    group_count_hour = count_location_per_hour(group)
    # group_count_hour.to_csv(os.path.join(destination + '-count-hour', str(idx).zfill(2) + '-' + id + '.csv'), index=False)
    group_count_hour.to_csv(os.path.join(destination + '-count-hour', id + '.csv'), index=False)

    group_quarter = extract_activity_features(group)
    # group_quarter.to_csv(os.path.join(destination + '-features-quarter', str(idx).zfill(2) + '-' + id + '.csv'), index=False)
    group_quarter.to_csv(os.path.join(destination + '-features-quarter', id + '.csv'), index=False)

    group_quarter_paper = extract_hourly_stats_per_quarter_paper(group)
    # group_quarter_paper.to_csv(os.path.join(destination + '-features-paper', str(idx).zfill(2) + '-' + id + '.csv'), index=False)
    group_quarter_paper.to_csv(os.path.join(destination + '-features-paper', id + '.csv'), index=False)


    print(id, group.shape, len(group.date.unique()))

    # break



destination = '/home/ali/PycharmProjects/tihm/physiology'
for idx, (id, group) in enumerate(physiology.sort_values('id').groupby('id')):

    group['date-time'] = pd.to_datetime(group['date-time'])
    group = group.sort_values('date-time')

    group['date'] = group['date-time'].dt.date

    # group['quarter'] = pd.cut(
    #     group['date-time'].dt.hour,
    #     bins=[0, 6, 12, 18, 24],
    #     labels=['00-06', '06-12', '12-18', '18-24'],
    #     right=False
    # )

    # if idx > 1:
    #     idx += 1

    # group.to_csv(os.path.join(destination, str(idx).zfill(2) + '-' + id + '.csv'), index=False)
    group.to_csv(os.path.join(destination, id + '.csv'), index=False)

    # group_count_quarter = vitals_to_quarter(group)
    group_count_quarter = vitals_to_day(group)
    # group_count_quarter.to_csv(os.path.join(destination + '-values', str(idx).zfill(2) + '-' + id + '.csv'), index=False)
    group_count_quarter.to_csv(os.path.join(destination + '-values-day', id + '.csv'), index=False)



destination = '/home/ali/PycharmProjects/tihm/label'
for idx, (id, group) in enumerate(labels.sort_values('id').groupby('id')):

    group['date-time'] = pd.to_datetime(group['date-time'])
    group = group.sort_values('date-time')

    group['date'] = group['date-time'].dt.date

    # group['quarter'] = pd.cut(
    #     group['date-time'].dt.hour,
    #     bins=[0, 6, 12, 18, 24],
    #     labels=['00-06', '06-12', '12-18', '18-24'],
    #     right=False
    # )

    # group_count_quarter = count_type_per_quarter(group)
    group_count_quarter = count_type_per_day(group)


    # group_count_quarter.to_csv(os.path.join(destination + '-count-quarter', str(idx).zfill(2) + '-' + id + '.csv'), index=False)
    group_count_quarter.to_csv(os.path.join(destination + '-count-day', id + '.csv'), index=False)




root_activity = '/home/ali/PycharmProjects/tihm/activity-count-day'
root_physiology = '/home/ali/PycharmProjects/tihm/physiology-values-day'
root_merged = '/home/ali/PycharmProjects/tihm/merged-day'
files = sorted(os.listdir(root_activity))

for idx, file in enumerate(files):

    # break

    try:
        act = pd.read_csv(os.path.join(root_activity, file))
        phy = pd.read_csv(os.path.join(root_physiology, file))
        dataFrame = merge_activity_physiology_day(
            act,
            phy)

        dataFrame.fillna(-1, inplace=True)

        dataFrame.to_csv(os.path.join(root_merged, file), index=False)
        print(idx, file, act.isna().any().shape, phy.isna().any().shape)
    except:
        print('---', idx, file)
        # break



root_merged = '/home/ali/PycharmProjects/tihm/merged-day'
files = sorted([file for file in os.listdir(root_merged) if file.endswith('.csv')])
csvs = [pd.read_csv(os.path.join(root_merged, file)) for file in files]

merged = pd.concat(csvs, ignore_index=True)
merged.to_csv('-.csv', index=False)






adding_columns = ['blood-pressure', 'agitation', 'body-water','pulse', 'weight', 'body-temperature-label']
root_merged = '/home/ali/PycharmProjects/tihm/merged-day'
root_label = '/home/ali/PycharmProjects/tihm/label-count-day'
root_altogether = '/home/ali/PycharmProjects/tihm/altogether-day'
files = sorted(os.listdir(root_merged))

# files = ['0f352.csv']
count = 0

for idx, file in enumerate(files):


    try:
        mer = pd.read_csv(os.path.join(root_merged, file))
        lab = pd.read_csv(os.path.join(root_label, file))
        dataFrame = merge_merged_label_day(
            mer,
            lab)

        # dataFrame.fillna(-1, inplace=True)

        dataFrame.to_csv(os.path.join(root_altogether, file), index=False)
        print(idx, file, act.isna().any().shape, phy.isna().any().shape)
    except:
        count += 1
        print('---', idx, file, count)

        mer[adding_columns] = np.nan
        mer.to_csv(os.path.join(root_altogether, file), index=False)

        # break



root_altogether = '/home/ali/PycharmProjects/tihm/altogether-day'
files = sorted([file for file in os.listdir(root_altogether) if file.endswith('.csv')])
csvs = [pd.read_csv(os.path.join(root_altogether, file)) for file in files]

altogether = pd.concat(csvs, ignore_index=True)

altogether.fillna(-1, inplace=True)
for col in altogether.select_dtypes(include='float').columns:
    if (altogether[col] == altogether[col].astype(int)).all():
        altogether[col] = altogether[col].astype(int)

altogether.to_csv('dataset-raw-day.csv', index=False)

next = add_agitation_next_column_day(altogether)
next = next[next['agitation-next'] != -10]
next.to_csv('dataset-raw-next-day.csv', index=False)

# altogether[altogether['agitation'] >= 1][['id', 'date', 'quarter', 'agitation']].to_csv('agitation.csv')
# labels[labels['type'] == 'agitation'].to_csv('agitation-.csv')

len(altogether)
len(altogether['id'].unique())
len(altogether[['id', 'date']].drop_duplicates())
len(altogether[['date']].drop_duplicates())
len(altogether[['id', 'date']].drop_duplicates())