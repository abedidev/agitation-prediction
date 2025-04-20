import os.path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from utils import count_location_per_day, count_location_per_hour, \
    count_location_per_half_day, extract_activity_features, extract_hourly_stats_per_quarter_paper, vitals_to_quarter, \
    merge_activity_physiology, count_type_per_quarter, count_location_per_quarter, merge_merged_label, \
    add_agitation_next_column



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