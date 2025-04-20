import os
import warnings

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, f1_score
from sklearn.model_selection import KFold, LeavePGroupsOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
import matplotlib
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tabpfn import TabPFNClassifier

matplotlib.use('TkAgg')
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler  # install via: pip install imbalanced-learn

warnings.filterwarnings('ignore')
import torch
from scipy.stats import skew
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

seed = 69
torch.manual_seed(seed)
np.random.seed(seed)


root = '/home/ali/PycharmProjects/tihm'

# dataset = pd.read_csv(os.path.join(root, 'dataset-raw.csv'))
# dataset = pd.read_csv(os.path.join(root, 'dataset-raw-next.csv'))

dataset = pd.read_csv(os.path.join(root, 'dataset-paper.csv'))
# dataset = pd.read_csv(os.path.join(root, 'dataset-features.csv'))


labels = ['agitation']
# labels = ['agitation-next']


y = np.array(dataset[labels]).squeeze()
y[y == -1] = 0

ids = np.array(dataset['id']).squeeze()
p = np.unique(ids, return_inverse=True)[1]


# columns = ['id', 'day', 'quarter', 'back-door', 'bathroom', 'bedroom',
#        'fridge-door', 'front-door', 'hallway', 'kitchen', 'lounge',
#
#        'body-temperature', 'body-weight', 'diastolic-blood-pressure',
#        'heart-rate', 'muscle-mass', 'systolic-blood-pressure',
#        'total-body-water', 'skin-temperature']

columns = ['id', 'day', 'quarter', 'back-door', 'bathroom', 'bedroom',
       'fridge-door', 'front-door', 'hallway', 'kitchen', 'lounge']

# columns = ['id', 'day', 'quarter', 'back-door-count-max', 'back-door-count-mean',
#        'back-door-count-std', 'back-door-count-sum', 'bathroom-count-max',
#        'bathroom-count-mean', 'bathroom-count-std', 'bathroom-count-sum',
#        'bedroom-count-max', 'bedroom-count-mean', 'bedroom-count-std',
#        'bedroom-count-sum', 'fridge-door-count-max', 'fridge-door-count-mean',
#        'fridge-door-count-std', 'fridge-door-count-sum',
#        'front-door-count-max', 'front-door-count-mean', 'front-door-count-std',
#        'front-door-count-sum', 'hallway-count-max', 'hallway-count-mean',
#        'hallway-count-std', 'hallway-count-sum', 'kitchen-count-max',
#        'kitchen-count-mean', 'kitchen-count-std', 'kitchen-count-sum',
#        'lounge-count-max', 'lounge-count-mean', 'lounge-count-std',
#        'lounge-count-sum', 'body-temperature', 'body-weight',
#        'diastolic-blood-pressure', 'heart-rate', 'muscle-mass',
#        'systolic-blood-pressure', 'total-body-water', 'skin-temperature']

columns = ['id', 'day', 'quarter', 'back-door-count-max', 'back-door-count-mean',
       'back-door-count-std', 'back-door-count-sum', 'bathroom-count-max',
       'bathroom-count-mean', 'bathroom-count-std', 'bathroom-count-sum',
       'bedroom-count-max', 'bedroom-count-mean', 'bedroom-count-std',
       'bedroom-count-sum', 'fridge-door-count-max', 'fridge-door-count-mean',
       'fridge-door-count-std', 'fridge-door-count-sum',
       'front-door-count-max', 'front-door-count-mean', 'front-door-count-std',
       'front-door-count-sum', 'hallway-count-max', 'hallway-count-mean',
       'hallway-count-std', 'hallway-count-sum', 'kitchen-count-max',
       'kitchen-count-mean', 'kitchen-count-std', 'kitchen-count-sum',
       'lounge-count-max', 'lounge-count-mean', 'lounge-count-std',
       'lounge-count-sum']

# columns = ['id', 'day', 'quarter', 'back-door', 'bathroom', 'bedroom',
#        'fridge-door', 'front-door', 'hallway', 'kitchen', 'lounge',
#        'total-events', 'unique-locations', 'active-location-ratio',
#        'private-to-public-ratio', 'location-entropy',
#        'location-dominance-ratio', 'back-and-forth-count', 'num-transitions']

# columns = ['id', 'day', 'quarter', 'back-door', 'bathroom', 'bedroom',
#        'fridge-door', 'front-door', 'hallway', 'kitchen', 'lounge',
#        'total-events', 'unique-locations', 'active-location-ratio',
#        'private-to-public-ratio', 'location-entropy',
#        'location-dominance-ratio', 'back-and-forth-count', 'num-transitions',
#        'body-temperature', 'body-weight', 'diastolic-blood-pressure',
#        'heart-rate', 'muscle-mass', 'systolic-blood-pressure',
#        'total-body-water', 'skin-temperature']

data = dataset[columns]


for ind, row in data.iterrows():
    nan_columns = data.columns[row == -1]
    if len(nan_columns) > 0:
        for column in nan_columns:
            temp = data[data['day'] == row['day']][column]
            temp = temp[temp != -1]
            if len(temp) > 0:
                data.loc[data.index == ind, column] = temp.mean().round(4)
            else:
                temp = data[data['id'] == row['id']][column]
                temp = temp[temp != -1]
                if len(temp) > 0:
                    data.loc[data.index == ind, column] = temp.mean().round(4)

                else:
                    temp = data[column]
                    temp = temp[temp != -1]
                    if len(temp) > 0:
                        data.loc[data.index == ind, column] = temp.mean().round(1)

            print(ind, column)

data.to_csv('imputation.csv', index=False)


data.drop(['id', 'day', 'quarter'], axis=1, inplace=True)
x = np.array(data)

np.savez('data.npz', x=x, y=y)


