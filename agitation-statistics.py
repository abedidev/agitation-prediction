import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score,confusion_matrix,precision_score,recall_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler

# import matplotlib.pyplot as plt
import seaborn as sns
import shap

from data_utils import correct_col_type,transform_category_to_counts,min_max_perpatient


data = pd.read_csv('participant_summary_df.csv')

print(data['agitation_episodes'].mean().round(4),
      data['agitation_episodes'].std().round(4),
      data['agitation_episodes'].min().round(4),
      data['agitation_episodes'].max().round(4),
      data[data['agitation_episodes'] > 0]['agitation_episodes'].mean().round(4),
      data[data['agitation_episodes'] > 0]['agitation_episodes'].std().round(4))

print(data['unique_dates'].mean().round(4),
      data['unique_dates'].std().round(4),
      data['unique_dates'].min().round(4),
      data['unique_dates'].max().round(4),
      data[data['unique_dates'] > 0]['unique_dates'].mean().round(4),
      data[data['unique_dates'] > 0]['unique_dates'].std().round(4))


dataset = pd.read_csv('dataset.csv')
dataset.fillna(0)

ids = dataset[dataset['agitation'] > 0]['id'].to_list()
diffs = []

for id in ids:
      dataset['day'] = pd.to_datetime(dataset['day'])
      date_diffs = dataset[np.logical_and(dataset['id'] == id, dataset['agitation'] == 1)]['day'].diff().dropna()
      diffs += list(date_diffs.dt.total_seconds() / (60 * 60 * 24))

print(np.mean(diffs).round(2),
      np.std(diffs).round(2))