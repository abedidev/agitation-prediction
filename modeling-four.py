import os
import warnings

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, f1_score, accuracy_score, \
    precision_score, recall_score
from sklearn.model_selection import KFold, LeavePGroupsOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
import matplotlib
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
matplotlib.use('TkAgg')
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler  # install via: pip install imbalanced-learn

warnings.filterwarnings('ignore')
import torch
from scipy.stats import skew
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from tabpfn import TabPFNClassifier

seed = 69
torch.manual_seed(seed)
np.random.seed(seed)


root = '/home/ali/PycharmProjects/tihm'

# dataset = pd.read_csv(os.path.join(root, 'dataset-raw.csv'))
dataset = pd.read_csv(os.path.join(root, 'dataset-raw-next.csv'))

# dataset = pd.read_csv(os.path.join(root, 'dataset-paper.csv'))
# dataset = pd.read_csv(os.path.join(root, 'dataset-features.csv'))


# labels = ['agitation']
labels = ['agitation-four']


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
#        'lounge-count-sum']
#
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




Y_TRUES = np.empty([0])
Y_PROBS = []
Y_PREDS = np.empty([0])


# cv = LeavePGroupsOut(n_groups=1)
# for train_idx, test_idx in cv.split(x, y, groups=p):
#     participant = np.unique(p[test_idx])[0]


# cv = KFold(n_splits=x.shape[0], shuffle=True, random_state=seed)
# for fold, (train_idx, test_idx) in enumerate(cv.split(x), start=1):
#     participant = np.unique(p[test_idx])[0]

cv = KFold(n_splits=5, shuffle=True, random_state=seed)
for fold, (train_idx, test_idx) in enumerate(cv.split(x), start=1):
    participant = fold

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print(participant, x_train.shape[0], x_test.shape[0])

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    normalizer = MinMaxScaler()
    normalizer.fit(x_train)
    x_train = normalizer.transform(x_train)
    x_test = normalizer.transform(x_test)

    # Oversample class 1
    # ros = RandomOverSampler(sampling_strategy=.1, random_state=seed)
    # x_train, y_train = ros.fit_resample(x_train, y_train)

    # ----- LightGBM
    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)
    params = {
        'objective': 'multiclass',
        'num_class': 4,  # Specify number of classes
        'metric': 'multi_logloss',  # or 'auc' if you prefer
        'num_leaves': 31,
        'learning_rate': 0.01,
        'n_estimators': 100,
        # 'is_unbalance': True  # Automatically balances positive and negative classes
        # 'scale_pos_weight': scale_pos_weight
    }
    bst = lgb.train(params, train_data, valid_sets=[train_data, test_data])
    y_probs = bst.predict(x_test, num_iteration=bst.best_iteration)



    # ----- Gradient Boosting Classifier
    # model = GradientBoostingClassifier(
    #     n_estimators=64,
    #     learning_rate=.1,
    #     max_depth=8,
    #     random_state=seed
    # )
    # model.fit(x_train, y_train)
    # y_probs = model.predict_proba(x_test)

    # ----- XGBoost
    # model = XGBClassifier()
    # model.fit(x_train, y_train)
    # y_probs = model.predict_proba(x_test)


    # ----- Random Forest
    # model = RandomForestClassifier(
    #     n_estimators=100,  # number of trees
    #     max_depth=None,  # let the trees grow fully
    #     random_state=seed
    # )
    # model.fit(x_train, y_train)
    # y_probs = model.predict_proba(x_test)


    # ----- SVC
    # model = SVC(kernel='rbf', probability=True, random_state=seed)
    # model.fit(x_train, y_train)
    # y_probs = model.predict_proba(x_test)

    # ----- DT
    # model = DecisionTreeClassifier(criterion="log_loss", max_depth=5, random_state=seed)
    # model.fit(x_train, y_train)
    # y_probs = model.predict(x_test)

    # ----- Transformer
    # model = TabPFNClassifier()
    # model.fit(x_train, y_train)
    # y_probs = model.predict_proba(x_test)



    Y_TRUES = np.append(Y_TRUES, y_test)
    Y_PROBS.append(y_probs)
    Y_PREDS = np.append(Y_PREDS, np.argmax(y_probs, axis=1))


Y_PROBS = np.concatenate(Y_PROBS, axis=0)

print(

    roc_auc_score(label_binarize(Y_TRUES, classes=[0, 1, 2, 3]), Y_PROBS, average='micro', multi_class='ovo').__round__(4),
    roc_auc_score(label_binarize(Y_TRUES, classes=[0, 1, 2, 3]), Y_PROBS, average='macro', multi_class='ovo').__round__(4),
    roc_auc_score(label_binarize(Y_TRUES, classes=[0, 1, 2, 3]), Y_PROBS, average='samples', multi_class='ovo').__round__(4),
    roc_auc_score(label_binarize(Y_TRUES, classes=[0, 1, 2, 3]), Y_PROBS, average='weighted', multi_class='ovo').__round__(4),

    # roc_auc_score(label_binarize(Y_TRUES, classes=[0, 1, 2, 3]), Y_PROBS, average='micro', multi_class='ovr').__round__(4),
    # roc_auc_score(label_binarize(Y_TRUES, classes=[0, 1, 2, 3]), Y_PROBS, average='macro', multi_class='ovr').__round__(4),
    # roc_auc_score(label_binarize(Y_TRUES, classes=[0, 1, 2, 3]), Y_PROBS, average='samples',multi_class='ovr').__round__(4),
    # roc_auc_score(label_binarize(Y_TRUES, classes=[0, 1, 2, 3]), Y_PROBS, average='weighted',multi_class='ovr').__round__(4),

)


indx = Y_TRUES.argsort()
Y_TRUES = Y_TRUES[indx]
Y_PROBS = Y_PROBS[indx]
Y_PREDS = Y_PREDS[indx]


plt.figure()
plt.plot(Y_TRUES, 'o', color='blue', alpha=.25, markersize=8, label='Ground-truth')
plt.plot(Y_PREDS, 'o', color='red', alpha=.25, markersize=8, label='Prediction')

accuracy = accuracy_score(Y_TRUES, Y_PREDS)
precision = precision_score(Y_TRUES, Y_PREDS, average='macro')
recall = recall_score(Y_TRUES, Y_PREDS, average='macro')
f1 = f1_score(Y_TRUES, Y_PREDS, average='macro')

print(f"Accuracy (macro): {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print(confusion_matrix(Y_TRUES, Y_PREDS))