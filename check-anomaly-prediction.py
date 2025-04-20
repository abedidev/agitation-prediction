# DeepSVDD
import numpy as np
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, f1_score, accuracy_score, \
    precision_score, recall_score
from deepod.models.tabular.dsvdd import DeepSVDD
from deepod.models.tabular.icl import ICL
from sklearn.model_selection import KFold, LeavePGroupsOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler

seed = 69

x = np.load(os.path.join('/home/ali/PycharmProjects/tihm/xyp', 'x.npy'))
y = np.load(os.path.join('/home/ali/PycharmProjects/tihm/xyp', 'y.npy'))
p = np.load(os.path.join('/home/ali/PycharmProjects/tihm/xyp', 'p.npy'))


kfold = True


name = 'DeepSVDD'.lower()

Y_TRUES = np.empty([0])
Y_PROBS = np.empty([0])
Y_PREDS = np.empty([0])


if kfold:
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    split_iterator = cv.split(x)
else:
    cv = LeavePGroupsOut(n_groups=1)
    split_iterator = cv.split(x, y, groups=p)

for i, (train_idx, test_idx) in enumerate(split_iterator, start=1):
    participant = i if kfold else np.unique(p[test_idx])[0]

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

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

    # # 1. Check original class distribution
    # print("Original class distribution:", Counter(y_train))
    # # 2. Define the minority class (adjust if needed)
    # minority_class = 1  # change this if your minority class label is different
    # current_minority_count = sum(y_train == minority_class)
    # # 3. Define desired new total count for the minority class (10x)
    # target_minority_count = current_minority_count * 10
    # # 4. Setup SMOTE with custom sampling strategy
    # smote = SMOTE(sampling_strategy={minority_class: target_minority_count}, random_state=seed)
    # # 5. Fit and resample
    # x_train, y_train = smote.fit_resample(x_train, y_train)
    # # 6. Confirm new distribution
    # print("Resampled class distribution:", Counter(y_train))


    # Undersample class 0
    # data_percent = 1
    # reduction_percent = 100 - data_percent
    # print("Original class distribution:", Counter(y_train))    
    # class_0_idx = np.where(y_train == 0)[0]
    # class_1_idx = np.where(y_train == 1)[0]
    # # How many class 0 samples to keep
    # n_keep = int(len(class_0_idx) * (1 - reduction_percent / 100.0))
    # sampled_0_idx = np.random.choice(class_0_idx, size=n_keep, replace=False)
    # # Combine sampled class 0 with all class 1
    # final_idx = np.concatenate([sampled_0_idx, class_1_idx])
    # x_train = x_train[final_idx]
    # y_train =  y_train[final_idx]
    # print("Resampled class distribution:", Counter(y_train))


    # DeepSVDD
    # model = DeepSVDD()
    model = ICL()
    model.fit(x_train, y=None)

    y_probs = model.decision_function(x_test)




    Y_TRUES = np.append(Y_TRUES, y_test)
    Y_PROBS = np.append(Y_PROBS, y_probs)
    # Y_PREDS = np.append(Y_PREDS, y_preds)


indx = Y_TRUES.argsort()
Y_TRUES = Y_TRUES[indx]
Y_PROBS = Y_PROBS[indx]
# Y_PREDS = Y_PREDS[indx]

auc_roc = roc_auc_score(Y_TRUES, Y_PROBS)
precision, recall, _ = precision_recall_curve(Y_TRUES, Y_PROBS)
auc_pr = auc(recall, precision)
print(auc_roc, auc_pr)