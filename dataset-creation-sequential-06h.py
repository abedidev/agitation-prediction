import os
import numpy as np
import pandas as pd
import warnings

from utils_data import hierarchical_imputation_columns_to_exclude, segment_dataframe

warnings.filterwarnings('ignore')




sh


root = '/home/ali/PycharmProjects/tihm/dataset/'

data = pd.read_csv(os.path.join(root, 'data-06h.csv'))

data = hierarchical_imputation_columns_to_exclude(data,
                                                  columns_to_exclude=['id', 'date', '6h', 'agitation', 'agitation-next', 'agitation-four'])

# data.fillna(0, inplace=True)

segs, labs = segment_dataframe(data, 6)

x = np.array(segs)
y = np.array(labs)

