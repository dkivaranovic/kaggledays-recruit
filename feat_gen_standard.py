
#### libraries
import numpy as np
import pandas as pd
import gc, os

print('Read data')
train = pd.read_pickle('input/train.pkl')
test = pd.read_pickle('input/test.pkl')

print('Concat train/test set')
ntrain = train.shape[0]
tr_te = pd.concat([train, test]).reset_index().drop('index', axis = 1)
del(train, test)
gc.collect()

print('Create features')

feat = ['unique_id', 'year', 'month', 'week', 'day', 'weekday', 'holiday', 'holiday_2', 'holiday_lag-2', 'holiday_lag-1', 'holiday_lag1', 
      'holiday_lag2', 'golden_week', 'air_genre_name', 'air_area_name', 'latitude', 'longitude', 'hpg_genre_name', 'hpg_area_name',
      ]

for f in feat:
	feat_name = f
	if not os.path.isfile('features/test/' + feat_name):
		print(feat_name)
		x = tr_te[f].values
		np.save('features/train/' + feat_name, x[:ntrain])
		np.save('features/test/' + feat_name, x[ntrain:])


print('Done!!!')

