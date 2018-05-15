
#### libraries
import numpy as np
import pandas as pd
import gc, os

print('Read data')
train = pd.read_pickle('input/train.pkl')
test = pd.read_pickle('input/test.pkl')
air_res = pd.read_pickle('input/air_res.pkl')

print('Concat train/test set')
ntrain = train.shape[0]
tr_te = pd.concat([train, test]).reset_index().drop('index', axis = 1)
del(train, test)
gc.collect()

#### take total number of reservations per day
res = air_res.groupby(['unique_id', 'visit_date'])['reserve_visitors'].sum().reset_index()

print('Create features')

for d in range(39):
	for lag in range(1, 13):
		feat_name = 'mean_res_visitors_week_lag' + np.str(lag) + '_day' + np.str(d)
		if not os.path.isfile('features/test/' + feat_name + '.npy'):
			print(feat_name)
			sel = pd.Series(range(1,8)) + (lag-1)*7 + d
			df = []
			for i in sel:
				tmp = pd.DataFrame({'unique_id': res['unique_id'], 'visit_date': res['visit_date'] + pd.DateOffset(i), 'reserve_visitors': res['reserve_visitors']}) 
				df.append(tmp)
			df = pd.concat(df)
			df = df.groupby(['unique_id', 'visit_date'])['reserve_visitors'].mean().reset_index()
			df = tr_te[['unique_id', 'visit_date']].merge(df, on = ['unique_id', 'visit_date'], how = 'left')
			x = df['reserve_visitors'].values
			np.save('features/train/' + feat_name, x[:ntrain])
			np.save('features/test/' + feat_name, x[ntrain:])


for d in range(39):
	for lag in range(1, 7):
		feat_name = 'mean_res_visitors_month_lag' + np.str(lag) + '_day' + np.str(d)
		if not os.path.isfile('features/test/' + feat_name + '.npy'):
			print(feat_name)
			sel = pd.Series(range(1,29)) + (lag-1)*28 + d
			df = []
			for i in sel:
				tmp = pd.DataFrame({'unique_id': res['unique_id'], 'visit_date': res['visit_date'] + pd.DateOffset(i), 'reserve_visitors': res['reserve_visitors']}) 
				df.append(tmp)
			df = pd.concat(df)
			df = df.groupby(['unique_id', 'visit_date'])['reserve_visitors'].mean().reset_index()
			df = tr_te[['unique_id', 'visit_date']].merge(df, on = ['unique_id', 'visit_date'], how = 'left')
			x = df['reserve_visitors'].values
			np.save('features/train/' + feat_name, x[:ntrain])
			np.save('features/test/' + feat_name, x[ntrain:])


for d in range(39):
	for lag in range(1, 7):
		feat_name = 'mean_res_visitors_4weekdays_lag' + np.str(lag) + '_day' + np.str(d)
		if not os.path.isfile('features/test/' + feat_name + '.npy'):
			print(feat_name)
			sel = pd.Series(range(1,5))*7 + (lag-1)*28 + 7*(d//7)
			df = []
			for i in sel:
				tmp = pd.DataFrame({'unique_id': res['unique_id'], 'visit_date': res['visit_date'] + pd.DateOffset(i), 'reserve_visitors': res['reserve_visitors']}) 
				df.append(tmp)
			df = pd.concat(df)
			df = df.groupby(['unique_id', 'visit_date'])['reserve_visitors'].mean().reset_index()
			df = tr_te[['unique_id', 'visit_date']].merge(df, on = ['unique_id', 'visit_date'], how = 'left')
			x = df['reserve_visitors'].values
			np.save('features/train/' + feat_name, x[:ntrain])
			np.save('features/test/' + feat_name, x[ntrain:])

print('Done!!!')





