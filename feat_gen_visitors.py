
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

for d in range(39):
	for lag in range(1, 15):
		feat_name = 'visitors_day_lag' + np.str(lag) + '_day' + np.str(d)
		if not os.path.isfile('features/test/' + feat_name + '.npy'):
			print(feat_name)
			df = pd.DataFrame({'unique_id': tr_te['unique_id'], 'visit_date': tr_te['visit_date'] + pd.DateOffset(lag+d), 'visitors': tr_te['visitors']})
			df = tr_te[['unique_id','visit_date']].merge(df, on = ['unique_id','visit_date'], how = 'left')
			x = df['visitors'].values
			np.save('features/train/' + feat_name, x[:ntrain])
			np.save('features/test/' + feat_name, x[ntrain:])


for d in range(39):
	for lag in range(1, 14):
		feat_name = 'visitors_weekday_lag' + np.str(lag) + '_day' + np.str(d)
		if not os.path.isfile('features/test/' + feat_name + '.npy'):
			print(feat_name)
			df = pd.DataFrame({'unique_id': tr_te['unique_id'], 'visit_date': tr_te['visit_date'] + pd.DateOffset((lag+d//7)*7), 'visitors': tr_te['visitors']})
			df = tr_te[['unique_id','visit_date']].merge(df, on = ['unique_id','visit_date'], how = 'left')
			x = df['visitors'].values
			np.save('features/train/' + feat_name, x[:ntrain])
			np.save('features/test/' + feat_name, x[ntrain:])


for d in range(39):
	for lag in range(1, 25):
		feat_name = 'mean_visitors_week_lag' + np.str(lag) + '_day' + np.str(d)
		if not os.path.isfile('features/test/' + feat_name + '.npy'):
			print(feat_name)
			sel = pd.Series(range(1,8)) + (lag-1)*7 + d
			df = []
			for i in sel:
				tmp = pd.DataFrame({'unique_id': tr_te['unique_id'], 'visit_date': tr_te['visit_date'] + pd.DateOffset(i), 'visitors': tr_te['visitors']}) 
				df.append(tmp)
			df = pd.concat(df)
			df = df.groupby(['unique_id', 'visit_date'])['visitors'].mean().reset_index()
			df = tr_te[['unique_id', 'visit_date']].merge(df, on = ['unique_id', 'visit_date'], how = 'left')
			x = df['visitors'].values
			np.save('features/train/' + feat_name, x[:ntrain])
			np.save('features/test/' + feat_name, x[ntrain:])


for d in range(39):
	for lag in range(1, 13):
		feat_name = 'mean_visitors_month_lag' + np.str(lag) + '_day' + np.str(d)
		if not os.path.isfile('features/test/' + feat_name + '.npy'):
			print(feat_name)
			sel = pd.Series(range(1,29)) + (lag-1)*28 + d
			df = []
			for i in sel:
				tmp = pd.DataFrame({'unique_id': tr_te['unique_id'], 'visit_date': tr_te['visit_date'] + pd.DateOffset(i), 'visitors': tr_te['visitors']}) 
				df.append(tmp)
			df = pd.concat(df)
			df = df.groupby(['unique_id', 'visit_date'])['visitors'].mean().reset_index()
			df = tr_te[['unique_id', 'visit_date']].merge(df, on = ['unique_id', 'visit_date'], how = 'left')
			x = df['visitors'].values
			np.save('features/train/' + feat_name, x[:ntrain])
			np.save('features/test/' + feat_name, x[ntrain:])


for d in range(39):
	for lag in range(1, 13):
		feat_name = 'mean_visitors_4weekdays_lag' + np.str(lag) + '_day' + np.str(d)
		if not os.path.isfile('features/test/' + feat_name + '.npy'):
			print(feat_name)
			sel = pd.Series(range(1,5))*7 + (lag-1)*28 + 7*(d//7)
			df = []
			for i in sel:
				tmp = pd.DataFrame({'unique_id': tr_te['unique_id'], 'visit_date': tr_te['visit_date'] + pd.DateOffset(i), 'visitors': tr_te['visitors']}) 
				df.append(tmp)
			df = pd.concat(df)
			df = df.groupby(['unique_id', 'visit_date'])['visitors'].mean().reset_index()
			df = tr_te[['unique_id', 'visit_date']].merge(df, on = ['unique_id', 'visit_date'], how = 'left')
			x = df['visitors'].values
			np.save('features/train/' + feat_name, x[:ntrain])
			np.save('features/test/' + feat_name, x[ntrain:])


print('Done!!!')


