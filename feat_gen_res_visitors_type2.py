
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

air_res['diff_days'] = (air_res['visit_date'] - air_res['reserve_date']).dt.days
air_res['weekday'] = air_res['visit_datetime'].dt.weekday

print('Create features')

for d in range(39):
	for lag in range(3):
		feat_name = 'res_visitors_t2_day_lag' + np.str(lag) + '_day' + np.str(d)
		if not os.path.isfile('features/test/' + 'count_' + feat_name + '.npy'):
			print('sum_' + feat_name)
			print('count_' + feat_name)
			df = air_res.copy()
			df = df[df['diff_days'] > d]
			df['visit_date'] += pd.DateOffset(lag)
			df = df.groupby(['unique_id', 'visit_date'])['reserve_visitors'].aggregate(['sum', 'count']).reset_index()
			df = tr_te[['unique_id','visit_date']].merge(df, on = ['unique_id','visit_date'], how = 'left')
			f = 'sum_' + feat_name
			x = df['sum'].values
			np.save('features/train/' + f, x[:ntrain])
			np.save('features/test/' + f, x[ntrain:])
			f = 'count_' + feat_name
			x = df['count'].values
			np.save('features/train/' + f, x[:ntrain])
			np.save('features/test/' + f, x[ntrain:])


for d in range(39):
	for lag in range(4):
		feat_name = 'res_visitors_t2_weekday_lag' + np.str(lag) + '_day' + np.str(d)
		if not os.path.isfile('features/test/' + 'count_' + feat_name + '.npy'):
			print('sum_' + feat_name)
			print('count_' + feat_name)
			df = air_res.copy()
			df = df[df['diff_days'] > d]
			df['visit_date'] += pd.DateOffset(lag*7)
			df = df.groupby(['unique_id', 'visit_date'])['reserve_visitors'].aggregate(['sum', 'count']).reset_index()
			df = tr_te[['unique_id','visit_date']].merge(df, on = ['unique_id','visit_date'], how = 'left')
			f = 'sum_' + feat_name
			x = df['sum'].values
			np.save('features/train/' + f, x[:ntrain])
			np.save('features/test/' + f, x[ntrain:])
			f = 'count_' + feat_name
			x = df['count'].values
			np.save('features/train/' + f, x[:ntrain])
			np.save('features/test/' + f, x[ntrain:])


for d in range(39):
	for lag in range(3):
		feat_name = 'res_visitors_t2_week_lag' + np.str(lag) + '_day' + np.str(d)
		if not os.path.isfile('features/test/' + 'count_' + feat_name + '.npy'):
			print('sum_' + feat_name)
			print('count_' + feat_name)
			res = air_res[air_res['diff_days'] > d].copy()
			sel = pd.Series(range(7)) + lag*7
			df = []
			for i in sel:
				tmp = pd.DataFrame({'unique_id': res['unique_id'], 'visit_date': res['visit_date'] + pd.DateOffset(i), 'reserve_visitors': res['reserve_visitors']}) 
				df.append(tmp)
			df = pd.concat(df)
			df = df.groupby(['unique_id', 'visit_date'])['reserve_visitors'].aggregate(['sum', 'count']).reset_index()
			df = tr_te[['unique_id', 'visit_date']].merge(df, on = ['unique_id', 'visit_date'], how = 'left')
			f = 'sum_' + feat_name
			x = df['sum'].values
			np.save('features/train/' + f, x[:ntrain])
			np.save('features/test/' + f, x[ntrain:])
			f = 'count_' + feat_name
			x = df['count'].values
			np.save('features/train/' + f, x[:ntrain])
			np.save('features/test/' + f, x[ntrain:])

for d in range(39):
	for lag in range(3):
		feat_name = 'res_visitors_t2_month_lag' + np.str(lag) + '_day' + np.str(d)
		if not os.path.isfile('features/test/' + 'count_' + feat_name + '.npy'):
			print('sum_' + feat_name)
			print('count_' + feat_name)
			res = air_res[air_res['diff_days'] > d].copy()
			sel = pd.Series(range(28)) + lag*28
			df = []
			for i in sel:
				tmp = pd.DataFrame({'unique_id': res['unique_id'], 'visit_date': res['visit_date'] + pd.DateOffset(i), 'reserve_visitors': res['reserve_visitors']}) 
				df.append(tmp)
			df = pd.concat(df)
			df = df.groupby(['unique_id', 'visit_date'])['reserve_visitors'].aggregate(['sum', 'count']).reset_index()
			df = tr_te[['unique_id', 'visit_date']].merge(df, on = ['unique_id', 'visit_date'], how = 'left')
			f = 'sum_' + feat_name
			x = df['sum'].values
			np.save('features/train/' + f, x[:ntrain])
			np.save('features/test/' + f, x[ntrain:])
			f = 'count_' + feat_name
			x = df['count'].values
			np.save('features/train/' + f, x[:ntrain])
			np.save('features/test/' + f, x[ntrain:])


for d in range(39):
	for lag in range(4):
		feat_name = 'res_visitors_t2_4weekdays_lag' + np.str(lag) + '_day' + np.str(d)
		if not os.path.isfile('features/test/' + 'count_' + feat_name + '.npy'):
			print('sum_' + feat_name)
			print('count_' + feat_name)
			res = air_res[air_res['diff_days'] > d].copy()
			sel = pd.Series(range(4))*7 + lag*28
			df = []
			for i in sel:
				tmp = pd.DataFrame({'unique_id': res['unique_id'], 'visit_date': res['visit_date'] + pd.DateOffset(i), 'reserve_visitors': res['reserve_visitors']}) 
				df.append(tmp)
			df = pd.concat(df)
			df = df.groupby(['unique_id', 'visit_date'])['reserve_visitors'].aggregate(['sum', 'count']).reset_index()
			df = tr_te[['unique_id', 'visit_date']].merge(df, on = ['unique_id', 'visit_date'], how = 'left')
			f = 'sum_' + feat_name
			x = df['sum'].values
			np.save('features/train/' + f, x[:ntrain])
			np.save('features/test/' + f, x[ntrain:])
			f = 'count_' + feat_name
			x = df['count'].values
			np.save('features/train/' + f, x[:ntrain])
			np.save('features/test/' + f, x[ntrain:])



print('Done!!!')


