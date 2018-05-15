
import pandas as pd

s1 = pd.read_csv('Downloads/LGB_01.csv')
s2 = pd.read_csv('Downloads/KERAS_01.csv')

df = s1.copy()
df['visitors'] = 0.7*s1['visitors'] + 0.3*s1['visitors']
df.to_csv('Downloads/blend.csv', index = False)

#### libraries
import numpy as np
import pandas as pd
import gc

print('Read data')
air_res = pd.read_csv('input/air_reserve.csv', parse_dates = ['visit_datetime', 'reserve_datetime'])
hpg_res = pd.read_csv('input/hpg_reserve.csv', parse_dates = ['visit_datetime', 'reserve_datetime'])
air_info = pd.read_csv('input/air_store_info.csv')
hpg_info = pd.read_csv('input/hpg_store_info.csv')
store_id = pd.read_csv('input/store_id_relation.csv')
date_info = pd.read_csv('input/date_info.csv', parse_dates = ['calendar_date'])
sample_sub = pd.read_csv('input/sample_submission.csv')
train = pd.read_csv('input/air_visit_data.csv', parse_dates = ['visit_date'])
test = pd.DataFrame({'air_store_id': sample_sub['id'].str[:20], 'visit_date': pd.to_datetime(sample_sub['id'].str[21:]), 'visitors': np.nan})

print('Concat train/test set')
ntrain = train.shape[0]
tr_te = pd.concat([train, test]).reset_index().drop('index', axis = 1)
del(train, test)
gc.collect()

print('Create unique id')
df1 = pd.DataFrame({'air_store_id': air_info['air_store_id'].unique()}).merge(store_id, on = 'air_store_id', how = 'left')
df2 = pd.DataFrame({'hpg_store_id': hpg_res['hpg_store_id'].unique()}).merge(store_id, on = 'hpg_store_id', how = 'left')
all_ids = pd.concat([df1, df2]).drop_duplicates().reset_index().drop('index', axis = 1)
all_ids['unique_id'] = range(all_ids.shape[0])
del(df1, df2)

print('Categorical features to numeric')
air_info['air_genre_name'] = pd.Categorical(air_info['air_genre_name']).codes
air_info['air_area_name'] = pd.Categorical(air_info['air_area_name']).codes

hpg_info['hpg_genre_name'] = pd.Categorical(hpg_info['hpg_genre_name']).codes
hpg_info['hpg_area_name'] = pd.Categorical(hpg_info['hpg_area_name']).codes

print('Append unique id to each set')
air_res = air_res.merge(all_ids[['air_store_id', 'unique_id']], on = 'air_store_id', how = 'left')
hpg_res = hpg_res.merge(all_ids[['hpg_store_id', 'unique_id']], on = 'hpg_store_id', how = 'left')
air_info = air_info.merge(all_ids[['air_store_id', 'unique_id']], on = 'air_store_id', how = 'left')
hpg_info = hpg_info.merge(all_ids[['hpg_store_id', 'unique_id']], on = 'hpg_store_id', how = 'left')
tr_te = tr_te.merge(all_ids[['air_store_id', 'unique_id']], on = 'air_store_id', how = 'left')

print('Create date features')
tr_te['visit_date'] = tr_te['visit_date'].dt.normalize()
tr_te['year'] = tr_te['visit_date'].dt.year
tr_te['month'] = tr_te['visit_date'].dt.month
tr_te['week'] = tr_te['visit_date'].dt.week
tr_te['day'] = tr_te['visit_date'].dt.day
tr_te['weekday'] = tr_te['visit_date'].dt.weekday
tr_te['week_count'] = tr_te['week'] + np.where(tr_te['visit_date'] >= '2017-01-02', 52, 0) - np.where(tr_te['visit_date'] <= '2016-01-03', 53, 0)

print('Create holiday features')
df = tr_te[['visit_date']].merge(date_info[['calendar_date', 'holiday_flg']], left_on = 'visit_date', right_on = 'calendar_date', how = 'left')
tr_te['holiday'] = df['holiday_flg']
tr_te['holiday_2'] = ((tr_te['holiday']==1) | (tr_te['weekday']>4)).astype(int)
tr_te['golden_week'] = np.where(((tr_te['visit_date'] >= '2016-04-29') & (tr_te['visit_date'] <= '2016-05-05')) | ((tr_te['visit_date'] >= '2017-04-29') & (tr_te['visit_date'] <= '2017-05-05')), 1, 0) 

for lag in [-2,-1,1,2]:
	df = date_info.copy()
	df['calendar_date'] += pd.DateOffset(lag)
	df = tr_te[['visit_date']].merge(df[['calendar_date', 'holiday_flg']], left_on = 'visit_date', right_on = 'calendar_date', how = 'left')
	tr_te['holiday' + '_lag' + np.str(lag)] = df['holiday_flg']

print('Create store info features')
df = tr_te[['unique_id']].merge(air_info, on = 'unique_id', how = 'left')
tr_te['air_genre_name'] = df['air_genre_name']
tr_te['air_area_name'] = df['air_area_name']
tr_te['latitude'] = df['latitude']
tr_te['longitude'] = df['longitude']

df = tr_te[['unique_id']].merge(hpg_info, on = 'unique_id', how = 'left')
tr_te['hpg_genre_name'] = df['hpg_genre_name']
tr_te['hpg_area_name'] = df['hpg_area_name']
#tr_te['hpg_latitude'] = df['latitude']
#tr_te['hpg_longitude'] = df['longitude']

print('Get date of reserve data')
air_res['visit_date'] = air_res['visit_datetime'].dt.normalize()
hpg_res['visit_date'] = hpg_res['visit_datetime'].dt.normalize()

air_res['reserve_date'] = air_res['reserve_datetime'].dt.normalize()
hpg_res['reserve_date'] = hpg_res['reserve_datetime'].dt.normalize()

print('Split train/test again')
train = tr_te[:ntrain].reset_index().drop('index', axis = 1)
test = tr_te[ntrain:].reset_index().drop('index', axis = 1)
del(tr_te)
gc.collect()

print('Create index for validation')
inTr = np.where((train['week_count'] >= 6) & (train['week_count'] <= 63))[0]
inTe = np.where(train['week_count'] > 63)[0]

print('Save data')
train.to_pickle('input/train.pkl')
test.to_pickle('input/test.pkl')
air_res.to_pickle('input/air_res.pkl')
hpg_res.to_pickle('input/hpg_res.pkl')
np.save('input/visitors.npy', train['visitors'].values)
np.save('input/id.npy', sample_sub['id'].values)
np.save('input/inTr.npy', inTr)
np.save('input/inTe.npy', inTe)


print('Done!!!')
