
#### libraries
import numpy as np
import pandas as pd
import gc, os

print('Read data')
train = pd.read_pickle('input/train.pkl')
test = pd.read_pickle('input/test.pkl')

nearest_station = pd.read_csv('weather_data/air_store_info_with_nearest_active_station.csv')
air_info = pd.read_csv('input/air_store_info.csv')
air_info = air_info.merge(nearest_station[['air_store_id', 'station_id']], on = 'air_store_id', how = 'left')
del(nearest_station)

print('Concat train/test set')
ntrain = train.shape[0]
tr_te = pd.concat([train, test]).reset_index().drop('index', axis = 1)
del(train, test)
gc.collect()

print('Append weather data')
df = pd.DataFrame({})
for i in range(air_info.shape[0]):
	air_id = air_info['air_store_id'][i]
	station_id = air_info['station_id'][i]
	d = pd.read_csv('weather_data/stations/' + station_id + '.csv', parse_dates = ['calendar_date'])
	d['air_store_id'] = air_id
	df = pd.concat([df, d])

df = df.rename(index = str, columns = {'calendar_date': 'visit_date'})
tr_te = tr_te[['unique_id', 'air_store_id', 'visit_date']].merge(df, on = ['air_store_id', 'visit_date'], how = 'left')

print('Save features')
feat = ['avg_temperature', 'high_temperature', 'low_temperature', 'precipitation', 'hours_sunlight', 'solar_radiation', 'deepest_snowfall', 'total_snowfall', 
		'avg_wind_speed', 'avg_vapor_pressure', 'avg_local_pressure', 'avg_humidity', 'avg_sea_pressure', 'cloud_cover']

for lag in range(-2, 3):
	for f in feat:
		feat_name = f + '_lag' + np.str(lag)
		if not os.path.isfile('features/test/' + feat_name + '.npy'):
			print(feat_name)
			df = pd.DataFrame({'unique_id': tr_te['unique_id'], 'visit_date': tr_te['visit_date'] + pd.DateOffset(lag), feat_name: tr_te[f]})
			df = tr_te[['unique_id','visit_date']].merge(df, on = ['unique_id','visit_date'], how = 'left')
			x = df[feat_name].values
			np.save('features/train/' + feat_name, x[:ntrain])
			np.save('features/test/' + feat_name, x[ntrain:])


print('Done!!!')


