
#### libraries
import numpy as np
np.random.seed(111)

import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.special import erfinv
import gc, os

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Flatten, BatchNormalization, concatenate
from keras.optimizers import Adam

#### root name of files
root_name = 'KERAS_01'

for d in range(39):

    file_name = root_name + '_day' + np.str(d)
    
    if not os.path.isfile('predictions/test/' + file_name + '.npy'):

        print(file_name)

        f0 = [  #### standard features
                'unique_id', 'year', 'month', 'week', 'day', 'weekday', 'holiday', 'holiday_2', 'holiday_lag-2', 'holiday_lag-1', 'holiday_lag1', 
                'holiday_lag2', 'golden_week', 'air_genre_name', 'air_area_name', 'latitude', 'longitude', 'hpg_genre_name', 'hpg_area_name',
        ]

        f1 = [  #### visitors features
                ## day
                'visitors_day_lag1', 'visitors_day_lag2', 'visitors_day_lag3', 'visitors_day_lag4', 'visitors_day_lag5', 'visitors_day_lag6', 'visitors_day_lag7', 
                'visitors_day_lag8', 'visitors_day_lag9', 'visitors_day_lag10', 'visitors_day_lag11', 'visitors_day_lag12', 'visitors_day_lag13', 'visitors_day_lag14',
                ## weekday
                'visitors_weekday_lag1', 'visitors_weekday_lag2', 'visitors_weekday_lag3', 'visitors_weekday_lag4', 'visitors_weekday_lag5', 'visitors_weekday_lag6', 'visitors_weekday_lag7', 
                'visitors_weekday_lag8', 'visitors_weekday_lag9', 'visitors_weekday_lag10', 'visitors_weekday_lag11', 'visitors_weekday_lag12', 'visitors_weekday_lag13',
                ## mean week
                'mean_visitors_week_lag1', 'mean_visitors_week_lag2', 'mean_visitors_week_lag3', 'mean_visitors_week_lag4', 'mean_visitors_week_lag5', 
                'mean_visitors_week_lag6', 'mean_visitors_week_lag7', 'mean_visitors_week_lag8', 'mean_visitors_week_lag9', 'mean_visitors_week_lag10', 
                'mean_visitors_week_lag11', 'mean_visitors_week_lag12', 'mean_visitors_week_lag13', 'mean_visitors_week_lag14', 'mean_visitors_week_lag15', 
                'mean_visitors_week_lag16', 'mean_visitors_week_lag17', 'mean_visitors_week_lag18', 'mean_visitors_week_lag19', 'mean_visitors_week_lag20', 
                'mean_visitors_week_lag21', 'mean_visitors_week_lag22', 'mean_visitors_week_lag23', 'mean_visitors_week_lag24',
                ## mean month (4 weeks)
                'mean_visitors_month_lag1', 'mean_visitors_month_lag2', 'mean_visitors_month_lag3', 'mean_visitors_month_lag4', 'mean_visitors_month_lag5', 
                'mean_visitors_month_lag6', 'mean_visitors_month_lag7', 'mean_visitors_month_lag8', 'mean_visitors_month_lag9', 'mean_visitors_month_lag10', 
                'mean_visitors_month_lag11', 'mean_visitors_month_lag12',
                ## mean 4 weekdays
                'mean_visitors_4weekdays_lag1', 'mean_visitors_4weekdays_lag2', 'mean_visitors_4weekdays_lag3', 'mean_visitors_4weekdays_lag4', 
                'mean_visitors_4weekdays_lag5', 'mean_visitors_4weekdays_lag6', 'mean_visitors_4weekdays_lag7', 'mean_visitors_4weekdays_lag8', 
                'mean_visitors_4weekdays_lag9', 'mean_visitors_4weekdays_lag10', 'mean_visitors_4weekdays_lag11', 'mean_visitors_4weekdays_lag12',
        ]
        f1 = [f + '_day' + np.str(d) for f in f1]


        f2 = [  #### air reservation features
                ## mean week
                'mean_res_visitors_week_lag1', 'mean_res_visitors_week_lag2', 'mean_res_visitors_week_lag3', 'mean_res_visitors_week_lag4', 'mean_res_visitors_week_lag5', 
                'mean_res_visitors_week_lag6', 'mean_res_visitors_week_lag7', 'mean_res_visitors_week_lag8', 'mean_res_visitors_week_lag9', 'mean_res_visitors_week_lag10', 
                'mean_res_visitors_week_lag11', 'mean_res_visitors_week_lag12',
                ## mean month (4 weeks)
                'mean_res_visitors_month_lag1', 'mean_res_visitors_month_lag2', 'mean_res_visitors_month_lag3', 'mean_res_visitors_month_lag4', 'mean_res_visitors_month_lag5', 
                'mean_res_visitors_month_lag6',
                ## mean 4 weekdays
                'mean_res_visitors_4weekdays_lag1', 'mean_res_visitors_4weekdays_lag2', 'mean_res_visitors_4weekdays_lag3', 'mean_res_visitors_4weekdays_lag4', 
                'mean_res_visitors_4weekdays_lag5', 'mean_res_visitors_4weekdays_lag6',

        ]  
        f2 = [f + '_day' + np.str(d) for f in f2]

        f3 = [  #### air reservation features (Type 2)
                ## day
                'sum_res_visitors_t2_day_lag0', 'sum_res_visitors_t2_day_lag1', 'sum_res_visitors_t2_day_lag2', 
                'count_res_visitors_t2_day_lag0', 'count_res_visitors_t2_day_lag1', 'count_res_visitors_t2_day_lag2', 
                ## weekday
                'sum_res_visitors_t2_weekday_lag0', 'sum_res_visitors_t2_weekday_lag1', 'sum_res_visitors_t2_weekday_lag2', 'sum_res_visitors_t2_weekday_lag3', 
                'count_res_visitors_t2_weekday_lag0', 'count_res_visitors_t2_weekday_lag1', 'count_res_visitors_t2_weekday_lag2', 'count_res_visitors_t2_weekday_lag3', 
                ## week
                'sum_res_visitors_t2_week_lag0', 'sum_res_visitors_t2_week_lag1', 'sum_res_visitors_t2_week_lag2', 
                'count_res_visitors_t2_week_lag0', 'count_res_visitors_t2_week_lag1', 'count_res_visitors_t2_week_lag2', 
                ## month (4 weeks)
                'sum_res_visitors_t2_month_lag0', 'sum_res_visitors_t2_month_lag1', 'sum_res_visitors_t2_month_lag2', 
                'count_res_visitors_t2_month_lag0', 'count_res_visitors_t2_month_lag1', 'count_res_visitors_t2_month_lag2',
                ## 4 weekdays
                'sum_res_visitors_t2_4weekdays_lag0', 'sum_res_visitors_t2_4weekdays_lag1', 'sum_res_visitors_t2_4weekdays_lag2', 'sum_res_visitors_t2_4weekdays_lag3', 
                'count_res_visitors_t2_4weekdays_lag0', 'count_res_visitors_t2_4weekdays_lag1', 'count_res_visitors_t2_4weekdays_lag2', 'count_res_visitors_t2_4weekdays_lag3', 
        ] 

        f3 = [f + '_day' + np.str(d) for f in f3]

        f4 = [  #### weather data
                'avg_temperature_lag0', 'precipitation_lag0', 'hours_sunlight_lag0', 'solar_radiation_lag0', 'total_snowfall_lag0', 'avg_wind_speed_lag0', 'avg_vapor_pressure_lag0', 'avg_local_pressure_lag0', 'avg_humidity_lag0', 
                'avg_sea_pressure_lag0', 'cloud_cover_lag0',
                'avg_temperature_lag1', 'precipitation_lag1', 'hours_sunlight_lag1', 'solar_radiation_lag1', 'total_snowfall_lag1', 'avg_wind_speed_lag1', 'avg_vapor_pressure_lag1', 'avg_local_pressure_lag1', 'avg_humidity_lag1', 
                'avg_sea_pressure_lag1', 'cloud_cover_lag1',
                'avg_temperature_lag2', 'precipitation_lag2', 'hours_sunlight_lag2', 'solar_radiation_lag2', 'total_snowfall_lag2', 'avg_wind_speed_lag2', 'avg_vapor_pressure_lag2', 'avg_local_pressure_lag2', 'avg_humidity_lag2', 
                'avg_sea_pressure_lag2', 'cloud_cover_lag2',
                'avg_temperature_lag-1', 'precipitation_lag-1', 'hours_sunlight_lag-1', 'solar_radiation_lag-1', 'total_snowfall_lag-1', 'avg_wind_speed_lag-1', 'avg_vapor_pressure_lag-1', 'avg_local_pressure_lag-1', 'avg_humidity_lag-1', 
                'avg_sea_pressure_lag-1', 'cloud_cover_lag-1',
                'avg_temperature_lag-2', 'precipitation_lag-2', 'hours_sunlight_lag-2', 'solar_radiation_lag-2', 'total_snowfall_lag-2', 'avg_wind_speed_lag-2', 'avg_vapor_pressure_lag-2', 'avg_local_pressure_lag-2', 'avg_humidity_lag-2', 
                'avg_sea_pressure_lag-2', 'cloud_cover_lag-2',
        ]

        if d == 37:
            f0 = [f for f in f0 if '-2' not in f]
            f4 = [f for f in f4 if '-2' not in f]

        if d == 38:
            f0 = [f for f in f0 if '-2' not in f]
            f0 = [f for f in f0 if '-1' not in f]
            f4 = [f for f in f4 if '-2' not in f]
            f4 = [f for f in f4 if '-1' not in f]

        feat = np.concatenate((f0,f1,f2,f3,f4))

        print('Load data')
        feat_emb = ['unique_id', 'month', 'week', 'day', 'weekday', 'air_genre_name', 'air_area_name', 'hpg_genre_name', 'hpg_area_name',]

        ytrain = np.load('input/visitors.npy')
        ntrain = len(ytrain)
        inTr = np.concatenate((np.load('input/inTr.npy'),np.load('input/inTe.npy')))
        ytr = np.log(ytrain[inTr] + 1)

        tr = []
        te = []
        for f in feat:
            x = np.concatenate((np.load('features/train/' + f + '.npy'), np.load('features/test/' + f + '.npy')))
            x[np.isnan(x)] = 0
            if f not in feat_emb:
                if np.any(x<0):
                    x = (x-x.mean())/x.std()   
                    tr.append(x[inTr])
                    te.append(x[ntrain:]) 
                else:
                    x = np.log(x+1)
                    x = (x-x.mean())/x.std()   
                    tr.append(x[inTr])
                    te.append(x[ntrain:])        

        tr = np.column_stack(tr)
        te = np.column_stack(te)

        print('Shape tr: {0}' .format(tr.shape))
        print('Shape te: {0}' .format(te.shape))

        #### categorical variables (with more than 2 categories)
        unique_id = np.concatenate((np.load('features/train/unique_id.npy'), np.load('features/test/unique_id.npy')))
        month = np.concatenate((np.load('features/train/month.npy'), np.load('features/test/month.npy')))
        week = np.concatenate((np.load('features/train/week.npy'), np.load('features/test/week.npy')))
        day = np.concatenate((np.load('features/train/day.npy'), np.load('features/test/day.npy')))
        weekday = np.concatenate((np.load('features/train/weekday.npy'), np.load('features/test/weekday.npy')))
        air_genre_name = np.concatenate((np.load('features/train/air_genre_name.npy'), np.load('features/test/air_genre_name.npy')))
        air_area_name = np.concatenate((np.load('features/train/air_area_name.npy'), np.load('features/test/air_area_name.npy')))

        n_unique_id = len(np.unique(unique_id)) + 1
        n_month = len(np.unique(month)) + 1
        n_week = len(np.unique(week)) + 1
        n_day = len(np.unique(day)) + 1
        n_weekday = len(np.unique(weekday)) + 1
        n_air_genre_name = len(np.unique(air_genre_name)) + 1
        n_air_area_name = len(np.unique(air_area_name)) + 1

        def nn_model():
            inp_unique_id = Input(shape=[1])
            inp_month = Input(shape=[1])
            inp_week = Input(shape=[1])
            inp_day = Input(shape=[1])
            inp_weekday = Input(shape=[1])
            inp_air_genre_name = Input(shape=[1])
            inp_air_area_name = Input(shape=[1])
            inp_tr = Input(shape=[tr.shape[1]])
            emb_unique_id = Embedding(n_unique_id, 32)(inp_unique_id)
            emb_month = Embedding(n_month, 6)(inp_month)
            emb_week = Embedding(n_week, 10)(inp_week)
            emb_day = Embedding(n_day, 10)(inp_day)
            emb_weekday = Embedding(n_weekday, 3)(inp_weekday)
            emb_air_genre_name = Embedding(n_air_genre_name, 6)(inp_air_genre_name)
            emb_air_area_name = Embedding(n_air_area_name, 10)(inp_air_area_name)
            main = concatenate([Flatten()(emb_unique_id), 
                                Flatten()(emb_month), 
                                Flatten()(emb_week),
                                Flatten()(emb_day),
                                Flatten()(emb_weekday), 
                                Flatten()(emb_air_genre_name),
                                Flatten()(emb_air_area_name),
                                inp_tr])
            x = Dense(100, activation = 'relu')(main)
            x = BatchNormalization()(x)
            x = Dropout(rate = 0.2)(x)
            x = Dense(100, activation = 'relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(rate = 0.2)(x)
            predictions = Dense(1, activation = 'relu')(x)
            model = Model(inputs = [inp_unique_id, inp_month, inp_week, inp_day, inp_weekday, inp_air_genre_name, inp_air_area_name,inp_tr], outputs = predictions)
            model.compile(loss = 'mse', optimizer = Adam(lr = 0.0005, decay = 0.00005))
            return(model)

        batch_size = 128
        nbag = 10
        pred = np.zeros(te.shape[0])

        print('Start training')
        for i in range(nbag):
            print('Bag {0}' .format(i))
            model = nn_model()
            model.fit([unique_id[inTr], month[inTr], week[inTr], day[inTr], weekday[inTr], air_genre_name[inTr], air_area_name[inTr], tr], ytr, 
                      batch_size = batch_size,
                      epochs = 25)
            pred += model.predict([unique_id[ntrain:], month[ntrain:], week[ntrain:], day[ntrain:], weekday[ntrain:], air_genre_name[ntrain:], air_area_name[ntrain:], te])[:,0]

        print('Save predictions')
        pred /= nbag
        np.save('predictions/test/' + file_name, pred)
        del(tr, te)
        gc.collect()

print('Done!!!')


