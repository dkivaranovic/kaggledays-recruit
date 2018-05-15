
#### libraries
import numpy as np
import pandas as pd
import lightgbm as lgb
import load
import gc, os

#### root name of files
root_name = 'LGB_01'

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
        train = load.TRAIN(feat)
        ytrain = np.load('input/visitors.npy')

        inTr = np.concatenate((np.load('input/inTr.npy'), np.load('input/inTe.npy')))
        dtr = lgb.Dataset(train.loc[inTr,feat], label = np.log(ytrain[inTr] + 1))
        del(train)
        gc.collect()

        test = load.TEST(feat)

        print('Start training')

        #### lgb parameters
        params = {
            'boosting_type':     'gbdt',
            'objective':         'regression',
            'metric':            'rmse',
            'max_depth':         16,
            'learning_rate':     0.0025,
            'num_leaves':        600, 
            'feature_fraction':  0.5, 
            'bagging_fraction':  0.9, 
            'bagging_freq':      30,
            'verbose':           -1
        }

        params['feature_fraction_seed'] = d+1
        params['bagging_seed'] = (d+1)**2

        #### number of rounds
        nrounds = np.int(np.round(np.load('nrounds/' + file_name + '.npy')*1.05))

        gbm = lgb.train(params,
                        dtr,
                        num_boost_round = nrounds)
        
        print('Prediction')
        pred = gbm.predict(test.loc[:,feat])
        np.save('predictions/test/' + file_name, pred)

        del(gbm, dtr, test, ytrain)
        gc.collect()




