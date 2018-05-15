# Solution for Recruit Restaurant Visitor Forecasting Competition

This is a complete solution for the Recruit Restaurant Visitor Forecasting Competition that will be presented at [Kaggle Days](https://www.kaggledays.com/) in Warsaw on May 19th, 2018. The ideas behind it are quite general and can be used for any time-series competition. Actually the solution here is almost a one-to-one copy of my solution of the Masters Caesars competition which my team won. The submission scores 0.505 on private LB. This would have been the 3rd place.

To run the solution, download the git repository and run `bash setup_directory.sh` first. This generates all the folders that you will need. Download the train/test data from the [competition homepage](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting) and put the data in *input/*. From [Hunter McGushion's wheater dataset](https://www.kaggle.com/huntermcgushion/rrv-weather-data) download the file *air_store_info_with_nearest_active_station.csv* and put it in *weather_data/*. Also download *1-1-16_5-31-17_Weather.zip* from there and put it in *weather_data/stations*.

Ok, you now have all the data you need. You also have to have python installed with the following packages: **numpy**, **pandas**, **scipy**, **scikit-learn**, **lightgbm** and **keras**. 

Run *preprocessing.py* first. This does just some basic preprocessing of the data. Then, generate all features by running *feat_gen_standard.py*, *feat_gen_visitors.py*, *feat_gen_res_visitors.py*, *feat_gen_res_visitors_type2.py* and *feat_gen_weather.py* (the order of the scripts does not matter).

Run the following scripts in the given order to obtain Lightgbm predictions for the validations set, for the test set and the submission file: *LGB_01_CV.py*, *LGB_01_LB.py* and *LGB_01_SUB.py*.

Run the following scripts in the given order to obtain Keras predictions for the validations set, for the test set and the submission file: *KERAS_01_CV.py*, *KERAS_01_LB.py* and *KERAS_01_SUB.py*.

Run *weighted_average_SUB.py* for a weighted average of the Lightgbm and Keras solution.

Performance of the models:
| Model            | Public LB | Private LB |
| ---------------- |-----------| ---------- |
| Lightgbm         | 0.468     | 0.509.     |
| Keras            | 0.474     | 0.513      |
| weighted average | 0.468     | 0.505      |

Have fun with it!
