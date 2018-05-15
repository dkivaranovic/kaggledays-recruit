
#### libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

def metric(x):
	p = x*pred1 + (1-x)*pred2
	error = np.sqrt(mean_squared_error(np.log(yte+1), p))
	return(error)

#### visitors validation
inTe = np.load('input/inTe.npy')
yte = np.load('input/visitors.npy')[inTe]

#### find optimal weights for each day
opt_val = np.zeros(39)

for d in range(39):
	pred1 = np.load('predictions/train/LGB_01_day' + np.str(d) + '.npy')
	pred2 = np.load('predictions/train/KERAS_01_day' + np.str(d) + '.npy')
	res = minimize(metric, x0 = 0.5, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
	opt_val[d] = res.x

#### take median value
x = np.median(opt_val)

#### create submission
df_lgb = pd.read_csv('submission/LGB_01.csv')
df_keras = pd.read_csv('submission/KERAS_01.csv')

df = df_lgb.copy()
df['visitors'] = np.exp(x*np.log(df_lgb['visitors']+1) + (1-x)*np.log(df_keras['visitors']+1))  - 1
df.to_csv('submission/weighted_ave_LGB_KERAS.csv', index = False)

