
#### libraries
import numpy as np
import pandas as pd

#### root name of files
root_name = 'KERAS_01'

#### test data
test = pd.read_pickle('input/test.pkl')

#### day
day = test['visit_date'].dt.dayofyear - test['visit_date'].dt.dayofyear.min()

#### predictions
pred = np.zeros(test.shape[0])

#### rounds and score
for d in range(39):
    file_name = root_name + '_day' + np.str(d)  + '.npy'
    s = np.load('score/' + file_name)
    p = np.load('predictions/test/' + file_name)
    pred[day == d] = p[day == d]
    print('Day {0} - RMSE: {1}' .format(d, np.round(s, 6)))

#### transform predictions
pred = np.exp(pred) - 1

#### test ids
id_test = np.load('input/id.npy')

#### create submission
df = pd.DataFrame({'id': id_test, 'visitors': pred}) 
df.to_csv('submission/' + root_name + '.csv', index = False)

