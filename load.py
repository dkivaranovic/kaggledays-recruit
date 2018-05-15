
#### libraries
import pandas as pd
import numpy as np

def TRAIN(feat):
    df = pd.DataFrame()
    for f in feat:
            df[f] = np.load('features/train/' + f + '.npy')
    return(df)

def TEST(feat):
    df = pd.DataFrame()
    for f in feat:
            df[f] = np.load('features/test/' + f + '.npy')
    return(df)


