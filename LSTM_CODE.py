import os
import math
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.preprocessing import *
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import *
from tensorflow.keras.utils import *

def csvloader(filename):
    return pd.read_csv(filename)

def make_dataset(data, label, window_size = 120):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i : i + window_size]))
        label_list.append(np.array(label.iloc[i + window_size]))
    return np.array(feature_list), np.array(label_list)

def ModelLayer(df, predictLabel, b_size): 
    df_scaled = pd.DataFrame(df[predictLabel])
    df_scaled.columns = [predictLabel]
    TEST_SIZE = b_size * 2
    
    train = df_scaled[ : -TEST_SIZE]
    train_feature = train[[predictLabel]]
    train_label = train[predictLabel]
    train_feature, train_label = make_dataset(train_feature, train_label, b_size)
    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size = b_size / len(df) * 2)
    
    model = Sequential()
    model.add(LSTM(16, 
                   input_shape=(train_feature.shape[1], train_feature.shape[2]), 
                   activation='relu', 
                   return_sequences=False))
    
    # IF YOU WANT TO ADD LAYER
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error', 
                  optimizer = 'adam')
    
    
    early_stop = EarlyStopping(monitor='val_loss', 
                               patience = 5)
    
    # IF YOU WANT TO SAVE MODEL WEIGHTS
    model_path = 'model'
    filename = os.path.join(model_path, 'MODELWEIGHT.h5')
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    
    history = model.fit(x_train, 
                        y_train, 
                        epochs = 100, 
                        batch_size = b_size,
                        validation_data = (x_valid, y_valid), 
                        callbacks = [early_stop, checkpoint])
    
    #IF YOU WANT TO SAVE MODEL
    model.save("MODELDATA.h5")
    
    return model
    
def test_features(df, predictLabel, b_size):
    df_scaled = pd.DataFrame(df[predictLabel])
    df_scaled.columns = [predictLabel]
    TEST_SIZE = b_size * 2
    test = df_scaled[-TEST_SIZE : ]
    feature_cols = [predictLabel]
    test_feature = test[feature_cols]
    test_label = test[predictLabel]
    test_feature, test_label = make_dataset(test_feature, test_label, b_size)
    return test_feature

def newFeatureCreator_FST(df, model, predictLabel, b_size):
    model_cases = test_features(df, predictLabel, b_size)
    model_pred = model.predict(model_cases)
    arr = [abs(round(model_pred[-1][0], 4))]
    return arr

def newFeatureCreator(df, model, feature):
    model_cases = feature
    model_pred = model.predict(model_cases)
    arr = [abs(round(model_pred[-1][0], 4))]
    return arr

def rotationFeature(feature):
    temp = []
    for i in range(0, len(feature)):
        if i == 0:
            temp = feature[i]
        elif i < len(feature) - 1:
            feature[i] = feature[i + 1]
        else:
            feature[i] = temp
    return feature

def processor(num, data, original_feat, model, one, predictLabel, b_size):
    lst = []
    feat = original_feat
    lst.append(one)
    for i in range(0, num):
        arr = newFeatureCreator(data, model, feat)
        newfeature = rotationFeature(feat[-1])
        newfeature[-1] = arr
        feat = test_features(data, predictLabel, b_size)
        feat[-1] = newfeature
        lst.append(arr)
        # print(arr)
    return lst

def LSTMMODEL(df, predictLabel, batch_size, days):
    model = ModelLayer(df, predictLabel, batch_size)
    arr = newFeatureCreator_FST(df, model, predictLabel, batch_size)
    feature = rotationFeature(test_features(df, predictLabel, batch_size)[-1])
    feature[-1] = arr
    newfeature = test_features(df, predictLabel, batch_size)
    newfeature[-1] = feature
    lst = processor(days, df, newfeature, model, arr, predictLabel, batch_size)
    return lst

def main():
    df = csvloader("NEWCH.csv")
    
    # NEXT FUTURE n DATA PREDICTION
    # IF DAYS is 7, IT MEANS THAT 7 OUTPUT WILL COME OUT
    days = 7
    label = "middle"
    batch_size = 120
    arr = LSTMMODEL(df, label, batch_size, days)
    print("NEXT 7 DATA ARE:", arr)
    
    
if __name__ == "__main__":
    main()