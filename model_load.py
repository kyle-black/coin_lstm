
import tensorflow as tf
from tensorflow import keras




import os

os.environ['TF_CCP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd

import tensorflow as tf
#from cryptowatch1 import BTCUSD
import cryptowatch1
from sklearn.preprocessing import StandardScaler

from datetime import datetime
from statistics import mean



def import_df(df_):

    df_['change'] = df_['close'].pct_change(periods =24)
    df_['change_price'] = df_['close'].shift(24) + (df_['change'] * df_['close'].shift(24))
    df_= df_.dropna()
   # df_ = df_[-48:]
    df =df_[:-24]
    predict_df =df_[-24:]

    #predict_time = predict_df['time']
    scaler = StandardScaler()


   
    plot_cols = ['time', 'change','change_price','open', 'high', 'low', 'close', 'vol_', 'volume', 'RSI', 'MACD',
       'macsignal', 'macdhist', 'ADX', 'Aroon', 'Trendmode', 'AD', 'ADOSC',
       'OBV', 'NATR', 'TRANGE','TWO_CROWS','THREE_BLACK_CROWS','THREE_INSIDE_UP_DOWN','CDL_3_LINE_STRIKE','CDL_3_OUTSIDE',
       'CDL_3_STARS_IN_SOUTH','CDL_3_WHITE_SOLDIERS','CDL_ABANDONED_BABY','CDL_ADVANCE_BLOCK','CDL_BELT_HOLD','CDL_BREAK_AWAY','CDL_CLOSING_MARUBOZU']


    plot_features =df[plot_cols]

    plot_features.index = df['time']

    df1 = plot_features


    df1= df1.astype(float)

    df1_time= df1.index


    df1.reset_index(drop=True, inplace=True)

    df1.describe().transpose()

   # BTCUSD =BTCUSD.loc[:23]

    

    column_indicies = {name: i for i, name in enumerate(df1.columns)}

    n =len(df1)
  
    train_df =df1[0:int(n*0.7)]
   # print(train_df)
    val_df = df1[int(n*0.7): int(n*0.9)]
   # print(val_df)
    test_df = df1[int(n*0.9):]
   # print(test_df)
    
    num_features =df1.shape[1]

    train_mean= train_df.mean()
    train_std = train_df.std()
    predict_mean = predict_df.mean()
    predict_std = predict_df.std()


    
    scaled_cols =['time', 'change','change_price','open', 'high', 'low', 'close', 'vol_', 'volume', 'RSI', 'MACD',
       'macsignal', 'macdhist', 'ADX', 'Aroon', 'Trendmode', 'AD', 'ADOSC',
       'OBV', 'NATR', 'TRANGE']

    cat_cols = ['TWO_CROWS','THREE_BLACK_CROWS','THREE_INSIDE_UP_DOWN','CDL_3_LINE_STRIKE','CDL_3_OUTSIDE',
       'CDL_3_STARS_IN_SOUTH','CDL_3_WHITE_SOLDIERS','CDL_ABANDONED_BABY','CDL_ADVANCE_BLOCK','CDL_BELT_HOLD','CDL_BREAK_AWAY','CDL_CLOSING_MARUBOZU']


    train_df_scaled =pd.DataFrame(scaler.fit_transform(train_df[scaled_cols]), columns =scaled_cols )
    train_df_cat = train_df[cat_cols]
    train_df_cat.reset_index(inplace=True)
    train_df = train_df_scaled.merge(train_df_cat, left_index=True, right_index=True)
    

    val_df_scaled =pd.DataFrame(scaler.fit_transform(val_df[scaled_cols]), columns =scaled_cols )
    val_df_cat = val_df[cat_cols]
    val_df_cat.reset_index(inplace=True)
    val_df = val_df_scaled.merge(val_df_cat, left_index=True, right_index=True)
   
    
    test_df_scaled =pd.DataFrame(scaler.fit_transform(test_df[scaled_cols]), columns =scaled_cols )
    test_df_cat = test_df[cat_cols]
    test_df_cat.reset_index(inplace=True)
  
    test_df = test_df_scaled.merge(test_df_cat, left_index=True, right_index=True)

    predict_df_scaled =pd.DataFrame(scaler.fit_transform(predict_df[scaled_cols]), columns =scaled_cols )
    predict_df_cat = predict_df[cat_cols]
    predict_df_cat.reset_index(inplace=True)
   
    predict_df = predict_df_scaled.merge(predict_df_cat, left_index=True, right_index=True)
   
    return train_df, val_df, test_df, predict_df, scaler , predict_mean, predict_std, train_mean, train_std


def create_time(df_, interval):
    strt = int(df_['new_time'].iloc[0])
    strt =datetime.utcfromtimestamp(strt).strftime('%Y-%m-%d %H:%M:%S')
    time_ = (pd.Series( data = pd.date_range(start=strt, periods=48, freq=interval)))
    return time_


def make_predict(coin,model,interval):
    
    train_df, val_df, test_df, predict_df, scaler, predict_mean,predict_std, train_mean,train_std = import_df(coin)
    
    
    last_price = predict_df['close'].iloc[-1]
    
    
    #print('inputs:',predict_df)
    coin_p =pd.DataFrame((predict_df), columns =predict_df.columns )
    #print('coin p:', np.array(coin_p))
    x= model.predict(np.array([coin_p,]))
    x =x[0][0:24,0]
  
    coin_p['time'] 

    
    prediction = x * predict_std['close'] + predict_mean['close']
    last_price = last_price * train_std['close'] +train_mean['close']
    coin_p = coin_p * predict_std + predict_mean
    
    predict_df['new_time'] = predict_df['time'] #* predict_std['time'] + predict_mean['time']
    #predict_df = predict_df['new_time']
    time_ =create_time(predict_df, interval)
   
    
    return  prediction, coin_p, last_price, time_
    
    





#################################
############# Predictions #######
####### BTCUSD ##################
def get_coin(pair, time):
    coin_ =cryptowatch1.get_ohlc(pair,time)
    model= tf.keras.models.load_model(f'models/{pair}/{pair}_{time}_LSTM.h5')
    coin_predict, coin_Prices,coin_LAST_PRICE, coin_Time = make_predict(coin_, model, time)
    time_list= ['5m', '15m','30m', '1h']
    for time_ in time_list:
        coin_24 = cryptowatch1.get_ohlc(pair,time_)
        coin_24['close'].to_csv(f'output_csv/{pair}/{pair}_{time_}_24h.csv', mode='w' )

    x=np.ravel(coin_predict)
    coin_predict = x[:24]
    print(coin_predict)
    coin_predict = coin_predict.tolist()
    coin_PRICES = coin_Prices['close'].tolist()
    print('coin_Prices:', coin_PRICES)
    predict_mean = round(mean(coin_predict),2)
    coin_PRICES.extend(coin_predict)
    outlist= coin_PRICES
    print(len(outlist))
    #print(outlist)
    coin_df = pd.DataFrame(coin_Time, columns=['new_time'])

   # print(coin_predict)
    coin_df['price'] = outlist 
    print('coin_df:',coin_df)
    #coin_df.to_csv(f'output_csv/BTCUSD/{pair}_{time}_LSTM.csv' )
    coin_df.to_csv(f'output_csv/{pair}/{pair}_{time}_LSTM.csv', mode='w' )

    return predict_mean



#################################


asset_list=['ADAUSD','BTCUSD','DOGEUSD', 'ETHUSD',  'HFTUSD','LTCUSD', 'MATICUSD','SHIBUSD', 'SOLUSD', 'XRPUSD']

for i in asset_list:
    #try:
    get_coin(i,'5m')
    get_coin(i,'15m')
    get_coin(i,'30m')
    get_coin(i,'1h')
   # except: print('not available')



