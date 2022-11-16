# CoinProphet

## CoinProphet uses a LSTM Neural Network to estimate the best price point for crypto currency closing prices for the next 24 periods.

## Predicted Prices are the Neural Network's best estimate for prices in the future. CoinProphet doesn't try to predict the variance of prices as much as it gives it's best estimation of where a price could be in the future.

 ## The neural network takes 34 seperate parameters and projects the next 24 periods. (5 min closes, 15 min closes, 30 min closes, 1hr closes)

 ## Models are trained on individual coins and individual time intervals. For example, ETHUSD at 5m is not mixed with ETHUSD at 30m.