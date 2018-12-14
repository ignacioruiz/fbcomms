## **************************************************** ##
##                                                      ##
##    fbcomms.py                                        ##
##                 submodules used by main notebook     ##
##    Use: import fbcomms as fbc                        ##
##                                                      ##
##    Requires Python 3.6, TF 1.12 and Keras 2.0        ##
##                                                      ##
## **************************************************** ##

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import time

### Metric Calculation function: takes three parameters,
# 2 arrays of predicted and real values
# printing out results is ON by default.

def calculate_metrics(y_hat, y_real, printout=True):
    
    metric_mae = mean_absolute_error(y_real, y_hat)
    metric_mse = mean_squared_error(y_real, y_hat)
    
    if printout:
        print("Direct equality fit:", np.sum(y_real == y_hat) / len(y_real))  
        
        # since y_real domain is integer, we need to transform y_hat
        print("Ceiling based fit: ", np.sum(y_real == np.ceil(y_hat)) / len(y_real))  
        print("Floor based fit: ", np.sum(y_real == np.floor(y_hat)) / len(y_real))  
        print("Round based fit: ", np.sum(y_real == np.round(y_hat)) / len(y_real))  
        print("MAE: ", (metric_mae))
        print("MSE: ", (metric_mse))

    return metric_mae, metric_mse
  
### This function will perform a Tuned Random Forest Regressor.
#  Takes the full dataset, a list of features we want to keep from the
#  dataset and an optional list of features we want to subtract.
#  Finally the name for label or target column/feature.
  
def TryRFR(full_dataset, list_features, droplist=None, target="Target"):
  # Capture initial time
  starting_t = time.time()
  
  # split the dataset
  train, test = train_test_split(full_dataset[list_features], test_size=0.20, random_state=42)
  
  # If there are additional features to drop from analysis
  # we append the label in preparation for training
  if droplist==None:
    finaldroplist=[target]
  else:
    droplist.append(target)
    # print(droplist)
    finaldroplist = droplist
    # print(finaldroplist)
    
  X_train = train.drop(finaldroplist, axis=1)
  y_train = train[target]
  
  # Here our optimized RFR call
  #rnd_reg = RandomForestRegressor(n_estimators=5000, min_samples_split=0.0015, n_jobs=-1, random_state=42)
  rnd_reg = RandomForestRegressor(n_estimators=500, min_samples_split=0.0015, n_jobs=-1, random_state=42)
  rnd_reg.fit(X_train, y_train)
  
  # Now let's prepare our test sample, and evaluate how
  # our model fits the actual data
  X_test = test.drop(finaldroplist, axis=1)
  y_test = test[target]

  y_pred_rf = rnd_reg.predict(X_test)
  
  # We calculate Median Absolute Error and Medium Square Error
  mae, mse = calculate_metrics(y_pred_rf, y_test)
  
  # Timing ends, we get delta and report it out
  ending_t = time.time()
  # seconds elapsed
  elapsed_time = ending_t - starting_t
  print("RFR running time (s): ", elapsed_time)
     
  return mae, mse, elapsed_time

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout


# This function will accept train/test datasets, optimizer and a list of 
# integer values that will define each layer's components. In case no layer 
# definition is passed it will construct the model with 2500,200.  The last 
# neuron cell is always 1 and activation_f linear 
def myRegressorMLP_mhl(full_dataset, list_features, p_optimizer, layers=None, p_activation='relu', p_dropout=0.10, p_epochs=1000, p_batch_size=5000):
    # Capture initial time
  starting_t = time.time()
  
  # standarizataion
  num_pipeline = Pipeline([('std_scaler', StandardScaler())])

  # split the dataset
  train, test = train_test_split(full_dataset, test_size=0.20, random_state=42)
  
  X_train = train.drop("Target", axis=1)
  y_train = train["Target"]
  
  X_train_std = num_pipeline.fit_transform(X_train)

  if layers==None:
    layers=[2500,200]        ## So far the optimal found
  
  model = Sequential()
  
  first=True
  input_features = X_train.shape[1]
  
  # TODO: seems i'm missing to include the bias
  
  for neurons in layers:
    if first==True:
      # model.add(Dense(neurons,input_dim=input_features, kernel_initializer='normal', use_bias=True, bias_initializer='ones'))
      model.add(Dense(neurons,input_dim=input_features, kernel_initializer='normal', activation=p_activation))
      first = False
    else:
      # model.add(Dense(neurons,use_bias=True, bias_initializer='ones'))
      model.add(Dense(neurons, kernel_initializer='normal', activation=p_activation))
    # model.add(Activation(activation=p_activation))    
    model.add(BatchNormalization())
    model.add(Dropout(p_dropout))
     
  model.add(Dense(1, kernel_initializer='normal'))
  # model.add(Activation(activation='tanh'))

  model.compile(loss='mean_squared_error', optimizer=p_optimizer,  metrics=['mean_squared_error'])

  # %time 
  model.fit(X_train_std, y_train, epochs=p_epochs, batch_size=p_batch_size, verbose=0)
  
  X_test = test.drop("Target", axis=1)
  y_test = test["Target"]
  
  X_test_std = num_pipeline.fit_transform(X_test)
  y_pred_rf = model.predict(X_test_std)
  
  # We calculate Median Absolute Error and Medium Square Error
  # mae, mse = calculate_metrics(y_pred_rf.flatten(), y_test, printout=False)
  mae, mse = calculate_metrics(y_pred_rf.flatten(), y_test)
  
  # Timing ends, we get delta and report it out
  ending_t = time.time()
  # seconds elapsed
  elapsed_time = ending_t - starting_t
  print("MLP running time (s): ", elapsed_time)

  return mae, mse, elapsed_time