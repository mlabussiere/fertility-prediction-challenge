
# -*- coding: utf-8 -*-
"""
Train pipeline
Created on Fri May 24 14:42:08 2024
@author: Bob & Bobette
package environment: py_uptodate (Python 3.12.3)
NOTE: this code documents all the training steps, but it is not reproducible due to lack of time, but we are very happy to provide a reproducible train function if needed.
"""

import os
import sys
# import FCT_create_df_gru_rnn as gr
from utility_Train import *
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

__file__ = 'training.py'
absolute_path = os.path.abspath(os.path.dirname(__file__))

path_data = os.path.join(absolute_path, 'Data')
path_output =  os.path.join(absolute_path, 'Outputs')
path_output_train =  os.path.join(absolute_path, 'Outputs', 'train')

#######################
# Load train datasets # (pickles compatible Python 3.12)
#######################

if not os.path.exists(os.path.join(path_output, 'PreFer_train_data_py312')):
    with open(os.path.join(path_data, "training_data","PreFer_train_data.csv")) as f:
          train_data = pd.read_csv(f, low_memory=False) 

    with open(os.path.join(path_output, 'PreFer_train_data_py312'), 'wb') as f:
             pickle.dump(train_data, f) 

else:
    with open(os.path.join(path_output, 'PreFer_train_data_py312'), 'rb') as f:
             train_data = pickle.load(f)
             
if not os.path.exists(os.path.join(path_output, 'PreFer_train_outcome_py312')):            
    with open(os.path.join(path_data, "training_data","PreFer_train_outcome.csv")) as f:
          train_outcome = pd.read_csv(f, low_memory=False) 

    with open(os.path.join(path_output, 'PreFer_train_outcome_py312'), 'wb') as f:
             pickle.dump(train_outcome, f)    
             
else:
    with open(os.path.join(path_output, 'PreFer_train_outcome_py312'), 'rb') as f:
             train_outcome = pickle.load(f) 

print('Datasets loaded')       



################################
# Transforms from wide to long #
################################

df_wide, df_long, dic_child_info, dic_type_codebook_wide,dic_norm_cont_long = wide_to_long_train_sub(train_data, path_data, path_output_train)

saved_params = {} # save normalisation for GRU propocessing  
saved_params['dic_norm_cont_long']= dic_norm_cont_long
saved_params['featList'] = list(df_long.columns)


df = df_long.copy()
df = df.sort_values(['nomem_encr','year'])
df.reset_index(inplace=True)


#######################
# Creation input cube #
#######################

cube = create_cube(df)

with open(os.path.join(path_output_train, 'Input_cube'), 'wb') as f:
          pickle.dump(cube, f)    
             
# with open(os.path.join(path_output_train, 'Input_cube'), 'rb') as f:
#           cube = pickle.load( f) 
###########################
# Creation outcomes input #
###########################

y1, y2, y3, y_for_pred = create_outputs_y(df, dic_child_info)

dic_y = {'y1':y1, 'y2':y2, 'y3':y3, 'y_for_pred': y_for_pred}

with open(os.path.join(path_output_train, 'Output_Ys'), 'wb') as f:
          pickle.dump(dic_y, f)    


#############################
# Creation 2020-2023 output #
#############################

i = df['nomem_encr'].nunique()
true_y = np.zeros((i))
for ind, index_ind in zip(df['nomem_encr'].unique(), range(i)):
    true_y[index_ind] = train_outcome.loc[train_outcome['nomem_encr']==ind, 'new_child'].iloc[0]
    
    
#############################
# Training gru Model        #
#############################   
    

y1 = dic_y['y1']
y2 = dic_y['y2']
y3 = dic_y['y3']
data = cube.copy()
data = data[:6400,:,:]
y1_true = y1[:6400,:]
y2_true = np.zeros((6400,7))
y3_true = np.zeros((6400,7))
for i in range(6400):
    if(y2[i,0]>0.5):
        y2_true[i,min(6,int(y2[i,0]))-1] = 1
    elif(y2[i,0]<0.5):
        y2_true[i,min(6,int(np.abs(y2[i,0]))):] = 1
    else:
        y2_true[i,:]=1
    if(y3[i,0]>0.5):
        y3_true[i,min(6,int(y3[i,0]))-1] = 1
    elif(y3[i,0]<0.5):
        y3_true[i,min(6,int(np.abs(y3[i,0]))):] = 1   
    else:
        y3_true[i,:]=1 

nvar = data.shape[2]
nT =  data.shape[1]


model = modelGru((nT,nvar))
model.summary()
model.fit(tf.convert_to_tensor(data), (tf.cast(tf.convert_to_tensor(y1_true), tf.float32),
                                       tf.cast(tf.convert_to_tensor(y2_true), tf.float32),
                                       tf.cast(tf.convert_to_tensor(y3_true), tf.float32)),
          batch_size=64,
          epochs=50,
          validation_split=0.1,
          verbose=1)
   
model.save( os.path.join(path_output_train, 'sumbissionGRU.keras'))   

model =  tf.keras.models.load_model(os.path.join(path_output_train, 'sumbissionGRU.keras')) 
parentInfo = (dic_y['y_for_pred'][:,0],
              dic_y['y_for_pred'][:,1],
              dic_y['y_for_pred'][:,2])


p = predict(cube,parentInfo,model)

#############################
# Training XGBoost Model    #
#############################   
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold,train_test_split
from sklearn.metrics import accuracy_score, f1_score
y_isna = train_outcome['new_child'].isnull()
X = df_wide.loc[~y_isna, df_wide.columns != 'nomem_encr']
Y = train_outcome.loc[~y_isna,['new_child']]


dic_type_codebook_wide.pop('nomem_encr', None)

# Identification categorical factors

cat_list = [key for key in dic_type_codebook_wide.keys() if dic_type_codebook_wide[key] == 'categorical']  

for var in cat_list:
    X[var].astype("category")


# Normalization numeric factors

cont_list = [key for key in dic_type_codebook_wide.keys() if dic_type_codebook_wide[key] == 'numeric']
    
dic_norm_cont_wide = {}
for var in cont_list:            
        dic_norm_cont_wide[var] = (np.nanmean(X[var]),np.nanstd(X[var]))
        X[var] = (X[var]-np.nanmean(X[var]))/np.nanstd(X[var])
        
saved_params['dic_norm_cont_wide']= dic_norm_cont_wide        

with open(os.path.join(path_output_train, 'saved_params'), 'wb') as f:
    pickle.dump(saved_params, f)
    
    
    
X['p_gru'] = p[~y_isna]


params = {
        'min_child_weight': [1, 5, 10],#
        'gamma': [0, 1, 2, 5],#0
        'subsample': [0.6, 0.8, 1.0],#1
        'colsample_bytree': [0.6, 1.0],#1
        'max_depth': [3, 5, 6]
        }

xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1, enable_categorical=True)

folds = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = GridSearchCV(xgb, param_grid=params, scoring='f1', n_jobs=4, cv=skf.split(X,Y), verbose=3)


random_search.fit(X, Y)


print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best F1:')
print(random_search.best_score_)
print('\n Best hyperparameters:')
print(random_search.best_params_)

# Feature importance
aaa = random_search.best_estimator_
xgb_fea_imp=pd.DataFrame(list(aaa.get_booster().get_fscore().items()),
columns=['feature','importance']).sort_values('importance', ascending=False)
print(xgb_fea_imp.iloc[0:15,:]) #feature importance

random_search.best_estimator_.predict(X)
pickle.dump(random_search.best_estimator_, open(os.path.join(path_output_train, 'sumbissionxgb2.json'), "wb"))


m2 = pickle.load(open(os.path.join(path_output_train, 'sumbissionxgb2.json'), "rb"))

#save
random_search.best_estimator_.save_model(os.path.join(path_output_train, 'sumbissionxgb.json'))
