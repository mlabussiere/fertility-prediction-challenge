"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
# import os
# import sys
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from class_DeepLongitudinal_DH  import Model_Longitudinal_Attention

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# path_data = 'C:/Users/mlabuss/OneDrive - UvA/CAREER/Training/PreFer_Challenge/Data/other_data/'
# with open(path_data + "PreFer_fake_data.csv") as f:
#               df = pd.read_csv(f, low_memory=False) 

# Functions
def wide_to_long_holdout(holdout_data):
    '''
    Parameters
    ----------
    holdout_data: LISS holdout_data as pd.DataFrame()
        
    Returns
    -------
    df_long : data frame in a long format. It contain columns:
                        -'nomem_encr' identifier
                        -'birthyear_bg' birhtdate in format YYYY
                        - 'nb_previous_kids' nbr of kids per individuals 
    nbr_kids : dictionary giving the number of children (value) for each individual (key)

    '''
    
    
    ######################
    # VARIABLE SELECTION #
    ######################
    
    # Variable names without date information
     # - core survey variables: module + question (e.g., cd14g034 -> cd034)
    
    key_var_1 = ['nomem_encr', 'birthyear_bg', 'gender_bg', 'migration_background_bg', 
                 'partner', 'woonvorm', 'burgstat', 'woning', 'nettohh_f', 'belbezig',
                 'oplmet', 'cf128', 'cf129']
    
    # oplmet vs. oplzon
    # household vs. individual income
    # net vs. gross income
    
    key_var_2 = ['sted', 'cd034', 'ch004', 'cr012', 'ch125', 'ch219']
    
    # cd034 number of rooms in dwelling
    # ch219 contact with gynaecologist in the last 12 months
    # ch125 have you ever smoked
    # ch004 self-rated health
    # cr012 do you consider yourself as part of a religious community (2008-2018)
    
    key_var_3 = ['ch085', 'ch169']
    
    # ch169 cholesterol 
    # ch085 diabetis
    
    fertility_var = ['cf0' + str(i) for i in range(35,52)] \
                  + ['cf' + str(i).zfill(3) for i in range(98,113)] \
                  + ['cf' + str(i) for i in range(456,471)]
    # Construction variables for time-to-event analysis (date of birth and
    # statuses of children)
    
    # OTHERS
    # cf483-488 gender division within household (2015-2020)
    
    select_var = key_var_1 + key_var_2 + key_var_3 + fertility_var
    
    
    
    ###########################################################
    # LOOP OVER VARIABLE NAMES TO RETRIEVE ALL RELEVANT YEARS #
    ###########################################################
    # Creates dictionary of stub names at the same time for wide_to_long()
    
    prep_wide = pd.DataFrame()
    list_coresurvey = []
    list_bckgd_cst = []
    list_bckgd_tv = []
    dic_stubnames = {}
    dic_var_type = {}
    for var in select_var:
        is_coresurvey = (holdout_data.columns.str.startswith(var[:2])) & holdout_data.columns.str.endswith(var[2:5])
        is_bckgd_cst = holdout_data.columns.str.fullmatch(var)
        is_bckgd_tv = holdout_data.columns.str.startswith(var)
        
        if is_coresurvey.any():
            var_in_col = holdout_data.columns[is_coresurvey].to_list()
            prep_wide[var_in_col] = holdout_data[var_in_col]
            list_coresurvey.append(var)
            
            for tv_var in var_in_col:
                dic_var_type[tv_var] = "CORE"
            
        elif is_bckgd_cst.any():
            var_in_col = holdout_data.columns[is_bckgd_cst].to_list()
            prep_wide[var_in_col] = holdout_data[var_in_col] 
            list_bckgd_cst.append(var)
            
            for tv_var in var_in_col:
                dic_var_type[tv_var] = "BCKGD_CST"
     
        elif is_bckgd_tv.any():
            var_in_col = holdout_data.columns[is_bckgd_tv].to_list()
            prep_wide[var_in_col] = holdout_data[var_in_col]
            list_bckgd_tv.append(var)
            
            for tv_var in var_in_col:
                dic_var_type[tv_var] = "BCKGD_TV"
               
        else:
            raise Exception("No matching for " + var)
            
        for tv_var in var_in_col:
            if tv_var[-6] == 'f' :  # recode _f into f (imputed variables) 
                dic_stubnames[tv_var] = var[:-2] + "f" 
            else:
                dic_stubnames[tv_var] = var   
      
     
    #####################   
    # FROM WIDE TO LONG #
    #####################
            
    
    # Creation dictionary variable names 
    #####################################
    
    # {old column names: new column names}
    dic_col_names_2new = {key: key[:2] + key[-3:] + '_' + key[2:4] for key in dic_stubnames.keys() if dic_var_type[key] == 'CORE'}
    
    for key in dic_stubnames.keys() :
        if dic_var_type[key] == 'BCKGD_TV':
            if (key[-6]=='f'): # case of imputed variables
                dic_col_names_2new[key] = key[:-7] + 'f_' + key[-2:] 
            else:
                dic_col_names_2new[key] = key[:-4] + key[-2:] 
        
        if dic_var_type[key] == 'BCKGD_CST':
            dic_col_names_2new[key] = key
    
    # {new column names: old column names} 
    dic_col_names_2old ={value:key for key, value in dic_col_names_2new.items()}
    
    # Rename column names to prepare for wide_to_long()
    prep_wide_renamed = prep_wide.rename(columns=dic_col_names_2new)
    
    
    # From wide to long
    #####################
            
    list_stacked_stub = [dic_stubnames[dic_col_names_2old[col]] for col in prep_wide_renamed.columns if dic_var_type[dic_col_names_2old[col]] != 'BCKGD_CST']
    
    print("Transforming from wide to long...")                     
    long_holdout_data = pd.wide_to_long(prep_wide_renamed,
                   stubnames = set(list_stacked_stub),
                   i = 'nomem_encr',
                   j = 'year',
                   sep = '_',
                   suffix='\d+')
    
    
    long_holdout_data.reset_index(inplace=True)      
        
     
        
    #################################################    
    # CREATION VARIABLES FOR TIME TO EVENT ANALYSIS #
    #################################################
    
    # Variables children's dates of birth
    for i in range(0,15):
        long_holdout_data['year_child_'+ str(i+1)] = long_holdout_data['cf' + str(37 + i).zfill(3)].fillna(long_holdout_data['cf' + str(456 + i)])
    
    # Variables for children's status
    for i in range(0,15):
        long_holdout_data['status_child_'+ str(i+1)] = long_holdout_data['cf' + str(98 + i).zfill(3)]
        long_holdout_data['status_child_'+ str(i+1)]  = np.where(long_holdout_data['status_child_'+ str(i+1)].isna(),
                                                               -1, 
                                                               # recode missing values of child type to -1 to be able to compare 
                                                               # if tuples are equal in dic_child_info_raw
                                                               long_holdout_data['status_child_'+ str(i+1)])
        
    # RAW DICTIONARY
        
    # Raw dictionary for children info: 
    # unique pairs of (child's date, child's status) for each individual 
    # Twins/triplets etc. are treated as one birth 
    dic_child_info_raw = {}
    for obs in range(0,long_holdout_data.shape[0]):
        individual = long_holdout_data.loc[obs,'nomem_encr']
    
        if not individual in dic_child_info_raw.keys():
            dic_child_info_raw[individual] = set()
                        
        for i in range(0,15):
            temp_year = long_holdout_data.loc[obs, 'year_child_'  + str(i+1)]
            temp_type = long_holdout_data.loc[obs, 'status_child_'+ str(i+1)]
            if not (np.isnan(temp_year) and (temp_type == -1)) : 
               dic_child_info_raw[individual].add((temp_year,temp_type))
    
    
    # FINAL DICTIONARY
    
    # Final dictionary for children info that filters only biological/adopted children
    # Recodes missing status into 1 if no other child occurrence with the same date and recorded status
    n_tmptot = 0 # total number of children
    n_tmp1 = 0 # number children born as biological
    n_tmp3 = 0 # number children born as adopted
    n_tmpm1 = 0 # number of children recoded from missing to biological in the absence of information to the contrary
    dic_child_info = {}
    
    for ind, child_info in dic_child_info_raw.items():
        if(len(child_info)==0):
            dic_child_info[ind] = [] 
        else:
            dic_child_info[ind] = []
            tmp_dates, tmp_status = list(zip(*child_info))
            tmp_status = np.array(tmp_status)
            for d in set(tmp_dates):
                n_tmptot +=1
                if not np.isnan(d):
                    st_ = tmp_status[tmp_dates==d]
                    if (1 in st_): 
                        # if one of the repeats is a 1 then 1
                        dic_child_info[ind]+= [(d,1)]
                        n_tmp1 +=1
                    elif (3 in st_): 
                        # if one of the repeats is a 3 (and there is no 1) then 3
                        dic_child_info[ind]+= [(d,1)]
                        n_tmp3 +=1
                    elif not ((2 in st_) or (4 in st_)): 
                        # if none of the repeats is a 2 or 4 (and there is no a 1 and 3) then 1 
                        # (== if all the repeats are -1)
                       dic_child_info[ind]+= [(d,1)]
                       n_tmpm1 +=1
    
    
    # preparation output number of kids per individual
    
    ind_list = set(long_holdout_data['nomem_encr'])
    nb_kids = {}
    for ind in ind_list:
        nb_kids[ind] = len(dic_child_info[ind])
        
    
    #############################
    # COVARIATES PRE-PROCESSING #
    #############################
    
    
    select_var_DHH = key_var_1 + key_var_2 + key_var_3 
    
    # Identification type of variable based on the codebook 
    
    with open("PreFer_codebook.csv") as f:
                codebook = pd.read_csv(f, low_memory=False) 
                
    dic_type_codebook = {}  
    for var in select_var_DHH: 
        if var not in list_bckgd_cst:
            if var[-2:] == '_f' :   
               var_f = var[:-2] + 'f'
               dic_type_codebook[var] = codebook.loc[(codebook['var_name'] == dic_col_names_2old[var_f + '_08']), 'type_var'].to_list()[0]
            else:
               dic_type_codebook[var] = codebook.loc[(codebook['var_name'] == dic_col_names_2old[var + '_08']), 'type_var'].to_list()[0]
        else:
            dic_type_codebook[var] = codebook.loc[(codebook['var_name'] == dic_col_names_2old[var]), 'type_var'].to_list()[0]
    
    
    # Creation dataset with model covariates (categorical are one-hot encoded)
    
    data_cov = pd.DataFrame()
    data_cov['year'] = long_holdout_data['year']
    
    # Fills data_cov with numerical variables and store categorical ones in dict
    dic_values_categorical = {}
    for var in select_var_DHH:
        if dic_type_codebook[var] == 'categorical':
            dic_values_categorical[var] = long_holdout_data[var].unique()
            dic_values_categorical[var].sort()
        else:
            if var[-2:] == '_f' :   
               var_f = var[:-2] + 'f'
               data_cov[var] = long_holdout_data[var_f]
            else:
                data_cov[var] = long_holdout_data[var]
       
                
    # Custom one-hot encoding for categorical variables that incorporate missing values
    for var in dic_values_categorical.keys():
        first = True
        for val in dic_values_categorical[var]:
            if not np.isnan(val):
                if first:
                    first=False # we drop first value of each variable 
                else:
                    name = var + '_' + str(int(val))
                    data_cov[name] = 0*long_holdout_data[var] # preserves the NaNs (0*NaN = NaN)
                    data_cov.loc[long_holdout_data[var] == val,name] = 1
    
    
    # Add number of kids extracted from dic_child_info

    data_cov = pd.merge(data_cov, pd.DataFrame({'nomem_encr': nb_kids.keys(), 'nb_previous_kids': nb_kids.values()}), on='nomem_encr') 
    
    # Add other construction variables
    
    data_cov['id'] = data_cov['nomem_encr']

    data_cov['times'] = (data_cov['year'] - data_cov['birthyear_bg']) + 2000 - 18 
 
    return data_cov


def f_construct_dataset_holdout(df, feat_list):
    '''

           
    '''

    grouped  = df.groupby(['id'])
    id_list  = pd.unique(df['id'])
    max_meas = np.max(grouped.count())[0]

    data     = np.zeros([len(id_list), max_meas, len(feat_list)+1])

    for i, tmp_id in enumerate(id_list):
        tmp = grouped.get_group(tmp_id).reset_index(drop=True)

        data[i, :int(tmp.shape[0]), 1:]  = tmp[feat_list]
        data[i, :int(tmp.shape[0]-1), 0] = np.diff(tmp['times'])
    
    return data


def cont_norm(df,dic_norm_cont): 
    df_ = df.copy()
    for var,(mu,sig) in dic_norm_cont.items():
        if var in df.columns:
            df_[var] = (df_[var]-mu)/sig
        else:
            print(var+' was not found in dataset.')

    return df_
    

def initialise_model(params):
    x_dim, x_dim_cont, x_dim_bin,num_Event, num_Category,max_length,new_parser = params  
    # INPUT DIMENSIONS
    input_dims                  = { 'x_dim'         : x_dim,
                                    'x_dim_cont'    : x_dim_cont,
                                    'x_dim_bin'     : x_dim_bin,
                                    'num_Event'     : num_Event,
                                    'num_Category'  : num_Category,
                                    'max_length'    : max_length }
    
    # NETWORK HYPER-PARMETERS
    network_settings            = { 'h_dim_RNN'         : new_parser['h_dim_RNN'],
                                    'h_dim_FC'          : new_parser['h_dim_FC'],
                                    'num_layers_RNN'    : new_parser['num_layers_RNN'],
                                    'num_layers_ATT'    : new_parser['num_layers_ATT'],
                                    'num_layers_CS'     : new_parser['num_layers_CS'],
                                    'RNN_type'          : new_parser['RNN_type'],
                                    'FC_active_fn'      : new_parser['FC_active_fn'],
                                    'RNN_active_fn'     : new_parser['RNN_active_fn'],
                                    'initial_W'         : tf.contrib.layers.xavier_initializer(),
    
                                    'reg_W'             : new_parser['reg_W'],
                                    'reg_W_out'         : new_parser['reg_W_out']
                                     }
    

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    sess = tf.Session(config=config)
    model = Model_Longitudinal_Attention(sess, "Dyanmic-DeepHit", input_dims, network_settings)

    return(sess,model)


def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """
    df_data_cov = wide_to_long_holdout(df)

    return df_data_cov


def predict_outcomes(df, background_df=None, model_path=None):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """


    # Preprocess the fake / holdout data
    data_cov = clean_df(df, background_df=None)

    with open('params_dictionary.pkl', 'rb') as f:
                saved_params = pickle.load(f)
    data_cov_norm = cont_norm(data_cov, saved_params['dic_norm_cont'])


    # Construction dataset for DynDeepHit (data)
    bin_list   = ['gender_bg_2', 'migration_background_bg_101',
         'migration_background_bg_102', 'migration_background_bg_201',
         'migration_background_bg_202', 'partner_1', 'woonvorm_2', 'woonvorm_3',
         'woonvorm_4', 'woonvorm_5', 'burgstat_2', 'burgstat_3', 'burgstat_4',
         'burgstat_5', 'woning_2', 'woning_4', 'belbezig_2', 'belbezig_3',
         'belbezig_4', 'belbezig_5', 'belbezig_6', 'belbezig_7', 'belbezig_8',
         'belbezig_9', 'belbezig_10', 'belbezig_11', 'belbezig_12',
         'belbezig_13', 'belbezig_14', 'oplmet_2', 'oplmet_3', 'oplmet_4',
         'oplmet_5', 'oplmet_6', 'oplmet_7', 'oplmet_8', 'oplmet_9', 'cf128_2',
         'cf128_3', 'sted_2', 'sted_3', 'sted_4', 'sted_5', 'ch004_2', 'ch004_3',
         'ch004_4', 'ch004_5', 'cr012_2', 'ch125_2', 'ch219_1', 'ch085_1',
         'ch169_1']
    cont_list = ['birthyear_bg', 'nettohh_f', 'cf129', 'cd034', 'nb_previous_kids']
    feat_list          = cont_list + bin_list
     
    data = f_construct_dataset_holdout(data_cov_norm, feat_list)

    # Creation mask for missing values (data_mi)
    data_mi                  = np.zeros(np.shape(data))
    data_mi[np.isnan(data)]  = 1
    data[np.isnan(data)]     = 0 


    # DYNAMIC DEEPHIT MODEL 
    #########################

    sess,model = initialise_model(saved_params['params_nx_init'])
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, "model.ckpt")
    pred = model.predict(data, data_mi)
    eval_time = 3
    _EPSILON = 1e-8
    risk = []

    # In the event of missing values for birthyear_bg
    data_cov['birthyear_bg'] = np.where(data_cov['birthyear_bg'].isna(), data_cov['birthyear_bg'].mean(), data_cov['birthyear_bg'])

    for idx in range(0,len(df['nomem_encr'])):  
        ind = df.loc[idx,'nomem_encr']
        
        last_meas = int(2020-data_cov.loc[data_cov['nomem_encr']==ind,'birthyear_bg'].iloc[0]-18)
      
        tmp_pred = pred[idx]
        tmp_risk = np.sum(tmp_pred[:,last_meas:(last_meas+eval_time)], axis=1) #risk score until eval_time
        tmp_risk2 = np.sum(tmp_pred, axis=1)
        # tmp_risk = np.sum(tmp_pred[last_meas:(last_meas+eval_time)]) #risk score until eval_time
        # tmp_risk = np.sum(tmp_pred[:(+eval_time)]) #risk score until eval_time
        # risk += [tmp_risk / (np.sum(tmp_pred[last_meas:]) +_EPSILON)] #conditioniong on t > t_pred

        risk += [tmp_risk[0]/(tmp_risk[1]+_EPSILON)] #conditioniong on t > t_pred
                
        # risk += [tmp_risk[1]] #conditioniong on t > t_pred
          

    # FOR TEST

    # test = pd.merge(df_data, df_outcome, on='nomem_encr' )
    # test = df_data.copy()
    # test['risk'] = risk
    # test = test.loc[test['new_child'].notna(),:]
    # test['pred'] = test['risk']>0.00025
    # pd.crosstab(test['pred'], test['new_child'])
    # test.groupby('new_child')['risk'].median()  

    # FOR PREDICT_OUTCOME()

    test = df.copy()
    test['risk'] = risk
    test['pred'] = test['risk']>0.00025

    df_predict = pd.DataFrame(
        {"nomem_encr": test["nomem_encr"], "prediction": test['pred']}
    )
    print("SHAPE PREDICT DATAFRAME")
    print(df_predict.shape)
    # Return only dataset with predictions and identifier
    return df_predict


# pred = predict_outcomes(df)