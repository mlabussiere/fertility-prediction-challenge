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
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import xgboost as xgb
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# path_data = 'C:/Users/mlabuss/OneDrive - UvA/CAREER/Training/PreFer_Challenge/Data/other_data/'
# with open(path_data + "PreFer_fake_data.csv") as f:
#               df = pd.read_csv(f, low_memory=False) 
              



def wide_to_long(data, dic_norm_cont):
    '''
    Parameters
    ----------
    train_data: LISS train_data as pd.DataFrame()
   
    Returns
    -------
    df_wide : data frame in a wide format with selected columns
    df_long : data frame in a long format. It contain columns:
                        -'nomem_encr' identifier
                        -'birthyear_non_norm' birthdate in format YYYY
    dic_child_info : dictionary
                        - keys 'nomem_encr' individuals
                        - values list of tuples (d,s). each element refers to a child of the individual
                                - d date of birth of the child in format YYYY
                                - s status of the child: 1 biological, 2 adopted 
    '''
        
    with open("PreFer_codebook.csv") as f:
                codebook = pd.read_csv(f, low_memory=False) 
                
                
                
            
    
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
    
    df_wide = pd.DataFrame()
    list_coresurvey = []
    list_bckgd_cst = []
    list_bckgd_tv = []
    dic_stubnames = {}
    dic_var_type = {}
    for var in select_var:
        is_coresurvey = (data.columns.str.startswith(var[:2])) & data.columns.str.endswith(var[2:5])
        is_bckgd_cst = data.columns.str.fullmatch(var)
        is_bckgd_tv = data.columns.str.startswith(var)
        
        if is_coresurvey.any():
            var_in_col = data.columns[is_coresurvey].to_list()
            df_wide[var_in_col] = data[var_in_col]
            list_coresurvey.append(var)
            
            for tv_var in var_in_col:
                dic_var_type[tv_var] = "CORE"
            
        elif is_bckgd_cst.any():
            var_in_col = data.columns[is_bckgd_cst].to_list()
            df_wide[var_in_col] = data[var_in_col] 
            list_bckgd_cst.append(var)
            
            for tv_var in var_in_col:
                dic_var_type[tv_var] = "BCKGD_CST"
     
        elif is_bckgd_tv.any():
            var_in_col = data.columns[is_bckgd_tv].to_list()
            df_wide[var_in_col] = data[var_in_col]
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
     
                
     
    ############################ 
    # PREPARATION WIDE TO LONG #
    ############################
            
    
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
    prep_wide = df_wide.rename(columns=dic_col_names_2new)
    
    # Identification variable type in codebook for df_wide 
    dic_type_codebook_wide = {}  
    for var in df_wide: 
        if var[-2:] == '_f' :   
            var_f = var[:-2] + 'f'
            dic_type_codebook_wide[var] = codebook.loc[(codebook['var_name'] == var_f), 'type_var'].to_list()[0]
        else:
            dic_type_codebook_wide[var] = codebook.loc[(codebook['var_name'] == var), 'type_var'].to_list()[0]

    

    
    # From wide to long
    #####################
            
    list_stacked_stub = [dic_stubnames[dic_col_names_2old[col]] for col in prep_wide.columns if dic_var_type[dic_col_names_2old[col]] != 'BCKGD_CST']
    
    print("Transforming from wide to long...")                     
    long_data = pd.wide_to_long(prep_wide,
                    stubnames = set(list_stacked_stub),
                    i = 'nomem_encr',
                    j = 'year',
                    sep = '_',
                    suffix='\d+')
    
    
    long_data.reset_index(inplace=True)      


    ###################################        
    # REMOVE ROWS WITH MISSING VALUES #
    ###################################
    
    list_tv_var = list_coresurvey + list_bckgd_tv
    for i,elmt in enumerate(list_tv_var):
        if elmt[-2:] == '_f' :   
            list_tv_var[i] = elmt[:-2] + 'f'
    
    long_data['missing_row'] = long_data.loc[:,list_tv_var].isnull().all(axis=1)
    long_data['missing_row'].value_counts()
    long_data.groupby('nomem_encr')['missing_row'].sum()
    
    long_data = long_data.loc[long_data['missing_row'] == 0,:]
    long_data.drop(columns=['missing_row'], inplace=True)
    
    
    long_data.reset_index(inplace=True)  
        
    
    #################################################    
    # CREATION VARIABLES FOR TIME TO EVENT ANALYSIS #
    #################################################
    
    # Variables children's dates of birth
    for i in range(0,15):
        long_data['year_child_'+ str(i+1)] = long_data['cf' + str(37 + i).zfill(3)].fillna(long_data['cf' + str(456 + i)])
    
    # Variables for children's status
    for i in range(0,15):
        long_data['status_child_'+ str(i+1)] = long_data['cf' + str(98 + i).zfill(3)]
        long_data['status_child_'+ str(i+1)]  = np.where(long_data['status_child_'+ str(i+1)].isna(),
                                                                -1, 
                                                                # recode missing values of child type to -1 to be able to compare 
                                                                # if tuples are equal in dic_child_info_raw
                                                                long_data['status_child_'+ str(i+1)])
        
    # RAW DICTIONARY
        
    # Raw dictionary for children info: 
    # unique pairs of (child's date, child's status) for each individual 
    # Twins/triplets etc. are treated as one birth 
    dic_child_info_raw = {}
    for obs in range(0,long_data.shape[0]):
        individual = long_data.loc[obs,'nomem_encr']
    
        if not individual in dic_child_info_raw.keys():
            dic_child_info_raw[individual] = set()
                        
        for i in range(0,15):
            temp_year = long_data.loc[obs, 'year_child_'  + str(i+1)]
            temp_type = long_data.loc[obs, 'status_child_'+ str(i+1)]
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
    
    
    
    #############################
    # COVARIATES PRE-PROCESSING #
    #############################
    
    
    select_var_RNN = key_var_1 + key_var_2 + key_var_3 
    
    # Identification type of variable based on the codebook 
                
    dic_type_codebook_long = {}  
    for var in select_var_RNN: 
        if var not in list_bckgd_cst:
            if var[-2:] == '_f' :   
                var_f = var[:-2] + 'f'
                dic_type_codebook_long[var] = codebook.loc[(codebook['var_name'] == dic_col_names_2old[var_f + '_08']), 'type_var'].to_list()[0]
            else:
                dic_type_codebook_long[var] = codebook.loc[(codebook['var_name'] == dic_col_names_2old[var + '_08']), 'type_var'].to_list()[0]
        else:
            dic_type_codebook_long[var] = codebook.loc[(codebook['var_name'] == dic_col_names_2old[var]), 'type_var'].to_list()[0]
    
    
    # Creation dataset with model covariates (categorical are one-hot encoded)
    
    df_long = pd.DataFrame()
    df_long['year'] = long_data['year']
    df_long['age']  = 2000 + long_data['year'] - long_data['birthyear_bg']
    
    # Fill df_long with numerical variables and store categorical ones in dict
    dic_values_categorical = {}
    for var in select_var_RNN:
        if dic_type_codebook_long[var] == 'categorical':
            dic_values_categorical[var] = long_data[var].unique()
            dic_values_categorical[var].sort()
        else:
            if var[-2:] == '_f' :   
                var_f = var[:-2] + 'f'
                df_long[var] = long_data[var_f]
            else:
                df_long[var] = long_data[var]
    
                
    # Normalize continuous covariates and store mean and std in dict

    var_to_norm = [var for var in select_var_RNN if (dic_type_codebook_long[var] == 'numeric') & (var != 'nomem_encr')] + ['age']
    df_long['birthyear_non_norm'] = df_long['birthyear_bg']
    
    for var in var_to_norm:             
            df_long[var] = (df_long[var]-dic_norm_cont[var][0])/dic_norm_cont[var][1]
    
    

            

    # Temporary filling of NaNs for continuous variables
    for var in var_to_norm:
        df_long[var] = np.where(df_long[var].isna(), 0, df_long[var])
    
    # Move nomem_encr to first position (cf. create_cube excludes first columns)
    nomem = df_long.pop('nomem_encr')
    df_long.insert(0, 'nomem_encr', nomem)
    birthy = df_long.pop('birthyear_non_norm')
    df_long.insert(2, 'birthyear_non_norm', birthy)
            
    # Custom one-hot encoding for categorical variables where NaN is reference category
    # all values have separate dummy except NaN
    # if all dummies = 0 => value is NaN 
    
    for var in dic_values_categorical.keys():

        for val in dic_values_categorical[var]:
            if not np.isnan(val):
                    name = var + '_' + str(int(val))
                    df_long.loc[long_data[var] == val, name] = 1
                    df_long.loc[long_data[var] != val, name] = 0
    
    
    
    
    
    df_long = df_long.sort_values(['nomem_encr','year'])
    df_long.reset_index(inplace=True)


    #######################
    # Creation input cube #
    #######################

    cube,dic_nomencr_idx = create_cube(df_long)
    
    
    y_for_pred = gru_y_pred(df_long, dic_child_info)

    parentInfo = (y_for_pred[:,0],
                  y_for_pred[:,1],
                  y_for_pred[:,2])
    
    return cube, parentInfo, df_wide,dic_type_codebook_wide,dic_nomencr_idx


def create_cube(df):
    '''
    Create 3-D array for features input 
    
    Parameters
    ----------
    df: dataframe in the long format as created by wide_to_long_train_grurnn()
             with values sorted by nomem_encr and year
             and reset index 
    
    Returns
    --------
    Cube: 3 dimensional array (i,j,k) with:
        i the number of individuals
        j the maximum number of years observed
        k the number of variables 
    '''
    
    # Set output dimensions
    i =  df['nomem_encr'].nunique() # number of individuals 
    j =  df.groupby('nomem_encr')['year'].nunique().max()  # maximum number of years observed
    k =  df.shape[1] - 4 # number of variables excluding index, nomem_encr, year, birthyear_non_norm
                      
    
    print("Creating 3D feature input array...(can take a while)")
    cube = np.zeros((i,j,k))
    dic_nomencr_idx = {}
    for ind, index_ind in zip(df['nomem_encr'].unique(), range(i)):
        dic_nomencr_idx[ind] = index_ind
        t_start = df.loc[df['nomem_encr']== ind, 'year'].min()
        t_end = df.loc[df['nomem_encr']== ind, 'year'].max()

        for t, index_t in zip(range(t_start, t_end+1), range(j)):
            if t <= t_end :
                for var, index_var in zip(df.columns[4:], range(k)):
                    val_in_df = df.loc[(df['nomem_encr'] == ind) & (df['year'] == t), var]
                    if len(val_in_df) != 0: 
                        #the year is recorded for the individual
                        cube[index_ind][index_t][index_var] = val_in_df.iloc[0]
                    else:
                        cube[index_ind][index_t][index_var] = 0               
                
            elif t > t_end :
                for var, index_var in zip(df.columns[4:], range(k)):
                    cube[index_ind][index_t][index_var] = 0 
    
    return cube,dic_nomencr_idx
    
   

def predictProb(distFirst,distSec,distThird,nKid,T,dT):
    t_ = T+dT
    if nKid==0:
        out = np.sum(distFirst[t_:min(t_+3,27)])/(1-np.sum(distFirst[:T]))
    elif nKid==1:
        out = np.sum(distSec[t_:min(t_+3,6)])/(1-np.sum(distSec[:T]))
    else:
        out = np.sum(distThird[t_:min(t_+3,6)])/(1-np.sum(distThird[:T]))
    
    if dT!=0:
        if nKid==0:
            out = out + np.sum(distFirst[T:min(t_,27)])/(1-sum(distFirst[:T]))*np.sum(distSec[t_:min(t_+3,10)]/(1-sum(distSec[:t_])))
        elif nKid==1:
            out = out + np.sum(distSec[T:min(t_,10)])/(1-sum(distSec[:T]))*np.sum(distThird[t_:min(t_+3,10)]/(1-sum(distSec[:t_])))
        else:
            out = out + np.sum(distThird[T:min(t_,10)])/(1-sum(distThird[:T]))*np.sum(distThird[t_:min(t_+3,10)]/(1-sum(distSec[:t_])))
    
    return out  
    
def predict(data,outcome,model):
    distFirst,distSec,distThird = model(tf.convert_to_tensor(data))
    nKid,T,dT = outcome
    nInd = data.shape[0]
    predProb = np.zeros(nInd)
    for i in range(nInd):
        predProb[i] = predictProb(np.asarray(distFirst[i,:]),
                                  np.asarray(distSec[i,:]),
                                  np.asarray(distThird[i,:]),
                                  int(nKid[i]),int(T[i]),int(dT[i]))
    

    return predProb

   
def gru_y_pred(df, dic_child_info):
    '''
    Create arrays for outcomes input 
    
    Parameters
    ----------
    df: dataframe in the long format as created by wide_to_long_train_grurnn()
             with values sorted by nomem_encr and year
             and reset index 
    dic_child_info: dictionary recording respondents' children information,
                    as created by wide_to_long_train_grurnn()
    
    Returns
    --------
    y_for_pred: outcome for calculating prediction of gru, with for each ind i:
                y_for_pred[i,0]: number of children at the last year of observation
                y_for_pred[i,1]: years since last child at the last year of observation
                y_for_pred[i,2]: years between 2020 and the last year of observation
    '''
    
    # Intermediate dictionaries

    dic_date_child = {}
    for key in dic_child_info.keys():
        dic_date_child[key] = []
        
        for e1,e2 in dic_child_info[key]:
            dic_date_child[key].append(int(e1))
            
    dic_age_at_birth = {}
    for key in dic_child_info.keys():
        dic_age_at_birth[key] = []
        date_parent = df.groupby('nomem_encr')['birthyear_non_norm'].min()[key] 
        
        for date_child in dic_date_child[key]:
            age_at_birth = date_child-date_parent
            dic_age_at_birth[key].append(age_at_birth) 
            
    dic_age_at_birth_centered = {}
    inconsistent_cases = []
    for key in dic_child_info.keys():
        dic_age_at_birth_centered[key] = []
        date_parent = df.groupby('nomem_encr')['birthyear_non_norm'].min()[key] 
        
        for date_child in dic_date_child[key]:
            age_centered_18 = date_child-date_parent-18
            if age_centered_18<0:
            # birth at age 15-18 are bottom coded
                if age_centered_18 >= -3:
                    dic_age_at_birth_centered[key].append(0)
                else:
            # birth below age 15 are ignored
                    inconsistent_cases.append(key)
            elif age_centered_18>45:
            # birth after age 45 are top coded
                dic_age_at_birth_centered[key].append(27) 
            else:
                dic_age_at_birth_centered[key].append(age_centered_18) 
            
    inconsistent_cases = set(inconsistent_cases)        
    # 16 individuals with incoherent dates of birth
      
    # Bottom coding creates duplicates with different births being recorded in year 0
    dic_age_at_birth_centered = {k: list(set(v)) for k,v in dic_age_at_birth_centered.items()}
    # Dictionary recording the number of children
    
    dic_number_children = {}
    for key in dic_age_at_birth_centered.keys():
        if len(dic_age_at_birth_centered[key]) == 0:
            dic_number_children[key] = -1
        elif len(dic_age_at_birth_centered[key]) != 0:
            dic_number_children[key] = len(dic_age_at_birth_centered[key])
    

    
    ###################################
    # Creation outcome for prediction #
    ###################################
    
    i = df['nomem_encr'].nunique()
    j = 3
    
    print("Creating 2D output prediction array...")   
    
    y_for_pred = np.zeros((i,j))
    
    for ind, index_ind in zip(df['nomem_encr'].unique(), range(i)):
        
        last_year = 2000 + df.loc[df['nomem_encr']== ind, 'year'].max() 
        
        y_for_pred[index_ind][0] = dic_number_children[ind]
        
        if len(dic_date_child[ind]) != 0:
        
            y_for_pred[index_ind][1] = last_year - max(dic_date_child[ind])
        
        else:
        
            y_for_pred[index_ind][1] = last_year - df.loc[df['nomem_encr']== ind, 'birthyear_non_norm'].min() - 18
        
        
        y_for_pred[index_ind][2] = 2020 - last_year
    
    
    # Replace number of children from -1 to 0 
    y_for_pred[:,0] = np.where(y_for_pred[:,0] == -1, 0, y_for_pred[:,0])
    
                    
    return  y_for_pred

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
    

    return df


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


    # load parameters
    with open('saved_params.pkl', 'rb') as f:
                saved_params = pickle.load(f)
                
                
    dic_norm_cont_long = saved_params['dic_norm_cont_long']
    dic_norm_cont_wide = saved_params['dic_norm_cont_wide']
    
    # Preprocess the fake / holdout data
    cube, parentInfo, df_wide, dic_type_codebook_wide,dic_nomencr_idx = wide_to_long(df, dic_norm_cont_long)


    #estimate probability with GRU RNN
    GRU =  tf.keras.models.load_model('sumbissionGRU.keras') 
    p = predict(cube, parentInfo, GRU)



    #  predict with an XGBoost 
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model('xgb_model.json')  # load data
    
    X = df_wide
    
    
    dic_type_codebook_wide.pop('nomem_encr', None)
    
    # Identification categorical factors
    
    cat_list = [key for key in dic_type_codebook_wide.keys() if dic_type_codebook_wide[key] == 'categorical']  
    
    for var in cat_list:
        X[var].astype("category")
    
    
    # Normalization numeric factors
    
    cont_list = [key for key in dic_type_codebook_wide.keys() if dic_type_codebook_wide[key] == 'numeric']
        
    
    for var in cont_list:            
            X[var] = (X[var]-dic_norm_cont_wide[0])/dic_norm_cont_wide[1]
            
    X['p_gru'] = p*0
    for r in range(X.shape[0]):
        X.loc[r,'p_gru'] = p[dic_nomencr_idx[ X.loc[r,'nomem_encr']]]
    
    prediction = bst.predict(X)
    df_pred = pd.DataFrame()
    df_pred['nomem_encr'] =  X['nomem_encr']
    df_pred['prediction'] =  prediction
    return df_pred


pred = predict_outcomes("PreFer_fake_data.csv")