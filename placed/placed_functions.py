import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt
from matplotlib import pyplot
import keras as K
import keras
import joblib
from lightgbm import plot_importance
from lightgbm import LGBMClassifier


### COMMON FUNCTION BETWEEN WINNER AND PLACED

def how_many_do_we_win(y_pred_value,y_real_value,df):
    """
    Return how many do we win and the mean values of the win odds
    """
    win_amount = 0
    List_odds = []
    for i in range(len(y_pred_value)):
        if y_pred_value[i] == y_real_value[i]:
            val = df.win_odds[:][y_pred_value[i]+1].tolist()[i] #+1 because we need horse_no and not indices from a list
            win_amount = win_amount + val
            List_odds.append(val)
    return win_amount, np.mean(List_odds)

    
def draw_evolution(df):
    """
    Draw the evolution according to the date
    """
    plt.figure(figsize=(10,10)).suptitle('Evolution of the profit with a $100 bet', fontsize=20)
    plt.ylabel('profit', fontsize=18)
    plt.xlabel('date', fontsize=16)
    plt.xticks(rotation=90)
    plt.plot(df.date, df.cumul_100)
  
    
def draw_evolution_race(df,model_name):
    """
    Draw the evolution according to the number of races
    """
    plt.figure(figsize=(10,10)).suptitle(f'Evolution of the profit with $100 for {model_name}', fontsize=20)
    plt.ylabel('profit ($)', fontsize=18)
    plt.xlabel('num bets', fontsize=16)
    plt.xticks(rotation=0,fontsize=15)
    plt.yticks(rotation=0,fontsize=15)
    plt.plot(df.index, df.cumul_100)    
    
    
def create_X_TEST(df_init):
    """
    return a simplified dataframe used to retrieve win_odds and place_odds to calcul profit
    """
    INDEX = ['race_id', 'date', 'race_no', 'venue', 'config', 'surface', 'distance', 'going', 'horse_ratings', 'prize', 'race_class']
    features = ['draw','place_odds','win_odds','result']
    return df_init[INDEX + features]



def create_x_and_y(df):
    """
    function which return 2 dataframe for the features and labels used for trainning and testing
    """
    
    data = df
    
    #all first columns will be use for X
    X = data[data.columns[:-14]] 
    ss = preprocessing.StandardScaler()
    X = pd.DataFrame(ss.fit_transform(X),columns = X.columns)

    #all 14 last columns will be used for y
    #we use 1 if the horse win the race and 0 otherwize
    y = data[data.columns[-14:]].applymap(lambda x: 1.0 if x == 1 else 0.0) 

    return X,y



#####

def prepare_and_split_data_placed(X_train_init,X_test_init):
    """
    this function do the data prepartion then split and give us the good datasets according to the months we are trainning
    """
    
    #remove non available odds
    X_train = remove_place_odds_non_available(X_train_init)
    X_test = remove_place_odds_non_available(X_test_init)
    
    
    #create a new columns with how many runners we need to bet on (2 or 3 horses)
    X_train = hm_runners(X_train)
    X_test = hm_runners(X_test)
    
    #create the label
    X_train = create_label(X_train)
    X_test = create_label(X_test)

    #dropping some columns
    Drop = ['race_id','race_no','hm_runners','date','won','place']
    X_tr = X_train.drop(Drop,axis=1,level=0)
    X_te = X_test.drop(Drop,axis=1,level=0)

    #how many features
    L = []
    for col in X_tr.columns.tolist():
        L.append(col[0])

    #print(f"We only keep {len(set(L))} columns in totals")

    #We slipt each dataset between features and labels
    X_train, y_train = create_x_and_y(X_tr)
    print("shape of the x_train: ", X_train.shape)
    print("shape of the y_train: ", y_train.shape)
    
   
    
    if X_te.shape[0] == 0 :
        print("Error, we can't work with this set")
        print("They are no values available")
        return False
        
    else :
        
        X_test, y_test = create_x_and_y(X_te)
        print("shape of the X_test: ", X_test.shape)
        print("shape of the y_test: ", y_test.shape)

        #compute y_train_value : show the winner for each races
        y_test_value = y_test.values.tolist()
        y_test_value = np.array([np.argmax(t) for t in y_test_value])
        
        #compute y_test_value
        y_train_value = y_train.values.tolist()
        y_train_value = np.array([np.argmax(t) for t in y_train_value])

    return X_train, y_train, X_test, y_test, y_train_value, y_test_value, X_test_init


def remove_place_odds_non_available(df):
    """
    This function return a df with only race where place odds are available for all horses
    """
    df['test'] = np.where((df['draw'] == 0).astype(int).sum(axis=1) == (df['place_odds'] == 0).astype(int).sum(axis=1), 1,0) #1 is good and 0 need to be removed
    df = df[df['test']==1].copy()
    df.drop('test',axis=1, level = 0, inplace=True)
    return df


def hm_runners(df):
    """
    return a df with a new columns with the number of placed horses needed
    """
    nb_of_vacant_position = (df['draw'] == 0).astype(int).sum(axis=1)
    df.insert(loc = 0, column = 'hm_runners', value = 14 - nb_of_vacant_position)
    return df



def to_3_value(pred,real,df,match_race_id_from_indices):
    """
    return 3 or 2 values for prediction and real value for placed horses depending on the numbers of runners
    """
    #compute real values
    arr = real
    real = [ar.argsort()[-3:][::-1]+1 if np.count_nonzero(ar==0)==11 else ar.argsort()[-2:][::-1]+1 for ar in arr]

    #compute prediciton values
    result = []
    arr = pred
    for i in range(len(arr)):
        res = arr[i].argsort()[:][::-1]+1
        k = []
        val = 0
        for j in range(len(res)):
            val_seuil = len(real[i])

            if (get_place_odds(df,match_race_id_from_indices[i],res[j])!=0 and val < val_seuil) :
                val = val + 1
                k.append(res[j])

        result.append(k)

    return real, result


def get_place_odds(df,race_id,horse_no):
    """
    Return the place_odds for a race_id and a horse_no
    """
    A = df[df['race_id']==race_id]['place_odds'].iloc[0][horse_no-1]
    # we could put a condition if the value does not exist and return 1 by default but we prefer discard bets when this happens
    return A 


def compute_df_placed(pred,real,df,match_id):
    """
    return a df with information to draw the evolution of our investement
    """
    real, pred = to_3_value(pred,real,df,match_id)
    L_odds = []
    for i in range(len(real)):
        L_odds.append([get_place_odds(df,match_id[i],real[i][j]) for j in range(len(real[i]))])

    df_draw = pd.DataFrame(list(zip(pred,real)), columns=['pred','real'])   
    df_draw['place_odds_real'] = L_odds
    df_draw['profit'] = df_draw.apply(list_to_odds,axis=1)
    df_draw['cumul'] = df_draw['profit'].cumsum()
    df_draw['cumul_100'] = df_draw['cumul'] + 100
    return df_draw

def list_to_odds(x):
    """
    return the amount of money we win according to prediction, real values and place_odds
    """
    tot = 0
    list_pred = x.pred
    list_real = x.real
    place_odds_real = x.place_odds_real
    for i in range(len(list_real)):
        if list_real[i] in list_pred:
            tot = tot + place_odds_real[i] - 1
        else:
            tot = tot - 1
    return tot

def create_label(df_entry):
    """
    return a df with label at 1 or 0
    """
    df = df_entry.copy()
    df.loc[df.hm_runners>=7,'result'] = df[df.hm_runners>=7][df.columns[-14:]].applymap(lambda x: 1 if 0.5 < x < 3.5 else 0)
    df.loc[df.hm_runners<7,'result'] = df[df.hm_runners<7][df.columns[-14:]].applymap(lambda x: 1 if 0.5 < x < 2.5 else 0)
    return df

def function_less_place_odds(df):
    """
    this function return the mean of each 3 less place_odds
    """
    List_ = df.place_odds[:].values
    M = []
    for i in range(len(L)):
        J = sorted(L[i][np.nonzero(L[i])])[:2]
        if len(J)!=0:
            M.append(np.mean(sorted(L[i][np.nonzero(L[i])])[:2]))
    return np.mean(M)

def compute_gain(y_test,y_pred,df,match_id):
    """
    compute the gain and some other usefull informations
    """
    good_guesses = 0
    tot = 0
    revenue = 0
    real, pred = to_3_value(y_pred,y_test.values,df,match_id)
    L = []
    for i in range(len(y_test)):
        
        for j in range(len(pred[i])):
            if pred[i][j] in real[i]:
                good_guesses += 1
                val = df.place_odds[:][pred[i][j]].tolist()[i]
                L.append(val)
                revenue = revenue + val

        tot = tot + len(real[i])

    return revenue, tot, good_guesses, np.mean(L)



def compute_df_placed_xgb(pred,real,df,match_id):
    """
    return a df with information to draw the evolution of our investement
    """
    L = []
    for i in range(len(real)):
        L.append([get_place_odds(df,match_id[i],real[i][j]) for j in range(len(real[i]))])

    Z = pd.DataFrame(list(zip(pred,real)), columns=['pred','real'])   
    Z['place_odds_real'] = L
    Z['profit'] = Z.apply(list_to_odds,axis=1)
    Z['cumul'] = Z['profit'].cumsum()
    Z['cumul_100'] = Z['cumul'] + 100
    return Z





def change_shape(pred_bad_shape):
    """
    change the shape of the prediction
    """

    A = [pred_bad_shape[i][:,1] for i in range(14)]

    pred_good_shape = [np.array([A[i][j] for i in range(14)]) for j in range(len(A[0]))]

    return pred_good_shape


def mean_place_odds(X_test_init):
    """
    function which return the mean place odd for all race and all horse for this month
    """
    list_place_odds = X_test_init.place_odds.values

    non_null_place_odds = []

    for L in list_place_odds:
        for j in L:
            if j!=0:
                non_null_place_odds.append(j)
            
    return np.mean(non_null_place_odds)

def ensemble_model_placed(pred_dl,pred_lgbm,coef_dl):
    """
    Compute the ensemble result thanks to percentage prediciton 
    """
    new_pred = coef_dl*pred_dl + (1-coef_dl)*pred_lgbm
    return new_pred



def train_dl(num_neutron,batch_size,epoch,X_train,y_train,X_test,y_test):
    """
    This function will allow us to train our deep learning model
    """
    import keras as K
    import numpy as np
    np.random.seed(1) # NumPy
    import random
    random.seed(2) # Python
    from tensorflow import random
    random.set_seed(3)



    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_neutron, activation='relu', input_shape=(1618,)),
        tf.keras.layers.Dense(14, activation='softmax')
        ])

    model.compile(optimizer=tf.keras.optimizers.Adam(5e-04),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.Precision(name='precision')])

    dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
    train_dataset = dataset.shuffle(len(X_train)).batch(batch_size)

    dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
    validation_dataset = dataset.shuffle(len(X_test)).batch(batch_size)

    print("Start training..\n")
    history = model.fit(train_dataset, epochs=epoch, validation_data=validation_dataset)
    print("Done.")
    return model



#### COMPUTE ALL PROFIT 


def compute_profil(month, X_train, y_train, X_test, y_test, y_train_value, y_test_value, X_test_init):
    
    
    
    """
    Compute profit for all models (Deep Learning, LGBM, Ensemble model)
    """
        
        
    # Set some variables for compute the profit 
    
    
    X_TEST = create_X_TEST(X_test_init)
    
    match_race_id_from_indices = X_test_init.race_id.to_list()
    
    
    # DEEP LEARNING
    
    
    model = keras.models.load_model(f'model/placed_DL_{month}.h5')
    y_pred_dl = model.predict(X_test)
    
    # Compute the deep learning profit
    
    #--  
    revenue,hm_bet,good_guesses,mean_sucess_pred =  compute_gain(y_test,y_pred_dl,X_TEST,match_race_id_from_indices)
 
    perc_dl = round((good_guesses / hm_bet) * 100,2)

    profit_DL = revenue-hm_bet 
    #--
    
    
    ###############
    
    # LGBM

    
    filename = f'model/winner_lgbm_{month}'

    #load saved model
    lgbm = joblib.load(filename)
    
    y_pred_lgbm = lgbm.predict_proba(X_test)
    
    
    # Compute the lgbm profit
    
    #--      
    revenue,hm_bet,good_guesses,mean_sucess_pred =  compute_gain(y_test,y_pred_lgbm,X_TEST,match_race_id_from_indices)
 
    perc_lgbm = round((good_guesses / hm_bet) * 100,2)

    profit_lgbm = revenue-hm_bet       
    #-- 

    
    ###############
    
    #Ensemble Model
    
    
    
    pred_proba_dl = model.predict_proba(X_test)
    pred_proba_lgbm = lgbm.predict_proba(X_test)
    
    pred_classes = ensemble_model_placed(pred_proba_dl,pred_proba_lgbm,0.3)
    
    # Compute the ensemble profit
    
    #--   
    revenue,hm_bet,good_guesses,mean_sucess_pred =  compute_gain(y_test,pred_classes,X_TEST,match_race_id_from_indices)
 
    perc_conso = round((good_guesses / hm_bet) * 100,2)

    profil_conso = revenue-hm_bet   
    #-- 


    return profit_DL, profit_lgbm, pred_proba_dl, pred_proba_lgbm, profil_conso,perc_dl,perc_lgbm,perc_conso, hm_bet
