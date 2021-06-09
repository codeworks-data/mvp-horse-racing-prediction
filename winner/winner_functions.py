import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt
from matplotlib import pyplot
import keras
import joblib
from lightgbm import plot_importance
from lightgbm import LGBMClassifier



def how_many_do_we_win(y_pred_value,y_real_value,df):
    """return how many do we win and the mean values of the win odds"""
    win = 0
    List_odds = []
    for i in range(len(y_pred_value)):
        if y_pred_value[i] == y_real_value[i]:
            val = df.win_odds[:][y_pred_value[i]+1].tolist()[i] #+1 because we need horse_no and not indices from a list
            win = win + val
            List_odds.append(val)
    return win, np.mean(List_odds)

def function_less_won_odds(df):
    """this function return the mean of each less win_odds"""
    L = df.win_odds[:].values
    M = []
    for i in range(len(L)):
        M.append(np.min(L[i][np.nonzero(L[i])]))
    return np.mean(M)

    
def compute_df(pred,real,model_name, df):
    """this function return a datframe with all needed information to draw the evolution of our investement"""
    L_odds_real = [df.win_odds[:][real[i]+1].tolist()[i] for i in range(len(real))]
    L_odds_pred = [df.win_odds[:][pred[i]+1].tolist()[i] for i in range(len(pred))]
    Z = pd.DataFrame(list(zip(df.date,pred,real,L_odds_real,L_odds_pred)), columns=['date','prediction','real_values','real_odds','pred_odds'])
    Z['is_same'] = np.where(Z['prediction']== Z['real_values'], True, False)
    Z['profit'] = np.where(Z['prediction']== Z['real_values'], Z['real_odds']-1, -1)
    Z['cumul'] = Z['profit'].cumsum()
    Z['cumul_100'] = Z['cumul'] + 100
    return Z


def draw_evolution(df):
    """draw the evolution according to the date"""
    plt.figure(figsize=(10,10)).suptitle('evolution of the profit with a 100 euro bet', fontsize=20)
    plt.ylabel('profit', fontsize=18)
    plt.xlabel('date', fontsize=16)
    plt.xticks(rotation=90)
    plt.plot(df.date, df.cumul_100)

def draw_evolution_race(df,model_name):
    """draw the evolution according to the number of races"""
    plt.figure(figsize=(10,10)).suptitle(f'Evolution of the profit with $100 for {model_name}', fontsize=20)
    plt.ylabel('profit ($)', fontsize=18)
    plt.xlabel('num bets', fontsize=16)
    plt.xticks(rotation=0,fontsize=15)
    plt.yticks(rotation=0,fontsize=15)
    plt.plot(df.index, df.cumul_100)

    
def create_X_TEST(df_init):
    'return a dataframe used to retrieve win_odds and place_odds'
    INDEX = ['race_id', 'date', 'race_no', 'venue', 'config', 'surface', 'distance', 'going', 'horse_ratings', 'prize', 'race_class']
    features = ['draw','place_odds','win_odds','result']
    return df_init[INDEX + features]



def create_x_and_y(df):
    'function which return 2 dataframe for the features and labels'
    data = df
    #all first columns will be use for X
    X = data[data.columns[:-14]] 
    ss = preprocessing.StandardScaler()
    X = pd.DataFrame(ss.fit_transform(X),columns = X.columns)

    #all 14 last columns will be used for y we just use 1 if the horse win the race
    y = data[data.columns[-14:]].applymap(lambda x: 1.0 if x==1 else 0.0) 

    return X,y


def prepare_and_split_data(X_train_init,X_test_init):
    "this function do the data prepartion, split and give us the good datasets according to the months we are trainning"

    #print("shape of the train set :", X_train_init.shape)
    #print("shape of the test set :", X_test_init.shape)

    #dropping some columns
    Drop = ['race_id','race_no','date','won','place']
    X_tr = X_train_init.drop(Drop,axis=1,level=0)
    X_te = X_test_init.drop(Drop,axis=1,level=0)

    #how many features
    L = []
    for col in X_tr.columns.tolist():
        L.append(col[0])

    #print(f"We only keep {len(set(L))} columns in totals")

    #We slipt each dataset between features and labels
    X_train, y_train = create_x_and_y(X_tr)
    print("shape of the x_train: ", X_train.shape)
    print("shape of the y_train: ", y_train.shape)

    X_test, y_test = create_x_and_y(X_te)
    print("shape of the x_train: ", X_test.shape)
    print("shape of the y_train: ", y_test.shape)

    #compute y_train_value : show the winner for each races
    y_test_value = y_test.values.tolist()
    y_test_value = np.array([np.argmax(t) for t in y_test_value])
    #compute y_test_value
    y_train_value = y_train.values.tolist()
    y_train_value = np.array([np.argmax(t) for t in y_train_value])

    return X_train, y_train, X_test, y_test, y_train_value, y_test_value, X_test_init



def multi_indexes_to_single(dataframe):
    "this function convert a multi indexes pandas dataframe to a single indexes pandas dataframe"

    new_columns = []

    for t in dataframe.columns:
        n = t[0]+ str(t[1])
        new_columns.append(n)
    
    dataframe.columns = new_columns
    
    return dataframe


def ensemble_model(pred_dl,pred_lgbm,coef_dl):
    new_pred = coef_dl * pred_dl + (1-coef_dl) * pred_lgbm
    return np.argmax(new_pred,axis=1)



 
def metrics_perso(y_pred,y_test_value,X_TEST):
    hm_bet = len(y_pred)

    a,b = how_many_do_we_win(y_pred,y_test_value,X_TEST)

    return a-hm_bet


############    TRAIN 





def train_dl(num_neutron,batch_size,epoch,X_train,y_train,X_test,y_test):
    "This function will allow us to train our deep learning model"
    
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







####### ALL RESULT


def compute_profil(month, X_train, y_train, X_test, y_test, y_train_value, y_test_value, X_test_init):
    
    
    # DEEP LEARNING
    
    model = keras.models.load_model(f'model/winner_DL_{month}.h5')
    y_pred_dl = model.predict_classes(X_test)
    
    # Compute the deep learning profit
    
    #--
    y_test_value = y_test.values.tolist()
    y_test_value = np.array([np.argmax(t) for t in y_test_value])

    X_TEST = create_X_TEST(X_test_init)

    good_guesses = np.equal(y_pred_dl, y_test_value).sum()
    hm_bet = len(y_pred_dl)
    
    perc_dl = round((good_guesses / hm_bet) * 100,2)

    a,b = how_many_do_we_win(y_pred_dl,y_test_value,X_TEST)

    profit_DL = a-hm_bet
    
    #--
    
    
    ###############
    
    # LGBM


    filename = f'model/winner_lgbm_{month}'

    #load saved model
    lgbm = joblib.load(filename)
    
    y_pred_lgbm = lgbm.predict(X_test)
    
    # Compute the lgbm profit
    
    #--    
    
    good_guesses = np.equal(y_pred_lgbm, y_test_value).sum()
    hm_bet = len(y_pred_lgbm)
    
    perc_lgbm = round((good_guesses / hm_bet) * 100,2)

    a,b = how_many_do_we_win(y_pred_lgbm,y_test_value,X_TEST)

    profil_lgbm = a-hm_bet
    
    #-- 
    
    
    
    
    
    
    ###############
    
    #Ensemble Model
    
    pred_proba_dl = model.predict_proba(X_test)
    pred_proba_xgb = lgbm.predict_proba(X_test)
    
    pred_classes = ensemble_model(pred_proba_dl,pred_proba_xgb,0.3)
    
    # Compute the ensemble profit
    
    #--      

    y_pred = pred_classes
    good_guesses = np.equal(y_pred, y_test_value).sum()
    hm_bet = len(y_pred)
    
    perc_conso = round((good_guesses / hm_bet) * 100,2)

    a,b = how_many_do_we_win(y_pred,y_test_value,X_TEST)

    profil_ensemble = a-hm_bet
    
    #-- 
    
    
    return profit_DL, profil_lgbm, pred_proba_dl, pred_proba_xgb, profil_ensemble,perc_dl,perc_lgbm,perc_conso,hm_bet

