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

def function_less_won_odds(df):
    """
    Return the mean for the set according to the min win_odds of each race
    """
    list_win_odds = df.win_odds[:].values
    list_min_win_odds = []
    for i in range(len(list_win_odds)):
        list_min_win_odds.append(np.min(list_win_odds[i][np.nonzero(list_win_odds[i])]))
    return np.mean(list_min_win_odds)

    
def compute_df(pred,real,model_name, df):
    """
    Return a dataframe with all needed informations to draw the evolution of our investement
    """
    L_odds_real = [df.win_odds[:][real[i]+1].tolist()[i] for i in range(len(real))]
    L_odds_pred = [df.win_odds[:][pred[i]+1].tolist()[i] for i in range(len(pred))]
    df_draw = pd.DataFrame(list(zip(df.date,pred,real,L_odds_real,L_odds_pred)), columns=['date','prediction','real_values','real_odds','pred_odds'])
    df_draw['is_same'] = np.where(df_draw['prediction']== df_draw['real_values'], True, False)
    df_draw['profit'] = np.where(df_draw['prediction']== df_draw['real_values'], df_draw['real_odds']-1, -1)
    df_draw['cumul'] = df_draw['profit'].cumsum()
    df_draw['cumul_100'] = df_draw['cumul'] + 100
    return df_draw


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


def prepare_and_split_data(X_train_init,X_test_init):
    """
    this function do the data prepartion then split and give us the good datasets according to the months we are trainning
    """

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


def multi_indexes_to_single(df):
    """
    Convert a multi indexes pandas dataframe to a single indexes pandas dataframe
    """
    new_columns = []

    for t in df.columns:
        n = t[0]+ str(t[1])
        new_columns.append(n)
    
    df.columns = new_columns
    
    return df


def metrics_perso(y_pred,y_test_value,X_TEST):
    """
    Creation of our metrics based on the profit we generate from our bet
    """
    hm_bet = len(y_pred)
    win_amount, mean_odds = how_many_do_we_win(y_pred,y_test_value,X_TEST)
    return win_amount - hm_bet


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




### Compute model results


def compute_profil(month, X_train, y_train, X_test, y_test, y_train_value, y_test_value, X_test_init):
    
    """
    Compute profit for all models (Deep Learning, LGBM, Ensemble model)
    """
    
    ###############
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




def ensemble_model(pred_dl,pred_lgbm,coef_dl):
    """
    Compute the ensemble result thanks to percentage prediciton 
    """
    new_pred = coef_dl * pred_dl + (1-coef_dl) * pred_lgbm
    return np.argmax(new_pred,axis=1)