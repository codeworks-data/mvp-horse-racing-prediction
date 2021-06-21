import os
import numpy as np
import pandas as pd
from config import Config
from functools import reduce
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder




#### Functions used to compute our features ###

def is_place(result):
    """
    return 1 or 0 if the horse is top 3
    """
    return (result <= 3)*1


def compute_horse_ratings(val):
    """
    Grouping horse rating (from 40 to 120) and assigning a number from 0 to 4
    """
    if val in ['40-15','40-0','40-10','40-20']:
        return 0
    if val in ['60-40','65-40','60-35']:
        return 1
    if val in ['80-60','85-60','80-55','75-55']:
        return 2
    if val in ['100-80','105-80','95-75','110-85','115-90','90+','90-70','95+','110-90','100+','100-75','115-95','120-95','80+','85+','105-85','95-70','110-80','120-100']:
        return 3
    if val in ['G']:
        return 4
   
    #else:
        #return val
        
        
### Functions which change the dataframe either create new columns or change existing columns

def compute_time_feats(df):
    """
    Create 3 columns for year, month and quarter
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = (df['month'] + 2)//3
    return df

def compute_race_feats(df):  
    """
    Create other features : - if horse is under 4 yo
                            - the number of runners by race (del)
                            - the number of horse below 4yo in the race (del)
                            - Compute the ration of horse below 4 yo 
    """   
    
    df['horse_under_4'] = df['horse_age'].apply(lambda a: 1 if a <= 4 else 0)
    df_runners_per_race = df[['race_id', 'horse_no']].groupby('race_id').size().reset_index(name='number_of_runners')
    df_runners_age = df[['race_id', 'horse_under_4']].groupby('race_id').sum().reset_index().rename(columns={'horse_under_4':'number_of_runners_under_4'})  
    
    df = pd.merge(df, df_runners_per_race, on='race_id', how='left')
    df = pd.merge(df, df_runners_age, on='race_id', how='left') 
    df['ratio_runners_under_4'] = df['number_of_runners_under_4'] / df['number_of_runners']
    
    del df_runners_per_race, df_runners_age
    
    return df


def compute_horse_feats(df, group_col='venue'):
    """
    Compute horse features :    - compute if horse is placed
                                - last place of the horse
                                - last draw position 
                                - rest time 
                                - diff weigth between two races
                                - diff distance between two races
                                - bind gear to numbers
                                - compute the horse ratings
                                - numbers of : runs, win, placed
                                - ratio of win and placed
                                - ratio of win and placed according to the venue
                                
    """
    
    df = df.sort_values('date')  
    # last results
    df['place'] = list(map(is_place, df['result']))
    df['last_place'] =   list(map(is_place, df.groupby('horse_id')['result'].shift()))
    df['last_place'] = df['last_place'].fillna(0)  
    df['last_draw'] = df.groupby('horse_id')['draw'].shift().fillna(14)    
    
    #rest time
    df['horse_rest_time'] = (df['date'] - df.groupby('horse_id')['date'].shift()).dt.days
    df['horse_rest_lest14'] = (df['horse_rest_time'] <= 14)*1
    df['horse_rest_over35'] = (df['horse_rest_time'] >= 35)*1
    
    #changement between races
    df['diff_declared_weight'] = df['declared_weight'] - df.groupby('horse_id')['declared_weight'].shift()
    df['diff_distance'] = df['distance'] - df.groupby('horse_id')['distance'].shift()
    
    #some binding (gear and ratings)
    df['horse_gear'] = df['horse_gear'].apply(lambda x: 1 if x=='--' else 0)
    df['horse_ratings'] = df['horse_ratings'].apply(compute_horse_ratings)
    
    #numbers of runs/wins/placed
    df['horse_runs'] = df.sort_values('date').groupby(['horse_id']).cumcount()
    df['horse_wins'] = df.sort_values('date').groupby(['horse_id'])['won'].cumsum().sub(df.won)
    df['horse_places'] = df.sort_values('date').groupby(['horse_id'])['place'].cumsum().sub(df.place)
    
    #some ratios
    try:
        df['ratio_win_horse'] = df['horse_wins'] / df['horse_runs']
        df['ratio_place_horse'] = df['horse_places'] / df['horse_runs']    
        
    except ZeroDivisionError:
        return 0
    
    #ratio according to the venue (field)
    if group_col:
        
        df[f'horse_{group_col}_runs'] = df.sort_values('date').groupby(['horse_id', group_col]).cumcount()
        df[f'horse_{group_col}_wins'] = df.sort_values('date').groupby(['horse_id', group_col])['won'].cumsum().sub(df.won)
        df[f'horse_{group_col}_places'] = df.sort_values('date').groupby(['horse_id', group_col])['place'].cumsum().sub(df.place)
        
        try:
            df[f'ratio_win_horse_{group_col}'] = df[f'horse_{group_col}_wins'] / df[f'horse_{group_col}_runs']
            df[f'ratio_place_horse_{group_col}'] = df[f'horse_{group_col}_places'] / df[f'horse_{group_col}_runs']
            
        except ZeroDivisionError:
            return 0 
   
    return df


def compute_owner_feats(df, owner=None, group_col='venue'):
    """
    function to compute statistics of the owner (jockey and the trainer) for runs/wins/placed depending on the group_col (venue or distance)
    """  

    #compute numbers of runs, wins, placed 
    df[f'{owner}_runs'] = df.sort_values(['date', 'race_no']).groupby(f'{owner}_id').cumcount()
    df[f'{owner}_wins'] = df.sort_values(['date', 'race_no']).groupby(f'{owner}_id')['won'].cumsum().sub(df.won)
    df[f'{owner}_places'] = df.sort_values(['date', 'race_no']).groupby(f'{owner}_id')['place'].cumsum().sub(df.place)
    
    #compute ratios
    try:
        df[f'ratio_win_{owner}'] = df[f'{owner}_wins'] / df[f'{owner}_runs']
        df[f'ratio_place_{owner}'] = df[f'{owner}_places'] / df[f'{owner}_runs']
        
    except ZeroDivisionError:
        return 0
    
    #if group_col is present adding new specific columns  
    if group_col:
        
        df[f'{owner}_{group_col}_runs'] = df.sort_values(['date', 'race_no']).groupby([f'{owner}_id', group_col]).cumcount()
        df[f'{owner}_{group_col}_wins'] = df.sort_values(['date', 'race_no']).groupby([f'{owner}_id', group_col])['won'].cumsum().sub(df.won)
        df[f'{owner}_{group_col}_places'] = df.sort_values(['date', 'race_no']).groupby([f'{owner}_id', group_col])['place'].cumsum().sub(df.place)
        
        try:
            df[f'ratio_win_{owner}_{group_col}'] = df[f'{owner}_{group_col}_wins'] / df[f'{owner}_{group_col}_runs']
            df[f'ratio_place_{owner}_{group_col}'] = df[f'{owner}_{group_col}_places'] / df[f'{owner}_{group_col}_runs']
            
        except ZeroDivisionError:
            return 0    
        
    return df



def compute_previous_std_ranking_to_winner(df, col='result'):
    """
    Compute the standard rank of the horse on his last 3 races giving us an overview of his state of form
    """

    df = df.sort_values(['date','race_no'])
    df['rank1'] = df.groupby('horse_id')[f'{col}'].shift()
    df['rank2'] = df.groupby('horse_id')[f'{col}'].shift(2)
    df['rank3'] = df.groupby('horse_id')[f'{col}'].shift(3)
    df['horse_std_rank'] = df.apply(lambda x: (  (  (x.rank1 - 1)**2 + (x.rank2 - 1)**2 + (x.rank3 - 1)**2    ) / 3) **(0.5)  ,axis=1).fillna(0)
    df.drop(['rank1','rank2','rank3'],axis=1, inplace=True)
    
    return df



def compute_combinaison_feats(df, owner_1, owner_2, hippo = None):
    """
    compute crossed informations on win/runs/placed between owner_1 and owner_2 which can be jockey/horse/trainer
    """
    
    
    df = df.sort_values(['date','race_no'])
    df[f'runs_{owner_1}_{owner_2}'] = df.sort_values(['date','race_no']).groupby([f'{owner_1}_id', f'{owner_2}_id']).cumcount()
    df[f'wins_{owner_1}_{owner_2}'] = df.sort_values(['date','race_no']).groupby([f'{owner_1}_id', f'{owner_2}_id']).won.cumsum().sub(df.won)
    
    df[f'ratio_win_{owner_1}_{owner_2}'] = (df[f'wins_{owner_1}_{owner_2}'] / df[f'runs_{owner_1}_{owner_2}'] )
    
    df[f'places_{owner_1}_{owner_2}'] = df.sort_values(['date','race_no']).groupby([f'{owner_1}_id', f'{owner_2}_id']).place.cumsum().sub(df.place)
    df[f'ratio_place_{owner_1}_{owner_2}'] = (df[f'places_{owner_1}_{owner_2}'] / df[f'runs_{owner_1}_{owner_2}'] )
    df[f'ratio_place_{owner_1}_{owner_2}'] = df[f'places_{owner_1}_{owner_2}'] / df[f'runs_{owner_1}_{owner_2}']
    df[f'first_second_{owner_2}'] = (df[f'runs_{owner_1}_{owner_2}'] <= 1).astype(int)
    df[f'same_last_{owner_2}'] = (df.groupby([f'{owner_1}_id'])[f'{owner_2}_id'].shift() == df[f'{owner_2}_id']).astype(int)
    
    if hippo:
        df[f'runs_{owner_1}_{owner_2}_{hippo}'] = df.sort_values(['date','race_no']).groupby([f'{owner_1}_id', f'{owner_2}_id', f'{hippo}']).cumcount()
        df[f'wins_{owner_1}_{owner_2}_{hippo}'] = df.sort_values(['date','race_no']).groupby([f'{owner_1}_id', f'{owner_2}_id', f'{hippo}']).won.cumsum().sub(df.won)
        df[f'ratio_win_{owner_1}_{owner_2}_{hippo}'] = (df[f'wins_{owner_1}_{owner_2}_{hippo}'] / df[f'runs_{owner_1}_{owner_2}_{hippo}'] )
        df[f'places_{owner_1}_{owner_2}_{hippo}'] = df.sort_values(['date','race_no']).groupby([f'{owner_1}_id', f'{owner_2}_id', f'{hippo}']).place.cumsum().sub(df.place)
        df[f'ratio_place_{owner_1}_{owner_2}_{hippo}'] = df[f'places_{owner_1}_{owner_2}_{hippo}'] / df[f'runs_{owner_1}_{owner_2}_{hippo}']
    
    return df


def compute_owner_prize_feats(df, owner=None):
  
    """
    Compute the gain for the owner, it can be either horse, jockey, or trainer
    """  
    
    df = df.sort_values(['date','race_no']) 
    df['prize_race'] = df['won']*df['prize'].fillna(0)
    
    #compute the cumulated prize before the race 
    df[f'prize_{owner}_cumul'] = df.groupby(f'{owner}_id').prize.cumsum().sub(df.prize)
    
    #compute some ratio with the gain and runs/wins
    df[f'avg_prize_wins_{owner}'] = df[f'prize_{owner}_cumul']/df[f'{owner}_wins'] 
    df[f'avg_prize_runs_{owner}'] = df[f'prize_{owner}_cumul']/df[f'{owner}_runs']
    
    #we drop this columns because it's an important leak because the 'won' columns was involed
    df.drop('prize_race', axis=1, inplace=True)
    
    return df


def fillna(df, na_cols, na_value=0):
    """
    fill na values to good default values
    """
    
    for col in na_cols:
        df[col] = df[col].fillna(na_value)
        df.replace([np.inf, -np.inf], 0, inplace=True)

    df.horse_country.fillna(df.horse_country.mode()[0], inplace=True)
    df.horse_type.fillna(df.horse_type.mode()[0], inplace=True)

    return df

def encode_categ_column(df):
    """
    Encode columns with either label or ordinal encoding
    """
    
    horse_type_encoder = LabelEncoder() 
    horse_country_encoder = LabelEncoder()
    venue_encoder =  LabelEncoder()

    going_encoder = OrdinalEncoder()
    config_encoder = OrdinalEncoder()

    df['horse_type'] = horse_type_encoder.fit_transform(df['horse_type'])
    df['horse_country'] = horse_country_encoder.fit_transform(df['horse_country'])
    df['venue'] = venue_encoder.fit_transform(df['venue'])

    df['going'] = going_encoder.fit_transform(df['going'].values.reshape(-1, 1))
    df['config'] = config_encoder.fit_transform(df['config'].values.reshape(-1, 1))

    return df 


if __name__ == "__main__":

    config = Config()
    
    #retrieve the date needed, we have races and runs informations
    df_runs = pd.read_csv(os.path.join(config.FILE_PATH, config.runs_file_name))
    df_races = pd.read_csv(os.path.join(config.FILE_PATH, config.races_file_name))

    #drop if we have draw place above 14
    strange_draw_index = df_runs[df_runs['draw'] > 14].index
    df_runs = df_runs.drop(strange_draw_index)

    #drop columns where information are given after the race
    after_race_cols = 'sec_time|time|place_combination|place_dividend|win_combination|win_dividend|lengths_behind|position_sec|behind_sec'
    df_runs = df_runs[df_runs.columns.drop(list(df_runs.filter(regex=after_race_cols)))]

    #create the merge with both initial dataframe
    df_runners = pd.merge(df_races, df_runs, on='race_id', how='left')

    #Start creating new features
    
    df_runners = compute_time_feats(df_runners)
    df_runners = compute_race_feats(df_runners)
    
    print("Computing horse feats ...")
    df_runners = compute_horse_feats(df_runners, group_col='venue')
    df_runners = compute_horse_feats(df_runners, group_col='surface')
    df_runners = compute_horse_feats(df_runners, group_col='going')
    df_runners = compute_horse_feats(df_runners, group_col='distance')
    df_runners = compute_horse_feats(df_runners, group_col='quarter')
    
    print("Computing owner feats ...")
    df_runners = compute_owner_feats(df_runners, owner='jockey')
    df_runners = compute_owner_feats(df_runners, owner='trainer')
    df_runners = compute_owner_feats(df_runners, owner='jockey', group_col='distance')
    df_runners = compute_previous_std_ranking_to_winner(df_runners, col='result')
    
    print("Computing combinaison feats ...")
    df_runners = compute_combinaison_feats(df_runners, owner_1 = 'horse', owner_2 = 'jockey', hippo = 'venue')
    df_runners = compute_combinaison_feats(df_runners, owner_1 = 'horse', owner_2 = 'trainer', hippo = 'venue')
    
    print("Computing owner prize ...")
    df_runners = compute_owner_prize_feats(df_runners, owner='horse')
    df_runners = compute_owner_prize_feats(df_runners, owner='trainer')
    df_runners = compute_owner_prize_feats(df_runners, owner='jockey')

    print("Filling NaN values ...")
    na_cols = df_runners.filter(regex="ratio|runs|wins|places|prize|avg|diff_").columns.tolist()
    df_runners = fillna(df_runners, na_cols)
    
    print("Encoding categorical columns ...")
    df_runners = encode_categ_column(df_runners)

    #save our final data into a csv file
    df_runners.to_csv(os.path.join(config.FILE_PATH, config.output_file_name), index=False)