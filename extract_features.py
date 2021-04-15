import os
import numpy as np
import pandas as pd
from config import Config
from functools import reduce

def is_place(result):
    return (result <= 3)*1

def compute_time_feats(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = (df['month'] + 2)//3
    
    return df

def compute_race_feats(df):  
    df['horse_under_4'] = df['horse_age'].apply(lambda a: 1 if a <= 4 else 0)
    df_runners_per_race = df[['race_id', 'horse_no']].groupby('race_id').size().reset_index(name='number_of_runners')
    df_runners_age = df[['race_id', 'horse_under_4']].groupby('race_id').sum().reset_index().rename(columns={'horse_under_4':'number_of_runners_under_4'})
    
    df = pd.merge(df, df_runners_per_race, on='race_id', how='left')
    df = pd.merge(df, df_runners_age, on='race_id', how='left')
    
    df['ratio_runners_under_4'] = df['number_of_runners_under_4'] / df['number_of_runners']
    del df_runners_per_race, df_runners_age
    
    return df

def compute_horse_feats(df, group_col='venue'):
    df = df.sort_values('date')  
    df['place'] = list(map(is_place, df['result']))
    df['last_place'] =   list(map(is_place, df.groupby('horse_id')['result'].shift()))
    df['last_draw'] = df.groupby('horse_id')['draw'].shift()    
    df['horse_rest_time'] = (df['date'] - df.groupby('horse_id')['date'].shift()).dt.days
    df['horse_rest_lest14'] = (df['horse_rest_time'] <= 14)*1
    df['horse_rest_over35'] = (df['horse_rest_time'] >= 35)*1
    df['diff_declared_weight'] = df['declared_weight'] - df.groupby('horse_id')['declared_weight'].shift()
    df['diff_distance'] = df['distance'] - df.groupby('horse_id')['distance'].shift()
    
    df['horse_runs'] = df.sort_values('date').groupby(['horse_id']).cumcount()
    df['horse_wins'] = df.sort_values('date').groupby(['horse_id'])['won'].cumsum().sub(df.won)
    df['horse_places'] = df.sort_values('date').groupby(['horse_id'])['place'].cumsum().sub(df.place)
    
    try:
        df['ratio_win_horse'] = df['horse_wins'] / df['horse_runs']
        df['ratio_place_horse'] = df['horse_places'] / df['horse_runs']    
        
    except ZeroDivisionError:
        return 0
    
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
  
    """ function to compute statistics of the jockey and the trainer """  

    df[f'{owner}_runs'] = df.sort_values(['date', 'race_no']).groupby(f'{owner}_id').cumcount()
    df[f'{owner}_wins'] = df.sort_values(['date', 'race_no']).groupby(f'{owner}_id')['won'].cumsum().sub(df.won)
    df[f'{owner}_places'] = df.sort_values(['date', 'race_no']).groupby(f'{owner}_id')['place'].cumsum().sub(df.place)
    
    try:
        df[f'ratio_win_{owner}'] = df[f'{owner}_wins'] / df[f'{owner}_runs']
        df[f'ratio_place_{owner}'] = df[f'{owner}_places'] / df[f'{owner}_runs']
        
    except ZeroDivisionError:
        return 0
    
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



def compute_std_ranking_to_1(df, col='result'):

    df = df.sort_values(['date','race_no'])
    df['rank1'] = df.groupby('horse_id')[f'{col}'].shift()
    df['rank2'] = df.groupby('horse_id')[f'{col}'].shift(2)
    df['rank3'] = df.groupby('horse_id')[f'{col}'].shift(3)
    df['horse_std_rank'] = df.apply(lambda x: (  (  (x.rank1 - 1)**2 + (x.rank2 - 1)**2 + (x.rank3 - 1)**2    ) / 3) **(0.5)  ,axis=1).fillna(0)
    df.drop(['rank1','rank2','rank3'],axis=1, inplace=True)
    
    return df



def compute_combinaison_feats(df, owner_1, owner_2, hippo = None):
    df = df.sort_values(['date','race_no'])
    df[f'total_runs_{owner_1}_{owner_2}'] = df.sort_values(['date','race_no']).groupby([f'{owner_1}_id', f'{owner_2}_id']).cumcount()
    df[f'total_wins_{owner_1}_{owner_2}'] = df.sort_values(['date','race_no']).groupby([f'{owner_1}_id', f'{owner_2}_id']).won.cumsum()
    
    df[f'ratio_win_{owner_1}_{owner_2}'] = (df[f'total_wins_{owner_1}_{owner_2}'] / df[f'total_runs_{owner_1}_{owner_2}'] ).fillna(0)
    
    df[f'total_place_{owner_1}_{owner_2}'] = df.sort_values(['date','race_no']).groupby([f'{owner_1}_id', f'{owner_2}_id']).place.cumsum()
    df[f'ratio_place_{owner_1}_{owner_2}'] = (df[f'total_place_{owner_1}_{owner_2}'] / df[f'total_runs_{owner_1}_{owner_2}'] ).fillna(0)
    df[f'ratio_place_{owner_1}_{owner_2}'].loc[(~np.isfinite(df[f'ratio_place_{owner_1}_{owner_2}'])) & df[f'ratio_place_{owner_1}_{owner_2}'].notnull()] = np.nan
    df[f'ratio_place_{owner_1}_{owner_2}'] = df[f'ratio_place_{owner_1}_{owner_2}'].fillna(0)
    df[f'first_second_{owner_2}'] = (df[f'total_runs_{owner_1}_{owner_2}'] <= 1).astype(int)
    df[f'same_last_{owner_2}'] = (df.groupby([f'{owner_1}_id'])[f'{owner_2}_id'].shift() == df[f'{owner_2}_id']).astype(int)
    
    if hippo:
        df[f'total_runs_{owner_1}_{owner_2}_{hippo}'] = df.sort_values(['date','race_no']).groupby([f'{owner_1}_id', f'{owner_2}_id', f'{hippo}']).cumcount()
        df[f'total_wins_{owner_1}_{owner_2}_{hippo}'] = df.sort_values(['date','race_no']).groupby([f'{owner_1}_id', f'{owner_2}_id', f'{hippo}']).won.cumsum()
        df[f'ratio_win_{owner_1}_{owner_2}_{hippo}'] = (df[f'total_wins_{owner_1}_{owner_2}_{hippo}'] / df[f'total_runs_{owner_1}_{owner_2}_{hippo}'] ).fillna(0)
        df[f'total_place_{owner_1}_{owner_2}_{hippo}'] = df.sort_values(['date','race_no']).groupby([f'{owner_1}_id', f'{owner_2}_id', f'{hippo}']).place.cumsum()
        df[f'ratio_place_{owner_1}_{owner_2}_{hippo}'] = (df[f'total_place_{owner_1}_{owner_2}_{hippo}'] / df[f'total_runs_{owner_1}_{owner_2}_{hippo}'] ).fillna(0)
        df[f'ratio_place_{owner_1}_{owner_2}_{hippo}'].loc[(~np.isfinite(df[f'ratio_place_{owner_1}_{owner_2}'])) 
                                                           & df[f'ratio_place_{owner_1}_{owner_2}_{hippo}'].notnull()] = np.nan
        df[f'ratio_place_{owner_1}_{owner_2}_{hippo}'] = df[f'ratio_place_{owner_1}_{owner_2}_{hippo}'].fillna(0) 
    
    return df


def compute_owner_gain_feats(df, owner=None):
  
    """ function to compute gain features of the jockey and the trainer """  
    
    df = df.sort_values(['date','race_no']) 
    df['gain_for_this_race'] = df['won']*df['prize'].fillna(0)
    df[f'gain_{owner}_cumul'] = df.groupby(f'{owner}_id').gain_for_this_race.cumsum().sub(df.gain_for_this_race)
    df[f'avg_prize_wins_{owner}'] = (df[f'gain_{owner}_cumul']/df[f'{owner}_wins']).fillna(0) #on prends en compte le décalage pour avoir une valeur d'avant course
    df[f'avg_prize_runs_{owner}'] = (df[f'gain_{owner}_cumul']/df[f'{owner}_runs']).fillna(0) #on prends en compte le décalage pour avoir une valeur d'avant course
    df.drop('gain_for_this_race',axis=1,inplace=True)
    
    return df


def fillna(df, na_cols, na_value=0):
    for col in na_cols:
        df[col] = df[col].fillna(na_value)
    return df


if __name__ == "__main__":

    config = Config()
    df_runs = pd.read_csv(os.path.join(config.FILE_PATH, config.runs_file_name))
    df_races = pd.read_csv(os.path.join(config.FILE_PATH, config.races_file_name))

    df_runners = pd.merge(df_races, df_runs, on='race_id', how='left')

    df_runners = compute_time_feats(df_runners)
    df_runners = compute_race_feats(df_runners)
    df_runners = compute_horse_feats(df_runners, group_col='venue')
    df_runners = compute_horse_feats(df_runners, group_col='surface')
    df_runners = compute_horse_feats(df_runners, group_col='going')
    df_runners = compute_horse_feats(df_runners, group_col='distance')
    df_runners = compute_horse_feats(df_runners, group_col='quarter')
    df_runners = compute_owner_feats(df_runners, owner='jockey')
    df_runners = compute_owner_feats(df_runners, owner='trainer')
    df_runners = compute_owner_feats(df_runners, owner='jockey', group_col='distance')
    df_runners = compute_std_ranking_to_1(df_runners, col='result')
    df_runners = compute_combinaison_feats(df_runners, owner_1 = 'horse', owner_2 = 'jockey', hippo = 'venue')
    df_runners = compute_combinaison_feats(df_runners, owner_1 = 'horse', owner_2 = 'trainer', hippo = 'venue')
    df_runners = compute_owner_gain_feats(df_runners, owner='horse')
    df_runners = compute_owner_gain_feats(df_runners, owner='trainer')
    df_runners = compute_owner_gain_feats(df_runners, owner='jockey')

    na_cols = df_runners.filter(regex="ratio|runs|wins|places").columns.tolist()
    df_runners = fillna(df_runners, na_cols)  

    df_runners.to_csv(os.path.join(config.FILE_PATH, config.output_file_name), index=False)