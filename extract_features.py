import os
import pandas as pd
from config import Config
from functools import reduce


def is_place(result):
    return (result <= 3)*1

def compute_time_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month   
    return df


def compute_owner_features(df, owner='jockey'):
  
    """ function to compute statistics of the jockey and the trainer """  

    df_owner_runs = df[['race_id', f'{owner}_id']].groupby(f'{owner}_id').size().reset_index(name=f'{owner}_runs')
    df_owner_wins = df[df.won == 1][['race_id', f'{owner}_id']].groupby(f'{owner}_id').size().reset_index(name=f'{owner}_wins')
    df_owner_places = df[df.place == 1][['race_id', f'{owner}_id']].groupby(f'{owner}_id').size().reset_index(name=f'{owner}_places')

    dfs = [df, df_owner_runs, df_owner_wins, df_owner_places]
    df = reduce(lambda  left,right: pd.merge(left,right,on=[f'{owner}_id'], how='left'), dfs)  

    df[f'ratio_win_{owner}'] = df[f'{owner}_wins'] / df[f'{owner}_runs']
    df[f'ratio_place_{owner}'] = df[f'{owner}_places'] / df[f'{owner}_runs']

    del df_owner_runs, df_owner_wins, df_owner_places, dfs   
    return df


if __name__ == "__main__":

    config = Config()
    df_runs = pd.read_csv(os.path.join(config.FILE_PATH, config.runs_file_name))
    df_races = pd.read_csv(os.path.join(config.FILE_PATH, config.races_file_name))

    df_runners = df_runs.merge(df_races, on='race_id')
    df_runners['date'] = pd.to_datetime(df_races['date'])
    df_runners['place'] = list(map(is_place, df_runners['result']))
    df_runners = compute_time_features(df_runners)

    df_runners['horse_under_4'] = df_runners['horse_age'].apply(lambda a: 1 if a <= 4 else 0)
    df_runners_per_race = df_runners[['race_id', 'horse_no']].groupby('race_id').size().reset_index(name='number_of_runners')
    df_runners_age = df_runners[['race_id', 'horse_under_4']].groupby('race_id').sum().reset_index().rename(columns={'horse_under_4':'number_of_runners_under_4'})
    
    df_runners = df_runners.merge(df_runners_age, on='race_id')
    df_runners = df_runners.merge(df_runners_per_race, on='race_id')   

    df_runners['ratio_runners_under_4'] = df_runners['number_of_runners_under_4'] / df_runners['number_of_runners']

    del df_runs, df_races
    del df_runners_per_race, df_runners_age

    df_runners = compute_owner_features(df_runners, owner='jockey')
    df_runners = compute_owner_features(df_runners, owner='trainer')        

    df_runners.to_csv(os.path.join(config.FILE_PATH, config.output_file_name))