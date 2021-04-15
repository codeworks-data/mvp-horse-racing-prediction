import os
import pandas as pd
from config import Config

def create_dataset(df, date_test):
    train_race_ids = df[df.date < date_test]['race_id'].unique().tolist()
    train_data = df[df.race_id.isin(train_race_ids)]
    test_data = df[~(df.race_id.isin(train_race_ids))]
    return train_data, test_data


if __name__=='__main__':

    config = Config()
    df_runners = pd.read_csv(os.path.join(config.FILE_PATH, config.output_file_name))
    df_runners[config.won_label] = df_runners[config.won_label].fillna(0).astype(int)
    train_data, test_data = create_dataset(df_runners, config.date_test)

    train_data.to_csv(os.path.join(config.FILE_PATH, config.train_file_name), index=False)
    test_data.to_csv(os.path.join(config.FILE_PATH, config.test_file_name), index=False)

