import os
import pandas as pd
from config import Config


def sort_result(element):
    """
    Place the label columns 'result' to the end of the dataframe
    """
    if element[0] == 'result':
        return 100 + element[1] # to make sure results are put near the end
    else:
        return element[1] 


def create_train_test(df, date_test):
    """
    Split train and test data according to the date_test.
    This is a month split.
    """
    train_race_ids = df[df.date < date_test]['race_id'].unique().tolist()
    train_data = df[df.race_id.isin(train_race_ids)]
    test_data = df[~(df.race_id.isin(train_race_ids))]
    return train_data, test_data


if __name__=='__main__':

    config = Config()
    
    #open the dataframe with all new features from extract_features
    df_runners = pd.read_csv(os.path.join(config.FILE_PATH, config.output_file_name))
    df_runners[config.won_label] = df_runners[config.won_label].fillna(0).astype(int)

    INDEX = config.index_pivot_table
    PIVOT_COLUMN = config.pivot_column
    FEATURES = df_runners.columns[len(INDEX):].tolist()


    df_runners_pivot = pd.pivot_table(df_runners, index=INDEX, columns=PIVOT_COLUMN, values=FEATURES).reset_index()

    #rearrange columns to put 'result' at the end of the dataframe
    rearranged_columns = sorted(list(df_runners_pivot.columns.values)[len(INDEX):], key=sort_result)
    rearranged_columns = list(df_runners_pivot.columns.values)[:len(INDEX)] + rearranged_columns

    df_runners_pivot = df_runners_pivot[rearranged_columns].fillna(0)

    if config.pivot:
        #do the split between train and test set
        train_data, test_data = create_train_test(df_runners_pivot, config.date_test)

        #save train and test set to hdf files because of the multindex pandas dataframe
        train_data.to_hdf(os.path.join(config.FILE_PATH, config.train_file_name), key='features')
        test_data.to_hdf(os.path.join(config.FILE_PATH, config.test_file_name), key='features')

