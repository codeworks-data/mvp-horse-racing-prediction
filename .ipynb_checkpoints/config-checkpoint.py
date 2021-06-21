class Config:
    def __init__(self):
        
        self.FILE_PATH = "data"
        self.runs_file_name = "runs.csv"
        self.races_file_name = "races.csv"
        self.output_file_name = "runners_stats.csv"   # created after feature engineering
        self.train_file_name = "train_runners.h5"    # data for training
        self.test_file_name = "test_runners.h5"      # data for testing models

        self.won_label = "won"
        self.place_label = "place"
        self.date_test = "2005-01-01"
        self.pivot = True,
        self.pivot_column = ('horse_no')
        
        #features corresponding to the run only
        self.index_pivot_table = ('race_id', 'date', 'race_no', 'venue', 'config', 'surface', 'distance', 'going', 'horse_ratings', 'prize', 'race_class') 