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
        self.model_file_name = "baseline_model.joblib"
        self.date_test = "2005-01-01"
        self.pivot = True,
        self.pivot_column = ('horse_no')
        self.index_pivot_table = ('race_id', 'date', 'venue', 'config', 'surface', 'distance', 'going', 'horse_ratings', 'prize', 'race_class')
        self.list_feats = ['result',
                        'horse_age',
                        'horse_country',
                        'horse_type',
                        'horse_rating',
                        'horse_gear',
                        'declared_weight',
                        'actual_weight',
                        'draw',
                        'win_odds']
