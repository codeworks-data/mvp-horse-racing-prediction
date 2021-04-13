class Config:
    def __init__(self):
        self.FILE_PATH = "data"
        self.runs_file_name = "runs.csv"
        self.races_file_name = "races.csv"
        self.output_file_name = "runners_stats.csv"   # created after feature engineering
        self.train_file_name = "train_runners.csv"    # data for training
        self.test_file_name = "test_runners.csv"      # data for testing models

        self.won_label = "won"
        self.place_label = "place"
        self.model_file_name = "baseline_model.joblib"
        self.year_test = 2004