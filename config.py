class Config:
    def __init__(self):
        self.FILE_PATH = "data"
        self.runs_file_name = "runs.csv"
        self.races_file_name = "races.csv"
        self.output_file_name = "runners_stats.csv"

        self.won_label = "won"
        self.place_label = "place"
        self.model_file_name = "baseline_model.joblib"