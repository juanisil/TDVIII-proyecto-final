import pandas as pd


class DatasetProcessor:
    """ Class to process the dataset """
    def __init__(self, file_path):
        self.file_path = file_path
        self.ds = None
        self.date_2_3 = None
        self.train = None
        self.test = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.process()

    def load_and_process_data(self):
        self.ds = pd.read_csv(self.file_path)
        self.ds = self.ds[self.ds["target"] != 0]
        self.ds["date"] = pd.to_datetime(self.ds["date"])
        dates = self.ds["date"]
        self.date_2_3 = dates.mean() + 2 / 3 * dates.std()

    def split_data(self):

        pairs = self.ds[["player_1", "player_2"]].drop_duplicates()
        sample_pairs = pairs.sample(100)

        self.test = self.ds[
            self.ds["player_1"].isin(sample_pairs["player_1"])
            & self.ds["player_2"].isin(sample_pairs["player_2"])
        ]
        self.train = self.ds[
            ~self.ds["player_1"].isin(sample_pairs["player_1"])
            | ~self.ds["player_2"].isin(sample_pairs["player_2"])
            | ~self.ds["player_1"].isin(sample_pairs["player_2"])
            | ~self.ds["player_2"].isin(sample_pairs["player_1"])
        ]

    def prepare_train_test_sets(self):
        """ Prepare train and test sets """
        train_left = self.train[self.train["date"] < self.date_2_3]
        test_right = self.test[self.test["date"] >= self.date_2_3]

        self.X_train = train_left.drop(columns=["target"])
        self.y_train = train_left["target"]

        self.X_test = test_right.drop(columns=["target"])
        self.y_test = test_right["target"]

    def process(self):
        """ Process the dataset """
        self.load_and_process_data()
        self.split_data()
        self.prepare_train_test_sets()

    def get_train_data(self):
        """ Get train data """
        return self.X_train, self.y_train

    def get_test_data(self):
        """ Get test data """
        return self.X_test, self.y_test

    def get_train_test_split(self):
        """ Get train and test split """
        return self.X_train, self.X_test, self.y_train, self.y_test

    def __repr__(self):
        return f"DatasetProcessor(file_path={self.file_path}), train={self.train.shape}, test={self.test.shape}"


if __name__ == "__main__":
    dp = DatasetProcessor("dataset.csv")
    print(dp.get_train_data())
    print(dp.get_test_data())
