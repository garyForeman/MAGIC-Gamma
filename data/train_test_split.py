import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
BASE_FILE = os.path.join(DATA_DIRECTORY, 'magic04.data')


class TrainTestSplitter:
    def __init__(self):
        self.full_data = None
        self.train_data = None
        self.test_data = None

    def load(self, filename):
        self.full_data = pd.read_csv(filename)

    def split(self):
        self.train_data, self.test_data = train_test_split(self.full_data,
                                                           test_size=0.2)

    def write(self, directory):
        train_file = os.path.join(directory, 'train.csv')
        test_file = os.path.join(directory, 'test.csv')
        self.train_data.to_csv(train_file, index=False)
        self.test_data.to_csv(test_file, index=False)


if __name__ == '__main__':
    tts = TrainTestSplitter()
    tts.load(BASE_FILE)
    tts.split()
    tts.write(DATA_DIRECTORY)
