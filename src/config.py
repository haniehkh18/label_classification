import argparse
from pathlib import Path


class BaseConfig:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--train_set', type=str,
                                 default=Path(__file__).parents[1].__str__() + '/data/train.csv',
                                 help='path to train set')

        self.parser.add_argument('--test_set', type=str,
                                 default=Path(__file__).parents[1].__str__() + '/data/test.csv',
                                 help='path to test set')

        self.parser.add_argument('--dataset', type=str,
                                 default=Path(__file__).parents[1].__str__() + '/data/articles.csv',
                                 help='path to dataset')

    def get_args(self):
        return self.parser.parse_args()
