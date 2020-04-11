

from paths import *
import pandas as pd
import numpy as np


def generate_train_test():
    data_file = data_dir / "stroke_all.csv"
    train_file = data_dir / "stroke_train.csv"
    test_file = data_dir / "stroke_test.csv"

    # split data into train and test
    train, test = split_train_test(data_file)

    # save the split data
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)


def split_train_test(data_file, train_frac=0.66, random_seed=1):
    # set the random seed
    np.random.seed(random_seed)

    # read the data
    data = pd.read_csv(data_file)

    # select instances for the train set
    n_all = data.shape[0]
    mask = np.random.rand(n_all) < train_frac

    # filter the data to train and test set
    train = data.iloc[mask]
    test = data.iloc[~mask]

    return train, test

if __name__ == "__main__":
    generate_train_test()
