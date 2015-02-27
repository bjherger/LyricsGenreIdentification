#!/usr/bin/env python
"""
coding=utf-8
"""
# imports
# *********************************
import random

import pandas as pd
from nb import naivebayes
# global variables
# *********************************


__author__ = 'bjherger'
__version__ = '1.0'
__email__ = 'b@revupsoftware.com'
__status__ = 'Development'
__maintainer__ = 'bjherger'


# functions
# *********************************

def generate_lyrics_df():
    lyrics_df = pd.read_csv('data/unique_tracks.csv', index_col = 0)
    return lyrics_df

def create_test_train(df, y_label):

    sample_size = int(len(df.index)*.1)
    test_rows = random.sample(df.index, sample_size)

    train = df.drop(test_rows)
    test = df.ix[test_rows]

    train_x_df = train.drop(y_label, axis=1)['lyrics']
    train_y_df = train[y_label]

    test_x_df = test.drop(y_label, axis=1)['lyrics']
    test_y_df = test[y_label]

    return train_x_df.as_matrix(), train_y_df.as_matrix(), test_x_df.as_matrix(), test_y_df.as_matrix()

def run_nb(train_x, train_y, test_x):
    nb = naivebayes.NaiveBayes()

    nb.fit(train_x, train_y)

    preds = nb.predict(test_x)
    return preds

def main():
    print 'hello world'
    lyrics_df = generate_lyrics_df()
    train_x, train_y, test_x, test_y = create_test_train(lyrics_df, 'genre')
    nb_preds = run_nb(train_x, train_y, test_x)
    print zip(nb_preds, test_y)



# main
# *********************************

if __name__ == '__main__':
    main()


