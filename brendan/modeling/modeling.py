#!/usr/bin/env python
"""
coding=utf-8
"""
# imports
# *********************************
import random

import pandas as pd
import numpy as np
from nb import naivebayes

import validation
# global variables
# *********************************


__author__ = 'bjherger'
__version__ = '1.0'
__email__ = '13herger@gmail.com'
__status__ = 'Development'
__maintainer__ = 'bjherger'


# functions
# *********************************

def generate_lyrics_df():
    lyrics_df = pd.read_csv('../../data/raw/rb_adultcontemp_train.csv',
                            index_col=0)
    return lyrics_df


def run_nb(train_x, train_y, test_x):
    nb = naivebayes.NaiveBayes()

    nb.fit(train_x, train_y)

    preds = nb.predict(test_x)
    return preds


def main():
    print 'hello world'
    lyrics_df = generate_lyrics_df()
    train_x, train_y, test_x, test_y = validation.create_test_train(lyrics_df,
                                                             'genre')
    nb_preds = run_nb(train_x, train_y, test_x)
    validation.compute_accuracy(nb_preds, test_y)
    results = validation.k_folds(data_frame=lyrics_df,
                             learner=naivebayes.NaiveBayes, k=5)

    print np.mean(results)
    print np.std(results)
    print results


# main
# *********************************

if __name__ == '__main__':
    main()


