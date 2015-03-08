#!/usr/bin/env python
"""
coding=utf-8
"""
# imports
# *********************************
import pandas as pd
import numpy as np
import random
# global variables
# *********************************


__author__ = 'bjherger'
__version__ = '1.0'
__email__ = '13herger@gmail.com'
__status__ = 'Development'
__maintainer__ = 'bjherger'


# functions
# *********************************

def create_test_train(df, y_label, test_rows=None):
    sample_size = int(len(df.index) * .1)

    if test_rows is None:
        test_rows = random.sample(df.index, sample_size)

    train = df.drop(test_rows)
    test = df.ix[test_rows]

    train_x_df = train.drop(y_label, axis=1)['lyrics_body']
    train_y_df = train[y_label]

    test_x_df = test.drop(y_label, axis=1)['lyrics_body']
    test_y_df = test[y_label]

    return train_x_df.as_matrix(), train_y_df.as_matrix(), test_x_df.as_matrix(), test_y_df.as_matrix()



def compute_accuracy(y_pred, y_act):

    result_df = pd.DataFrame({'y_act': y_act, 'y_pred': y_pred})


    result_df['correct'] = result_df['y_pred'] == result_df['y_act']

    return np.mean(result_df['correct'])


def k_folds(data_frame, learner, k):
    data_frame = pd.DataFrame(data_frame)

    list_of_subsets = np.array_split(data_frame.index, k)

    accuracy_list = list()
    for subset in list_of_subsets:
        # local_df = data_frame.copy(deep=True)
        train_x, train_y, test_x, test_y = create_test_train(data_frame,
                                                             'genre',
                                                             test_rows=subset)
        learner_inst = learner()
        learner_inst.fit(train_x, train_y)
        predictions = learner_inst.predict(test_x)



        result_df = pd.DataFrame({'y_act': test_y, 'y_pred': predictions})

        result_df['correct'] = result_df['y_act'] == result_df['y_pred']
        # print data_frame

        accuracy_list.append(np.mean(result_df['correct']))
    return accuracy_list

def main():
    print 'hello world'

# main
# *********************************

if __name__ == '__main__':
    main()


