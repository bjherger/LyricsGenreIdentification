# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:18:21 2015
@author: laylamartin
"""

import numpy as np
import pandas as pd
import time

from sklearn import ensemble
from sklearn import grid_search
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm


start_time = time.time()


def transform_csv(data, target_col=0):
    """ 
    Transforms a pandas dataframe into a dictionary to use with sklearn machine learning algorithms
    :param: data: a pandas DataFrame
    :param: target_col: integer index representing which column is the dependent variable
    :return: a Python dictionary with keys 'data' and 'target'
    """  
    
    predictor_cols = ['genre', 'density_unique_word', 'density_noun', 'density_verb',
                      'density_stop_word', 'density_cuss_words', 'avg_words_per_line', 
                      'num_words']
    new_cols = [col for col in data.columns if col in predictor_cols]
    data = data[new_cols]
           
    # change to string of column name:
    if type(target_col) == int:
        target_col = list(data[[target_col]].columns)[0] 
    # define target values:      
    data_target = data[target_col]
    data_target = data_target.values    
    # define data values: 
    data_data = data.drop(target_col, axis = 1)
    data_data = data_data.values   

    dic = {'data': data_data,
           'target': data_target,
           }
           
    return dic

def optimize_logistic(X_train, y_train):
    """
    :param: X_train: np array of target variables
    :param: y_train: np array of features
    :return: optimal_params: hyperparameters optimized across five fold cross 
      validation to be used on test data
    """
    param_grid = [
                 {'penalty': ['l1', 'l2']}
                 ]
                 
    estimator = linear_model.LogisticRegression()  
    clf = grid_search.GridSearchCV(estimator, param_grid, scoring='accuracy',
                                   cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    optimal_params = clf.best_params_
    
    return optimal_params

def run_logistic(X_train, y_train, X_test, y_test, logistic_params):
    """
    :param: X_train: np array of target variables
    :param: y_train: np array of features
    :param: X_test: np array of features
    :param: y_test: np array of target variables
    :param: logistic_params: optimized parameters from function optimize_logistic()
    """
    clf = linear_model.LogisticRegression(penalty = logistic_params['penalty'])
    clf.fit(X_train, y_train)
    
    test_accuracy = clf.score(X_test, y_test)
    print "Test accuracy:", test_accuracy

    return

def optimize_svm(X_train, y_train):
    """
    :param: X_train: np array of target variables
    :param: y_train: np array of features
    :return: optimal_params: hyperparameters optimized across five fold cross 
      validation to be used on test data
    """
    gamma_range = 10.0 ** np.arange(-5, 4)
    #gamma_range=np.arange(1.0*10**-2, 9.0*10**-1, 0.2)
    C_range = 10.0 ** np.arange(-5, 4)
    #C_range = np.arange(1.0*10**-1, 9.0*10**0, 0.2)
    #print C_range
    #print gamma_range
    param_grid = [
                 {'C': C_range, 'gamma':gamma_range}
                 ]
            
    estimator = svm.SVC(kernel='rbf')
    clf = grid_search.GridSearchCV(estimator, param_grid, scoring='accuracy',
                                   cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    optimal_params = clf.best_params_
    
    return optimal_params
        
def run_svm(X_train, y_train, X_test, y_test, svm_params):
    """
    :param: X_train: np array of target variables
    :param: y_train: np array of features
    :param: X_test: np array of features
    :param: y_test: np array of target variables
    :param: svm_params: optimized parameters from function optimize_svm()
    """
    clf = svm.SVC(kernel = 'rbf', 
                  C = svm_params['C'],
                  gamma = svm_params['gamma'])
    clf.fit(X_train, y_train)
    
    test_accuracy = clf.score(X_test, y_test)
    print "Test accuracy:", test_accuracy

    return    
    
def optimize_rf(X_train, y_train):
    """
    :param: X_train: np array of target variables
    :param: y_train: np array of features
    :return: optimal_params: hyperparameters optimized across five fold cross 
      validation to be used on test data
    """
    n_estimator_range = np.arange(100, 1001, 100)
    criterions = ['gini', 'entropy']
    max_features = ['auto', 'log2']
    
    param_grid = [
                 {'n_estimators': n_estimator_range, 
                  'criterion':criterions,
                  'max_features': max_features}
                 ]
            
    estimator = ensemble.RandomForestClassifier()
    clf = grid_search.GridSearchCV(estimator, param_grid, scoring='accuracy',
                                   cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    optimal_params = clf.best_params_
    
    return optimal_params
       
def run_rf(X_train, y_train, X_test, y_test, rf_params):
    """
    :param: X_train: np array of target variables
    :param: y_train: np array of features
    :param: X_test: np array of features
    :param: y_test: np array of target variables
    :param: rf_params: optimized parameters from function optimize_rf()
    """
    clf = ensemble.RandomForestClassifier(n_estimators = rf_params['n_estimators'],
                                          criterion = rf_params['criterion'],
                                          max_features = rf_params['max_features'])
    clf.fit(X_train, y_train)
    
    test_accuracy = clf.score(X_test, y_test)
    print "Test accuracy:", test_accuracy

    return    

def optimize_adaboost(X_train, y_train):
    """
    :param: X_train: np array of target variables
    :param: y_train: np array of features
    :return: optimal_params: hyperparameters optimized across five fold cross 
      validation to be used on test data
    """
    n_estimator_range = np.arange(100, 1001, 100)
    learning_rate = 10**np.arange(0, 1)
    
    param_grid = [
                 {'n_estimators': n_estimator_range, 
                  'learning_rate':learning_rate}
                 ]
            
    estimator = ensemble.AdaBoostClassifier()
    clf = grid_search.GridSearchCV(estimator, param_grid, scoring='accuracy',
                                   cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    optimal_params = clf.best_params_
    
    return optimal_params
      
def run_adaboost(X_train, y_train, X_test, y_test, ada_params):
    """
    :param: X_train: np array of target variables
    :param: y_train: np array of features
    :param: X_test: np array of features
    :param: y_test: np array of target variables
    :param: adaboost_params: optimized parameters from function optimize_adaboost()
    """
    clf = ensemble.AdaBoostClassifier(n_estimators = ada_params['n_estimators'],
                                      learning_rate = ada_params['learning_rate'])
    clf.fit(X_train, y_train)
    
    test_accuracy = clf.score(X_test, y_test)
    print "Test accuracy:", test_accuracy

    return 

def optimize_gradientboost(X_train, y_train):
    """
    :param: X_train: np array of target variables
    :param: y_train: np array of features
    :return: optimal_params: hyperparameters optimized across five fold cross 
      validation to be used on test data
    """
    n_estimator_range = np.arange(100, 1001, 100)
    learning_rate = 10**np.arange(0, 1)
    
    param_grid = [
                 {'n_estimators': n_estimator_range, 
                  'learning_rate':learning_rate}
                 ]
            
    estimator = ensemble.GradientBoostingClassifier()
    clf = grid_search.GridSearchCV(estimator, param_grid, scoring='accuracy',
                                   cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    optimal_params = clf.best_params_
    
    return optimal_params
    
def run_gradientboost(X_train, y_train, X_test, y_test, gradient_params):
    """
    :param: X_train: np array of target variables
    :param: y_train: np array of features
    :param: X_test: np array of features
    :param: y_test: np array of target variables
    :param: gradientboost_params: optimized parameters from function optimize_gradientboost()
    """
    clf = ensemble.GradientBoostingClassifier(n_estimators = gradient_params['n_estimators'],
                                      learning_rate = gradient_params['learning_rate'])
    clf.fit(X_train, y_train)
    
    test_accuracy = clf.score(X_test, y_test)
    print "Test accuracy:", test_accuracy

    return

def optimize_knn(X_train, y_train):
    """
    :param: X_train: np array of target variables
    :param: y_train: np array of features
    :return: optimal_params: hyperparameters optimized across five fold cross 
      validation to be used on test data
    """
    n_neighbors_range = np.arange(1, 20)
    
    param_grid = [
                 {'n_neighbors': n_neighbors_range}
                 ]
            
    estimator = neighbors.KNeighborsClassifier()
    clf = grid_search.GridSearchCV(estimator, param_grid, scoring='accuracy',
                                   cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    optimal_params = clf.best_params_
    
    return optimal_params
    
def run_knn(X_train, y_train, X_test, y_test, knn_params):
    """
    :param: X_train: np array of target variables
    :param: y_train: np array of features
    :param: X_test: np array of features
    :param: y_test: np array of target variables
    :param: knn_params: optimized parameters from function knn_gradientboost()
    """
    clf = neighbors.KNeighborsClassifier(n_neighbors = knn_params['n_neighbors'])
    clf.fit(X_train, y_train)
    
    test_accuracy = clf.score(X_test, y_test)
    print "Test accuracy:", test_accuracy

    return


def main():
    # read in data with features:
    raw_data = pd.read_csv( 'train.csv' )
    data_dic = transform_csv(raw_data)
    X_train = data_dic['data']
    y_train = data_dic['target']
    
    test_data = pd.read_csv( 'holdout.csv' )
    data_dic = transform_csv(test_data) 
    X_test = data_dic['data']
    y_test = data_dic['target']

    ########################################################

    print "Logistic Regression:"
    logistic_params = optimize_logistic(X_train, y_train)
    print "Optimal parameters:", logistic_params
    run_logistic(X_train, y_train, X_test, y_test, logistic_params)
    
    print ''
    
    print "SVC:"
    svm_params = optimize_svm(X_train, y_train)
    print "Optimal parameters:", svm_params
    run_svm(X_train, y_train, X_test, y_test, svm_params)

    print ''
    
    print "Random Forest:"
    rf_params = optimize_rf(X_train, y_train)
    print "Optimal parameters:", rf_params
    run_rf(X_train, y_train, X_test, y_test, rf_params)

    print ''
    
    print "AdaBoost:"
    ada_params = optimize_adaboost(X_train, y_train)
    print "Optimal parameters:", ada_params
    run_adaboost(X_train, y_train, X_test, y_test, ada_params)

    print ''

    print "Stochastic Gradient Boost:"
    gradient_params = optimize_gradientboost(X_train, y_train)
    print "Optimal parameters:", gradient_params
    run_gradientboost(X_train, y_train, X_test, y_test, gradient_params)

    print ''

    print "KNN:"
    knn_params = optimize_knn(X_train, y_train)
    print "Optimal parameters:", knn_params
    run_knn(X_train, y_train, X_test, y_test, knn_params)

    
main()
print ('runtime: %d seconds' % (time.time() - start_time))






