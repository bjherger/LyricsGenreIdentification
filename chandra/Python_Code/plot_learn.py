#__author__ ='Chandra'
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pylab as pl
import csv
from sklearn import svm
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import sklearn.cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import sklearn.metrics
from sklearn import metrics
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn import grid_search
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn import cross_validation
import sklearn.cross_validation
from sklearn.ensemble import RandomForestClassifier
import plot_learn


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.2, 1.0, 4)):
    print " entered returning from learn plot"
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print train_scores_mean
    print test_scores_mean
    plt.grid()

    plt.xlim(0,1000)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    print " returning from learn plot"
    plt.show()
    return plt
