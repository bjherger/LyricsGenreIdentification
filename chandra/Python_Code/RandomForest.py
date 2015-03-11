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

def compute_importances(ensemble, normalize=False):
    trees_importances = [base_model.tree_.compute_feature_importances(normalize=normalize)
                         for base_model in ensemble.estimators_]
    return sum(trees_importances) / len(trees_importances)


def plot_feature_importances(importances, normalize=False, color=None, alpha=0.5, label=None, chunk=None):
    if hasattr(importances, 'estimators_'):
        importances = compute_importances(importances, normalize=normalize)
    if chunk is not None:
        importances = importances[chunk]
    plt.bar(range(len(importances)), importances, color=color, alpha=alpha, label=label)


def RandomF(X,y):

# dividing the original dataset into train and test set
    X,y = shuffle(X,y)
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model =  RandomForestClassifier(n_estimators=10)
    model= model.fit(X_train, y_train)
    print model.score(X_train, y_train)
    print model
    plt.figure(figsize=(16, 3))
    plot_feature_importances(model.fit(X_train, y_train), label='Features')
    _ = plt.legend()
    plt.title('Feature Importance Plot - Random Forests')
    plt.xlabel("Column Numbers")


    tuned_parameters = [{'max_features': ['sqrt', 'log2'],'max_depth':[10] ,  'n_estimators': [100, 200,300,400]}]
    rf = GridSearchCV( RandomForestClassifier(min_samples_split=1, compute_importances=False, n_jobs=-1), tuned_parameters, cv=3, verbose=2 ).fit(X_train, y_train)
    print 'Best parameters for Random Forest:'
    print rf.best_estimator_
    model =rf.best_estimator_
    bestmodel= rf.best_estimator_

    X1, y1 = shuffle(X_train,y_train)

    scor=[]
    skf = sklearn.cross_validation.StratifiedKFold(y1, 3)
    for train, test in skf:
        XTrain,YTrain= X1[train], y1[train]
        XTest,YTest = X1[test], y1[test]
        model.fit(XTrain, YTrain)
        preds = model.predict(XTest)
        scor.append(sklearn.metrics.accuracy_score(YTest, preds))
        print sklearn.metrics.accuracy_score(YTest, preds)
        print sklearn.metrics.recall_score(YTest, preds)
        print sklearn.metrics.f1_score(YTest,preds)
    print " --- "
    print np.mean(scor)


    predicted = model.predict(X_test)
    print metrics.accuracy_score(y_test, predicted)
    print metrics.confusion_matrix(y_test, predicted)
    print metrics.classification_report(y_test, predicted)
    preds = model.predict(X_test)
    print "Accuracy - ", sklearn.metrics.accuracy_score(y_test, preds)
    print "Precision -",sklearn.metrics.precision_score(y_test,preds)
    print "Recall -",sklearn.metrics.recall_score(y_test, preds)
    print "F1 score -",sklearn.metrics.f1_score(y_test,preds)

    param_range = range(5,50)
    # Validation curve
    train_scores, test_scores = validation_curve(model,X,y,"max_depth",param_range)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with Random Forest")
    plt.xlabel("maximum depth")
    plt.ylabel("Score")
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()
    # Learning curve

    train_sizes, train_scores, valid_scores = learning_curve(model,X1,y1, train_sizes=np.array([ 0.1, 0.325, 0.55, 0.775, 1. ]))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    plt.grid()
    plt.plot(train_sizes,train_scores_mean,'o-',color="r",label="Train Score")
    plt.plot(train_sizes,test_scores_mean,'o-',color="g",label ="Cross validation Score")
    plt.legend(loc="best")
    plt.title("Learning Curve")
    plt.xlabel("Train Sizes")
    plt.ylabel("Score")
