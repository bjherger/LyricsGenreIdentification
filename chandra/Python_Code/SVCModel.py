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


def SVCM(X,y):
    X,y = shuffle(X,y)
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # SVM
    # grid search for SVM
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]}]

    scores = ['f1','accuracy','precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring=score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()
        if (score =='accuracy'):
            model = clf.best_estimator_

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")

        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
    print model
    X_2d = X_train[:, :2]
    X_2d = X_2d[y_train > 0]
    Y_2d = y_train[y_train > 0]
    Y_2d -= 1

    # It is usually a good idea to scale the data for SVM training.
    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    X_2d = scaler.fit_transform(X_2d)

    C_range = 10.0 ** np.arange(-2, 9)
    gamma_range = 10.0 ** np.arange(-5, 4)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedKFold(y=y_train, n_folds=3)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X_train, y_train)

    print("The best classifier is: ", grid.best_estimator_)

    # Validation curve for SVM


    param_range = np.logspace(-6, -1, 5)
    train_scores, test_scores = validation_curve(grid.best_estimator_, X, y, param_name="gamma", param_range=param_range,
        cv=10, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVM")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()

     ## learning curve
    print "Plotting Learning Curve"
    X, y = X_train,y_train
    title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10,
                                       test_size=0.2, random_state=0)
    estimator = SVC(gamma=0.0001,C=10)
    plot_learn.plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

    plt.show()
