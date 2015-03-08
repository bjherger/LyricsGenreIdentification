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




def knn(X,y):
    from sklearn.utils import shuffle
    X,y = shuffle(X,y)
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # defining classifiers
    # KNN
    ## Finding best hyperparameters using grid search
    # Set the parameters by cross-validation
    import numpy as np
    tuned_parameters = [{'n_neighbors': np.arange(1,100),'p':[1,2]}]
    scores = ['f1','accuracy','precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring=score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        if(score =='accuracy'):
            model = clf.best_estimator_

        y_true, y_pred = y_test, clf.predict(X_test)
        print "Accuracy"
        print accuracy_score(y_true, y_pred)
        print(classification_report(y_true, y_pred))
        print()
    print model


    X1, y1 = shuffle(X_train,y_train)
    import sklearn.metrics
    scor=[]
    skf = sklearn.cross_validation.StratifiedKFold(y1, 3)
    for train, test in skf:
        XTrain,YTrain= X1[train], y1[train]
        XTest,YTest = X1[test], y1[test]
        model.fit(XTrain, YTrain)
        preds = model.predict(XTest)
        print sklearn.metrics.accuracy_score(YTest, preds)
        scor.append(sklearn.metrics.accuracy_score(YTest, preds))
        print sklearn.metrics.recall_score(YTest, preds)
    print " -- "
    print np.mean(scor)

    # Validation curve for KNN
    print(__doc__)

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    from sklearn.learning_curve import validation_curve

    param_range = np.arange(1,100)
    train_scores, test_scores = validation_curve(
        KNeighborsClassifier(), X_train, y_train, param_name="n_neighbors", param_range=param_range,
        cv=3, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with KNN")
    plt.xlabel("$\neighbors$")
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
    from sklearn.learning_curve import learning_curve
    from sklearn import cross_validation

    X, y = X_train,y_train
    title = "Learning Curves (KNN  $\neighbors=60$)"
    cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10,
                                       test_size=0.2, random_state=0)

    estimator = KNeighborsClassifier(n_neighbors=68)
    ylim= (0.7, 1.01)
    # plot_learn.plot_learning_curve(estimator
    plot_learn.plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)
    plt.show()
