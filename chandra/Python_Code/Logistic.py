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


def Logist(X,y):
    # dat = pd.read_csv("DatasetwithFeatures.csv")
    # dat.loc[dat.genre=='adult_contemp', 'Target'] =1
    # dat.loc[dat.genre=='R_and_B', 'Target'] =0
    # print dat.head()
    # print dat.shape()
    #
    # y = dat.iloc[:,[26]]
    # X = dat.iloc[:,[17,18,19,20,21,22,23,24,25]]
    # cols = X.columns
    model = LogisticRegression(C=1)
    model = model.fit(X,y)
    model.score(X,y)
    # dividing the original dataset into train and test set
    X,y = shuffle(X,y)
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(C=1)
    model = model.fit(X_train,y_train)
    model.score(X_train, y_train)
    print model.coef_
    plt.plot(model.coef_[0].ravel())
    probs = model.predict_proba(X_test)
    predicted = model.predict(X_test)
    print metrics.accuracy_score(y_test, predicted)
    print metrics.confusion_matrix(y_test, predicted)
    print metrics.classification_report(y_test, predicted)

    # Grid search for best parameters for logisic model with L2 regularization
    j = range(1,50)
    parameters = {'C':j}
    svr = LogisticRegression(penalty='l2')
    clf = grid_search.GridSearchCV(svr, parameters)
    clf.fit(X_train,y_train)

    print clf.best_params_
    print clf.best_score_
    print clf.best_estimator_
    print clf.grid_scores_[0]
    print len(clf.grid_scores_[0])
    sc=[]
    for i in range(0,49):
        sc.append(clf.grid_scores_[i][1])
    plt.plot(j,sc)
    plt.xlabel('C')
    plt.ylabel('Mean Score')
    plt.title('Grid search Results')
    model2 = clf.best_estimator_

    # Grid search for best parameters for logisic model with L1 regularization
    j = range(1,50)
    parameters = {'C':j}
    svr = LogisticRegression(C=0,penalty='l1')
    clf = grid_search.GridSearchCV(svr, parameters)
    clf.fit(X_train,y_train)
    print clf.best_params_
    print clf.best_score_
    print clf.best_estimator_
    print clf.grid_scores_[0]
    sc=[]
    for i in range(0,49):
        sc.append(clf.grid_scores_[i][1])
    plt.plot(j,sc)
    plt.xlabel('C')
    plt.ylabel('Mean Score')
    plt.title('Grid search Results')
    model1 = clf.best_estimator_


    # From Grid search results, L1 regualrization gives better accuracy compared to L2.
    model = model1
    X1, y1 = shuffle(X_train,y_train)

    # Staratified 3 fold cross validation
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

    # Plotting Validation Curve for Logistic Regression Model

    param_range = range(1,50)
    train_scores, test_scores = validation_curve(model,X,y,'C',param_range)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with Logistic Regression")
    plt.xlabel("C")
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

    # Plotting Learning curve for Logistic Regression Model

    train_sizes, train_scores, valid_scores = learning_curve(model,X,y, train_sizes=np.array([ 0.1, 0.325, 0.55, 0.775, 1. ]))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    plt.grid()
    plt.plot(train_sizes,train_scores_mean,'o-',color="r",label="Train Score")
    plt.plot(train_sizes,test_scores_mean,'o-',color="g",label="Cross validation Score")
    plt.legend(loc="best")
    plt.title("Learning Curve")
    plt.xlabel("Training Sizes")
    plt.ylabel("Score")

    # Plotting Prediction Error Rate
    error={}
    for x in range(1,50):
        error[x] = []

    for i in range(20):
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X,y, test_size=.2, random_state=0)
        for x in range(1,50):
            m1 =LogisticRegression(C=x)
            m2 = m1.fit(X_train_new,y_train_new)
            y_predict = m2.predict(X_test_new)
            error_rate = np.mean(y_predict != y_test_new)
            error[x].append(error_rate)

    error_path = pd.DataFrame(error).mean()
    error_path.plot(style = 'o-k', label = 'Error rate')
    error_path.cummin().plot(style = 'r-', label = 'Lower envelope')
    plt.xlabel('C (regularization parameter)')
    plt.ylabel('Prediction error rate')
    plt.legend(loc = 'upper right')
    plt.title('Logistic Regression - C vs Prediction Error Rate')

    # 10- fold cross validation

    kf = cross_validation.KFold(X.shape[0], n_folds=10, indices=False)

    param_range = [0.1,0.5, 1, 2, 4, 16, 64]

    train_scores, test_scores = validation_curve(model,X,y,'C',param_range=param_range,cv=kf)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="g")
    plt.errorbar(param_range, test_scores_mean,yerr=test_scores_std**2,color = 'r')
    plt.legend(loc="best")
    plt.show()

