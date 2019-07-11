import pandas as pd

from sklearn import svm
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift

# outlist - словарь с выходами
# ycolumn - столбец c выходными данными
# category - столбец с категорией ( 0 или 1)
# disp - описываемая дисперсия
# param - параметр, в который пользователь передает либо числовое значение, либо факторное (словарь с группой значений)


def quantity(param):
    if type(param) is dict:
        return 1

def sgd_classifier(y_column_name):
    X = [df[y_column_name], [1, 1]]
    y = [0, 1]
    clf = SGDClassifier()
    clf.fit(X, y)


def kernel_approximation():
    X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    y = [0, 0, 1, 1]
    rbf_feature = RBFSampler()
    X_features = rbf_feature.fit_transform(X)
    clf = SGDClassifier()
    clf.fit(X_features, y)
    clf.score(X_features, y)

def linear_svc():
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC()
    clf.fit(X, y)

def naive_bayes():
    iris = datasets.load_iris()
    gnb = GaussianNB()
    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)

def kneighbors_classifier():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

def ensemble_classifiers():
    bagging = BaggingClassifier(KNeighborsClassifier())

def kmeans():

def spectral_clustering():
    sc = SpectralClustering()
    sc.fit_predict(adjacency_matrix)

def minibatch_kmeans():

def mean_shift():
    X = [[0., 0.], [1., 1.]]
    ms = MeanShift()
    ms.fit(X)

def sgd_regressor():
    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    clf = linear_model.SGDRegressor()
    clf.fit(X, y)

def lasso():
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit([[0, 0], [1, 1]], [0, 1])

def elastic_net():

def svr_linear():
    X = [[0, 0], [2, 2]]
    y = [0.5, 2.5]
    clf = svm.SVR(kernel='linear')
    clf.fit(X, y)

def svr_rbf():
    X = [[0, 0], [2, 2]]
    y = [0.5, 2.5]
    clf = svm.SVR()
    clf.fit(X, y)
def ensemble_regressors():

def ridge_regressor():
    reg = linear_model.Ridge(alpha=.5)
    reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])



def cheat_sheet(y_column_name, category, param):
    def classification():
        if df.__len__() < 100000:
            try:
                linear_svc()
            except:
                print("not working")
                if type(df[y_column_name][1]) is str:  # проверка типа данных
                    naive_bayes()
                else:
                    try:
                        kneighbors_classifier()
                    except:
                        print("not working")
                        ensemble_classifiers()
        else:
            sgd_classifier()

    def clustering(outlist):
        if outlist!=None:
            if df.__len__() < 10000:
                try:
                    kmeans()
                except:
                    print("not working")
                    spectral_clustering()
            else:
                minibatch_kmeans()
        else:
            if df.__len__() < 10000:
                mean_shift()
            else:
                print("Tough luck")

    def regression(disp):
        if df.__len__() < 100000:
            if disp <= 0.8:
                lasso()
                elastic_net()
            else:
                try:
                    ridge_regressor()
                    svr_linear()
                except:
                    print("not working")
                    ensemble_regressors()
                    svr_rbf()



    def dimensionality():
        print("use Randomized PCA or")
        if df.__len__() < 10000:
            print("use Isomap and Spectral Embedding or LLE")
        else:
            print("use kernel approximation")

    if df.__len__() > 50:
      for _, row in df.iterrows():
            if row[category] == 1:
                if df[y_column_name]!=None:
                    classification(y_column_name)
                else:
                    clustering("test")
            else:
                if quantity(param) == 1:
                    regression("test")
                else:
                    dimensionality()
    else:
        print("Get more data")

if __name__ == '__main__':
    df = pd.read_csv('df.csv', sep=';', encoding="cp1251")
    cheat_sheet("test", "test", "test")
