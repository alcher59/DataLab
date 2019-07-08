import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn import svm
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn import linear_model

# outlist - словарь с выходами
# ycolumn - столбец c выходными данными
# category - столбец с категорией ( 0 или 1)
# disp - описываемая дисперсия
# param - параметр, в который пользователь передает либо числовое значение, либо факторное (словарь с группой значений)


def quantity(param):
    if type(param) is dict:
        return 1

def sgd_classifier():

def kernel_approximation():

def linear_svc():

def naive_bayes():

def kneighbors_classifier():

def ensemble_classifiers():

def kmeans():

def spectral_clustering():

def minibatch_kmeans():

def mean_shift():

def sgd_regressor():

def lasso():

def elastic_net():

def svr_kernel():

def ensemble_regressors():

def ridge_regressor():



def cheat_sheet(y_column_name, category, param):
    def classification():
        if df.__len__() < 100000:
            linear_svc() or
            if type(df[y_column_name][1]) is str:    # проверка типа данных
                naive_bayes()
            else:
                kneighbors_classifier() or ensemble_classifiers()
        else:
            sgd_classifier()

    def clustering(outlist):
        if outlist!=None:
            if df.__len__() < 10000:
                kmeans() or spectral_clustering()
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
                lasso() and elastic_net()
            else:
                ridge_regressor() and svr_kernel() or
                ensemble_regressors() and svr_kernel()

    def dimensionality():
        print("use Randomized PCA or")
        if df.__len__() < 10000:
            print("use Isomap and Spectral Embedding or LLE")
        else:
            print("use kernel approximation")

    if df.__len__() > 50:
      for _, row in df.iterrows():
            if row[category] == 1:
                if df[y_column_name]:
                    classification()
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
