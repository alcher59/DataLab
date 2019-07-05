﻿import pandas as pd

# outlist - словарь с выходами
# ycolumn - столбец c выходными данными
# category - столбец с категорией ( 0 или 1)
# disp - описываемая дисперсия
# param - параметр, в который пользователь передает либо числовое значение, либо факторное (словарь с группой значений)

df = pd.read_csv('df.csv', sep=';', encoding="cp1251")

def quantity(param):
    if type(param) is dict:
        return 1

def cheat_sheet(ycolumn, category, param):
    def classification():
        if df.__len__() < 100000:
            print("use Linear SVC or Text Data")
            if type(df[ycolumn][1]) is str:    # проверка типа данных
                print("use Naive Bayes")
            else:
                print("use Kneighbors Classifier or SVC and Ensemble Classifiers")
        else:
            print("use SGD Classifier or kernel approximation")

    def clustering(outlist):
        if outlist:
            if df.__len__() < 10000:
                print("use KMeans or Spectral Clustering and GMM")
            else:
                print("use MiniBatch KMeans")
        else:
            if df.__len__() < 10000:
                print("use MeanShift and VBGMM")
            else:
                print("Tough luck")

    def regression(disp):
        if df.__len__() < 100000:
            if disp <= 0.8:
                print("use Lasso and ElasticNet")
            else:
                print("use RidgeRegression and SVR(kernel='linear') or SVR(kernel='rbf') and EnsembleRegressions ")

    def dimensionality():
        print("use Randomized PCA or")
        if df.__len__() < 10000:
            print("use Isomap and Spectral Embedding or LLE")
        else:
            print("use kernel approximation")

    if df.__len__() > 50:
      for index, row in df.iterrows():
            if row[category] == 1:
                if df[ycolumn]:
                    classification()
                else:
                    clustering()
            else:
                if quantity(param) == 1:
                    regression()
                else:
                    dimensionality()
    else:
        print("Get more data")
