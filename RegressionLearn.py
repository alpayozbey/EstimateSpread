import numpy as np
from sklearn.linear_model import LogisticRegression
from numpy import log2 as log
import math
import statistics
import random as rd
import pandas as pd
import re
import time
import datetime
import operator
import numpy as np
import pandas as pd
import collections
import unicodedata
import collections
import seaborn as sns
import collections
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from datetime import datetime, date, timedelta
from IPython.display import Image
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import itertools


def UpdateOptions(obj, options):
    if len(options) == 0:
        return obj, {}

    out = {}

    for option in options.keys():
        if option in obj.__dict__.keys():
            obj.__dict__.update({option: options[option]})

        else:
            out[option] = options[option]

    return obj, out


def Split_Data(Data, ratio):
    size = Data.shape[0]
    indx = list(range(0, size))
    rd.shuffle(indx)
    indx_train = indx[:round(size * ratio)]
    indx_test = indx[round(size * ratio):]

    Data_Train = Data.iloc[indx_train, :].reset_index(drop=True)
    Data_Test = Data.iloc[indx_test, :].reset_index(drop=True)
    return Data_Train, Data_Test


def OptFeatures(ObjLeaner, DATA):
    DATA_Train, DATA_Test = train_test_split(DATA, test_size=0.2, shuffle=True)
    Label = list(DATA.keys())[-1]
    Features = list(DATA.keys())[:-1]
    best_subset = []
    BestResult = 1000
    for i in (list(range(1, 5)) + list(range(len(Features), len(Features) - 5, -1))):
        PossibleCombinations = list(itertools.combinations(Features, i))
        for combination in PossibleCombinations:
            ObjLeaner.FeatureSubset = {}
            DATA_Train_new = DATA_Train[list(combination)].copy()
            DATA_Train_new[Label] = DATA_Train[Label]
            DATA_Test_new = DATA_Test[list(combination)].copy()
            DATA_Test_new[Label] = DATA_Test[Label]
            ObjLeaner.TrainLearner(DATA_Train_new)
            predictions = ObjLeaner.Predict(DATA_Test_new)
            MAE = ObjLeaner.CalculatePerformance(predictions, DATA_Test_new.iloc[:, -1].to_numpy().astype(int))
            if MAE < BestResult:
                BestResult = MAE
                best_subset = list(combination)
                print("Done: #" + str(i))
                print("MAE: " + str(BestResult))

            print("Done: #" + str(i))
            print("MAE: " + str(BestResult))

    ObjLeaner.FeatureSubset = {}
    DATA_Train_new = DATA_Train[best_subset].copy()
    DATA_Test_new = DATA_Test[best_subset].copy()
    ObjLeaner.TrainLearner(DATA_Train_new)
    predictions = ObjLeaner.Predict(DATA_Test_new)
    MAE = ObjLeaner.CalculatePerformance(predictions, DATA_Test_new.iloc[:, -1].to_numpy().astype(int))
    return best_subset, MAE

def OptParameters(ObjLearner, DATA, parameter2try):
    DATA_Train, DATA_Test = train_test_split(DATA, test_size=0.2, shuffle=True)
    allcombinations = parameter2try[list(parameter2try.keys())[0]]
    for k in list(parameter2try.keys())[1:]:
        allcombinations = itertools.product(allcombinations, parameter2try[k])
    BestResult = 1000
    for combination in allcombinations:
        params = {}
        for key in list(parameter2try.keys())[::-1]:
            if combination.size == 2:
                params[key] = combination[1]
                combination = combination[0]
            else:
                params[key] = combination

            for k in params.keys():
                setattr(ObjLearner.Model, k, params[k])

        ObjLearner.TrainLearner(DATA_Train)
        predictions = ObjLearner.Predict(DATA_Test)
        MAE = ObjLearner.CalculatePerformance(predictions, DATA_Test.iloc[:, -1].to_numpy().astype(int))
        print(str(MAE))
        if MAE < BestResult:
            BestResult = MAE
            BestParameters = params

    for k in BestParameters.keys():
        setattr(ObjLearner.Model, k, BestParameters[k])
    ObjLearner.TrainLearner(DATA_Train)
    predictions = ObjLearner.Predict(DATA_Test)
    MAE = ObjLearner.CalculatePerformance(predictions, DATA_Test.iloc[:, -1].to_numpy().astype(int))
    return BestParameters, MAE


class RandomForest:
    pass


class LSTM:
    pass


class NegativeBinomial:

    def __init__(self, args={}):
        self.NegativeBinomial = {}
        self.Results = {}
        self.alpha = 100
        self.ModelFormula = ''
        self.__dict__.update(args)

    def extractFormula(self, data):
        columns = data.columns
        ModelFormula = columns[-1] + ' ~ '
        for c in columns[:-2]:
            ModelFormula += str(c) + ' + '

        ModelFormula += str(columns[-2])
        self.ModelFormula = ModelFormula
        return self.ModelFormula

    def TrainModel(self, data, args={}):
        self.__dict__.update(args)
        self.NegativeBinomial = smf.glm(formula=self.extractFormula(data),
                                        data=data,
                                        family=sm.families.NegativeBinomial(alpha=self.alpha))
        self.Results = self.NegativeBinomial.fit()

    def Predict(self, data):
        predictions = self.Results.predict(data).astype(int)
        return predictions

    def CalculatePerformance(self, result, target):

        if self.NegativeBinomial == {}:
            return None
        else:
            return eval_measures.meanabs(result, target)


class Train:

    def __init__(self, options={}):
        self.__dict__.update(options)
        # Initialize:
        self.Validation = 'CrossValidation'
        self.Shuffle = 'Yes'
        self.NFolds = 5
        self.TrainRatio = 0.8

        if self.Validation == 'CrossValidation':
            if self.NFolds == 1:
                self.TrainRatio = 1
                self.Validation = 'No'
            else:
                self.TrainRatio = 1 - (1 / self.NFolds)
        elif self.Validation == 'No':
            self.TrainRatio = 1
            self.NFolds = 1
        elif self.Validation == 'OnePass':
            self.NFolds = 1

    def SplitData4Validation(self, Data):

        rd.seed(1)
        size = Data.shape[0]
        indx = list(range(0, size))

        if self.Shuffle == 'Yes':
            rd.shuffle(indx)

        stepsize = round(size * (1 - self.TrainRatio))
        TrainingData = {}
        ValidationData = {}
        start = 0
        for i in range(0, self.NFolds):
            stop = start + stepsize
            if stop > size:
                stop = size

            validation_indx = indx[start:stop]
            train_indx = indx[:start] + indx[stop:]

            ValidationData[i] = Data.iloc[validation_indx, :].reset_index(drop=True)
            TrainingData[i] = Data.iloc[train_indx, :].reset_index(drop=True)
            start = stop

        return TrainingData, ValidationData

    def Train(self, Model, Data):

        # Prepare Data:

        TrainingData, ValidationData = self.SplitData4Validation(Data)

        # Train Model:

        if self.TrainRatio == 1:
            Model.TrainModel(TrainingData.get(0))
            return Model, -1
        else:
            Performance = []
            for i in range(0, self.NFolds):
                Model.TrainModel(TrainingData.get(i))
                Performance.append(Model.CalculatePerformance(Model.Predict(ValidationData.get(i).iloc[:, :-1]),
                                                              ValidationData.get(i).iloc[:, -1]))

            return Model, statistics.mean(Performance)


class Learner:

    def __init__(self, method, args={}):

        if method == 'NegativeBinomial':
            self.Method = 'NegativeBinomial'
            self, options = UpdateOptions(self, args)
            self.Model = NegativeBinomial(options)

        elif method == 'RandomForest':
            self.Method = 'RandomForest'
            self, options = UpdateOptions(self, args)
            self.Model = RandomForest(options)

        elif method == 'LSTM':
            self.Method = 'LSTM'
            self, options = UpdateOptions(self, args)
            self.Model = LSTM(options)

        self.Train = Train(options)

    def TrainLearner(self, DATA):

        self.Model, accuracy = self.Train.Train(self.Model, DATA)

        return accuracy

    def Predict(self, DATA):

        return self.Model.Predict(DATA)

    def OptimizeFeatures(self, DATA):

        features, accuracy = OptFeatures(self, DATA)

        return features, accuracy

    def OptimizeModelParameters(self, DATA, parameter2try):

        parameters_new, accuracy = OptParameters(self, DATA, parameter2try)

        return parameters_new, accuracy

    def CalculatePerformance(self, result, target):

        if self.Model == {}:
            return None
        else:
            return self.Model.CalculatePerformance(result, target)
