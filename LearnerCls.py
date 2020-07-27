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
from keras.models import Sequential
from keras.layers import Dense
from sklearn.impute import KNNImputer


def CalculateACC(Result, Target):
    ACC = 0
    for i in range(0, len(Result)):
        if Result[i] == Target[i]:
            ACC += 1
    return ACC / len(Result)


def Split_Data(Data, ratio):
    size = Data.shape[0]
    indx = list(range(0, size))
    rd.shuffle(indx)
    indx_train = indx[:round(size * ratio)]
    indx_test = indx[round(size * ratio):]

    Data_Train = Data.iloc[indx_train, :].reset_index(drop=True)
    Data_Test = Data.iloc[indx_test, :].reset_index(drop=True)
    return Data_Train, Data_Test


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


class Optimize:

    def __init__(self, LearnerObj):

        self.LearnerObj = LearnerObj

    def OptFeatures(self, DATA):

        DATA_Train, DATA_Test = train_test_split(DATA, test_size=0.2, shuffle=True)
        Label = list(DATA.keys())[-1]
        Features = list(DATA.keys())[:-1]
        best_subset = []
        BestResult = 1000
        options = {'FeatureSubset': {}}

        for i in (list(range(1, 3))):
            PossibleCombinations = list(itertools.combinations(Features, i))
            for combination in PossibleCombinations:
                rd.seed(1)
                DATA_Train_new = DATA_Train[list(combination)].copy()
                DATA_Train_new[Label] = DATA_Train[Label]
                DATA_Test_new = DATA_Test[list(combination)].copy()
                DATA_Test_new[Label] = DATA_Test[Label]
                self.LearnerObj.TrainModel(DATA_Train_new, options)
                predictions = self.LearnerObj.Predict(DATA_Test_new.iloc[:, :-1])
                MAE = self.LearnerObj.CalculatePerformance(predictions,
                                                           DATA_Test_new.iloc[:, -1].to_numpy().astype(int))
                if MAE < BestResult:
                    BestResult = MAE
                    best_subset = list(combination)
            print("Done: #" + str(i))
            print("MAE: " + str(BestResult))

        DATA_Train_new = DATA_Train[best_subset].copy()
        DATA_Test_new = DATA_Test[best_subset].copy()
        self.LearnerObj.TrainModel(DATA_Train_new, options)
        predictions = self.LearnerObj.Predict(DATA_Test_new.iloc[:, :-1])
        MAE = self.LearnerObj.CalculatePerformance(predictions, DATA_Test_new.iloc[:, -1].to_numpy().astype(int))

        return best_subset, MAE

    def OptParameters(self, DATA, parameter2try):

        DATA_Train, DATA_Test = train_test_split(DATA, test_size=0.2, shuffle=True)
        allcombinations = parameter2try[list(parameter2try.keys())[0]]
        for k in list(parameter2try.keys())[1:]:
            allcombinations = itertools.product(allcombinations, parameter2try[k])
        BestResult = 1000
        for combination in allcombinations:
            params = {}
            for key in list(parameter2try.keys()):

                if type(combination) is tuple:
                    params[key] = combination[1]
                    combination = combination[0]
                else:
                    params[key] = combination

            self.LearnerObj.TrainModel(DATA_Train, params)
            predictions = self.LearnerObj.Predict(DATA_Test.iloc[:, :-1])
            MAE = self.LearnerObj.CalculatePerformance(predictions, DATA_Test.iloc[:, -1].to_numpy().astype(int))
            print(str(MAE))
            if MAE < BestResult:
                BestResult = MAE
                BestParameters = params

        self.LearnerObj.TrainModel(DATA_Train, BestParameters)
        predictions = self.LearnerObj.Predict(DATA_Test.iloc[:, :-1])
        MAE = self.LearnerObj.CalculatePerformance(predictions, DATA_Test.iloc[:, -1].to_numpy().astype(int))

        return BestParameters, MAE


class RandomForest:

    def __init__(self, args={}):
        self.RandomForest = {}
        self.n_estimators = 100
        self.random_state = 8
        self, options = UpdateOptions(self, args)

    def TrainModel(self, data, args={}):
        self, options = UpdateOptions(self, args)

        train_X = data.iloc[:, :-1]
        train_Y = data.iloc[:, -1]

        self.RandomForest = RandomForestRegressor(n_estimators=self.n_estimators,
                                                  random_state=self.random_state)
        # print(train_X.shape)
        # print(train_Y.shape)
        self.RandomForest.fit(train_X, train_Y)
        return -1

    def Predict(self, data):

        predictions = self.RandomForest.predict(data).astype(int)
        return predictions

    def CalculatePerformance(self, result, target):

        if self.RandomForest == {}:
            return None
        else:
            return eval_measures.meanabs(result, target)


class MLP:

    def __init__(self, args={}):
        self.MLP = {}
        self.inputSize = 0
        self, options = UpdateOptions(self, args)

    def TrainModel(self, data, args={}):

        self.inputSize = data.shape[1] - 1
        self.MLP = Sequential()
        self.MLP.add(Dense(16, input_shape=(self.inputSize,), kernel_initializer='normal', activation='relu'))
        self.MLP.add(Dense(32, kernel_initializer='normal', activation='relu'))
        self.MLP.add(Dense(16, kernel_initializer='normal', activation='relu'))
        self.MLP.add(Dense(1, kernel_initializer='normal', activation='linear'))
        self.MLP.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        # early_stopping_monitor = EarlyStopping(patience=3)
        # model.fit(train_X, train_y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])

        self.MLP.fit(data.iloc[:, :-1], data.iloc[:, -1], epochs=100, batch_size=32, validation_split=0.1)

        return self.MLP

    def Predict(self, data):
        predictions = self.MLP.predict(data).astype(int)
        return np.reshape(predictions, [len(predictions)])

    def CalculatePerformance(self, result, target):

        if self.MLP == {}:
            return None
        else:
            return eval_measures.meanabs(result, target)


class NegativeBinomial:

    def __init__(self, args={}):
        self.NegativeBinomial = {}
        self.Results = {}
        self.alpha = 100
        self.ModelFormula = {}
        self, options = UpdateOptions(self, args)

    def extractFormula(self, data):
        columns = data.columns
        ModelFormula = columns[-1] + ' ~ '
        for c in columns[:-2]:
            ModelFormula += str(c) + ' + '

        ModelFormula += str(columns[-2])
        self.ModelFormula = ModelFormula
        return self.ModelFormula

    def TrainModel(self, data, args={}):

        self, options = UpdateOptions(self, args)

        self.NegativeBinomial = smf.glm(formula=self.extractFormula(data),
                                        data=data,
                                        family=sm.families.NegativeBinomial(alpha=self.alpha))
        self.Results = self.NegativeBinomial.fit()
        return -1

    def Predict(self, data):
        predictions = self.Results.predict(data).astype(int)
        return predictions

    def CalculatePerformance(self, result, target):

        if self.NegativeBinomial == {}:
            return None
        else:
            return eval_measures.meanabs(result, target)


class MultipleRandomForest:

    def __init__(self, NumberOfModels=50, NumberOfOutModels=5, args={}):

        self.NumberOfModels = NumberOfModels
        self.NumberOfOutModels = NumberOfOutModels
        self.Models = {}
        self.Accuracy = []

        self, options = UpdateOptions(self, args)

    def TrainModel(self, DATA, args={}):
        np.random.seed(1)
        self, options = UpdateOptions(self, args)

        self.Models = {}
        self.Accuracy = []
        DATA_X = DATA.iloc[:, :-1]
        DATA_Y = DATA.iloc[:, -1]
        models = {}
        acc = []
        for i in range(0, self.NumberOfModels):
            newData = np.random.randint(DATA.shape[0], size=DATA.shape[0])
            newTest = np.delete(np.arange(0, DATA.shape[0]), pd.unique(newData))

            data = DATA.iloc[newData, :].reset_index(drop=True)

            models[i] = RandomForest()
            models[i].TrainModel(data)

            tst_X = DATA_X.iloc[newTest, :].reset_index(drop=True)
            tst_Y = DATA_Y.iloc[newTest].reset_index(drop=True)

            predictions = models[i].Predict(tst_X).astype(int)
            acc.append(eval_measures.meanabs(predictions, tst_Y))

        acc = np.asarray(acc)

        for i in range(0, self.NumberOfOutModels):
            index = acc.argmin()
            self.Accuracy.append(acc[index])
            acc[index] = acc.max()
            self.Models[i] = models[index]

        self.Accuracy = np.asarray(self.Accuracy)

        # print(str(self.Accuracy.mean()))

        return self.Accuracy.mean()

    def Predict(self, DATA):
        df = DATA.copy()

        output = pd.DataFrame()
        for i in range(0, len(self.Models)):
            output.insert(i, str(i), self.Models[i].Predict(df).astype(int))

        output['out'] = output.mode(axis=1).iloc[:, 0]
        return output['out'].to_numpy().astype(int)
        return self.Model.Predict(DATA)

    def CalculatePerformance(self, result, target):

        if self.Models == {}:
            return None
        else:
            return eval_measures.meanabs(result, target)


class Train:

    def __init__(self, Model, args={}):

        # Initialize:
        self.Model = Model
        self.Validation = 'CrossValidation'
        self.Shuffle = 'Yes'
        self.NFolds = 5
        self.TrainRatio = 0.8

        self, out = UpdateOptions(self, args)

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

    def TrainModel(self, Data, args={}):

        self, options = UpdateOptions(self, args)

        # Prepare Data:

        TrainingData, ValidationData = self.SplitData4Validation(Data)

        # Train Model:

        if self.TrainRatio == 1:
            self.Model.TrainModel(TrainingData.get(0), options)
            return self.Model, -1
        else:
            Performance = []
            for i in range(0, self.NFolds):
                self.Model.TrainModel(TrainingData.get(i), options)
                Performance.append(self.Model.CalculatePerformance(
                    self.Model.Predict(ValidationData.get(i).iloc[:, :-1]),
                    ValidationData.get(i).iloc[:, -1]))

            self.Model.TrainModel(TrainingData.get(np.asarray(Performance).argmin()), options)

            return self.Model, statistics.mean(Performance)


class Learner:

    def __init__(self, method, args={}):

        self, options = UpdateOptions(self, args)

        if method == 'NegativeBinomial':
            self.Method = 'NegativeBinomial'
            self.Model = NegativeBinomial(options)

        elif method == 'RandomForest':
            self.Method = 'RandomForest'
            self.Model = RandomForest(options)

        elif method == 'MLP':
            self.Method = 'MLP'
            self.Model = MLP(options)

        elif method == 'MultipleRandomForest':
            self.Method = 'MultipleRandomForest'
            self.Model = MultipleRandomForest(50, 10, options)

        self.Train = Train(self.Model, options)
        self.Optimize = Optimize(self)
        self.FeatureSubset = []

    def TrainModel(self, DATA, args={}):

        self, options = UpdateOptions(self, args)

        if len(self.FeatureSubset) != 0:
            DATA_new = DATA[self.FeatureSubset].copy()
        else:
            DATA_new = DATA.copy()

        self.Model, accuracy = self.Train.TrainModel(DATA_new, options)
        return accuracy

    def Predict(self, DATA):

        return self.Model.Predict(DATA)

    def OptimizeFeatures(self, DATA):

        features, accuracy = self.Optimize.OptFeatures(DATA)
        self.FeatureSubset = features
        print("Feature subset that used updated.")
        return features, accuracy

    def OptimizeModelParameters(self, DATA, parameter2try):

        parameters_new, accuracy = self.Optimize.OptParameters(DATA, parameter2try)

        return parameters_new, accuracy

    def CalculatePerformance(self, result, target):

        if self.Model == {}:
            return None
        else:
            return self.Model.CalculatePerformance(result, target)


def fillNans(df_train, df_test):
    idf_train = df_train.copy()
    idf_train.drop('total_cases', axis=1, inplace=True)
    idf_train['source'] = 'Train'

    idf_test = df_test.copy()
    idf_test['source'] = 'Test'

    idf_sum = pd.concat([idf_train, idf_test]).reset_index(drop=True)

    idf_sum_in = idf_sum.copy()
    idf_sum_in.drop('source', axis=1, inplace=True)

    imputer = KNNImputer(n_neighbors=5, weights="distance")
    idf = pd.DataFrame(imputer.fit_transform(idf_sum_in))
    idf.columns = idf_sum_in.columns
    idf.index = idf_sum_in.index
    idf['source'] = idf_sum['source']

    idf_out_train = idf[idf.source == 'Train'].reset_index(drop=True).copy()
    idf_out_train.drop('source', axis=1, inplace=True)
    idf_out_train['total_cases'] = df_train['total_cases'].reset_index(drop=True)
    idf_out_test = idf[idf.source == 'Test'].reset_index(drop=True).copy()
    idf_out_test.drop('source', axis=1, inplace=True)
    return idf_out_train, idf_out_test
