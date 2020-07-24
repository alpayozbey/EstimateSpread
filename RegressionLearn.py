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

eps = np.finfo(float).eps


def FindEntropy(df):
    Targets = df.keys()[-1]
    entropy = 0
    values = df[Targets].unique()
    for value in values:
        fraction = df[Targets].value_counts()[value] / len(df[Targets])
        entropy += -fraction * np.log2(fraction)
    return entropy


def FindEntropyOfFeature(df, feature):
    Targets = df.keys()[-1]
    target_variables = df[Targets].unique()
    variables = df[
        feature].unique()
    entropy_f = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[feature][df[feature] == variable][df[Targets] == target_variable])
            den = len(df[feature][df[feature] == variable])
            fraction = num / (den + eps)
            entropy += -fraction * log(fraction + eps)
        fraction_f = den / len(df)
        entropy_f += -fraction_f * entropy
    return abs(entropy_f)


def FindBest(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
        #         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(FindEntropy(df) - FindEntropyOfFeature(df, key))
    return df.keys()[:-1][np.argmax(IG)]


def CreateSubTable(df, node, value):
    return df[df[node] == value].reset_index(drop=True), df[df[node] == value].index


def buildIDLOG(df, tree=None, Level=0, d=math.inf):
    Targets = df.keys()[-1]
    root = FindBest(df)  # root of that level

    values = np.unique(df[root])

    if tree is None:
        tree = {}
        tree[root] = {}

    for value in values:

        subtable, indx = CreateSubTable(df, root, value)
        clValue, counts = np.unique(subtable[Targets], return_counts=True)

        if len(counts) == 1:
            tree[root][value] = clValue[0]
        else:
            if Level < d:
                Level = Level + 1
                tree[root][value] = buildIDLOG(subtable, None, Level, d)  # generate and connect a subtree on subset
            else:
                if Level == d:
                    tree[root][value] = LogisticRegression()
                    tree[root][value].fit(subtable.iloc[:, :-1], subtable.iloc[:, -1])

    return tree


def ClassifyIDLog(tree, df):
    if df.shape[0] == 0:
        return None
    else:

        Output = np.empty(df.shape[0])
        Output[:] = np.nan

        TM = tree
        if len(tree) == 1:
            for key in TM.keys():
                TM = TM[key]
                for value in TM.keys():
                    T = TM[value]
                    stbl, indx = CreateSubTable(df, key, value)
                    if stbl.shape[0] != 0 and stbl.ndim != 1 and stbl.size != 0:

                        A = str(type(T)).split('class \'')[1].split('\'>')[0]
                        if A == 'dict':
                            Output[indx] = ClassifyIDLog(T, stbl)
                        elif A == 'numpy.int64':
                            Output[indx] = T
                        elif A == 'sklearn.linear_model._logistic.LogisticRegression':
                            # print(stbl)
                            Output[indx] = T.predict(stbl)

                    # else:
                    # print('Yes')

        return Output.astype('int64')


def FindOutProb(df):
    if df is None or df.size == 0:
        return None
    Targets = df.keys()[-1]
    prob = 0
    values = df[Targets].unique()
    prob = {}
    for value in values:
        fraction = df[Targets].value_counts()[value] / len(df[Targets])
        prob[value] = fraction
    return prob


def FindLHProbFeat(df, feature):
    Targets = df.keys()[-1]
    target_variables = df[Targets].unique()
    variables = df[
        feature].unique()
    prob = {}

    for variable in variables:
        prob[variable] = {}
        for target_variable in target_variables:
            num = len(df[feature][df[feature] == variable][df[Targets] == target_variable])
            den = len(df[feature][df[Targets] == target_variable])
            fraction = num / den
            prob[variable][target_variable] = fraction

    return prob


def BuildNaiveBayes(df):
    p = FindOutProb(df)

    like_p = {}
    for c in df[:-1].keys():
        like_p[c] = FindLHProbFeat(df, c)

    mdl = {'p_target': p, 'p_likelihood': like_p}
    return mdl


def ClassifyNaiveBayes(mdl, df):
    if mdl == {}:
        return None
    if df.size == 0:
        return None

    p = mdl['p_target']
    p_like = mdl['p_likelihood']

    output = []

    for i in df.index:

        p_output = {}

        for target in p.keys():
            prob = p[target]

            for feature in df.keys():

                if feature in p_like.keys():
                    if df[feature][i] in p_like[feature].keys():
                        prob = prob * p_like[feature][df[feature][i]][target]

            p_output[target] = prob

        output.append(max(p_output))

    return output


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

    def __init__(self, Model):

        self.Model = Model

    def OptFeatures(self, DATA):

        DATA_Train, DATA_Test = train_test_split(DATA, test_size=0.2, shuffle=True)
        Features = list(DATA.keys())[:-1]
        best_subset = []
        BestResult = 1000
        for i in range(len(Features), 0, -1):
            PossibleCombinations = list(itertools.combinations(Features, i))
            for combination in PossibleCombinations:
                DATA_Train_new = DATA_Train[list(combination)].copy()
                DATA_Test_new = DATA_Test[list(combination)].copy()
                self.Model.Train(DATA_Train_new, {'ModelFormula': {}})
                predictions = self.Model.Predict(DATA_Test_new)
                MAE = self.Model.CalculatePerformance(predictions, DATA_Test_new.iloc[:, -1].to_numpy().astype(int))
                if MAE < BestResult:
                    BestResult = MAE
                    best_subset = list(combination)
                    print("Done: #" + str(i))
                    print("MAE: " + str(BestResult))
            print("Done: #" + str(i))
            print("MAE: " + str(BestResult))

        DATA_Train_new = DATA_Train[best_subset].copy()
        DATA_Test_new = DATA_Test[best_subset].copy()
        self.Model.Train(DATA_Train_new, {'ModelFormula': {}})
        predictions = self.Model.Predict(DATA_Test_new)
        MAE = self.Model.CalculatePerformance(predictions, DATA_Test_new.iloc[:, -1].to_numpy().astype(int))

        return best_subset, MAE

    def OptParameters(self, DATA, parameter2try):

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

            self.Model.Train(DATA_Train, params)
            predictions = self.Model.Predict(DATA_Test)
            MAE = self.Model.CalculatePerformance(predictions, DATA_Test.iloc[:, -1].to_numpy().astype(int))

            if MAE < BestResult:
                BestResult = MAE
                BestParameters = params

        self.Model.Train(DATA_Train, BestParameters)
        predictions = self.Model.Predict(DATA_Test)
        MAE = self.Model.CalculatePerformance(predictions, DATA_Test.iloc[:, -1].to_numpy().astype(int))

        return BestParameters, MAE


class IDLog:

    def __init__(self, args={}):
        self.Tree = None
        self.treeDepth = 3
        self.__dict__.update(args)

    def Train(self, data, args={}):
        self.__dict__.update(args)
        self.Tree = buildIDLOG(data, None, 0, self.treeDepth)

    def Predict(self, data):
        return ClassifyIDLog(self.Tree, data)

    def CalculatePerformance(self, result, target):

        if self.Tree == {}:
            return None
        else:
            return CalculateACC(result, target)


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
        self.Optimize = Optimize(self)
        self.__dict__.update(args)

    def extractFormula(self, data):
        columns = data.columns
        ModelFormula = columns[-1] + ' ~ '
        for c in columns[:-2]:
            ModelFormula += str(c) + ' + '

        ModelFormula += str(columns[-2])
        self.ModelFormula = ModelFormula

    def Train(self, data, args={}):
        self.__dict__.update(args)

        if len(self.ModelFormula) == 0:
            self.extractFormula(data)

        self.NegativeBinomial = smf.glm(formula=self.ModelFormula,
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

    def __init__(self, args={}):

        # Initialize:
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
            if i == 0:
                TrainingData[i] = Data.iloc[train_indx, :].reset_index(drop=True)
            else:
                TrainingData[i] = pd.concat([TrainingData[i - 1], Data.iloc[train_indx, :].reset_index(drop=True)],
                                            ignore_index=True)

            start = stop

        return TrainingData, ValidationData

    def TrainModel(self, Model, Data):

        # Prepare Data:

        TrainingData, ValidationData = self.SplitData4Validation(Data)

        # Train Model:

        if self.TrainRatio == 1:
            Model.Train(TrainingData.get(0))
            return Model, -1
        else:
            Performance = []
            for i in range(0, self.NFolds):
                Model.Train(TrainingData.get(i))
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

    def TrainModel(self, DATA):

        self.Model, accuracy = self.Train.TrainModel(self.Model, DATA)

        return accuracy

    def Predict(self, DATA):

        return self.Model.Predict(DATA)

    def OptimizeFeatures(self, DATA):

        features, accuracy = self.Model.Optimize.OptFeatures(DATA)

        return features, accuracy

    def OptimizeModelParameters(self, DATA, parameter2try):

        parameters_new, accuracy = self.Model.Optimize.OptParameters(DATA, parameter2try)

        return parameters_new, accuracy
