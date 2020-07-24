import pandas as pd
import RegressionLearn as RL
import numpy as np


D_train_features = pd.read_csv('dengue_features_train.csv')
D_test = pd.read_csv('dengue_features_test.csv')
D_train_labels = pd.read_csv('dengue_labels_train.csv')

D_train = pd.merge(D_train_features, D_train_labels, on=['city', 'year', 'weekofyear'])

D_train_sj = D_train[D_train.city == 'sj'].copy()
D_train_iq = D_train[D_train.city == 'iq'].copy()

D_test_sj = D_test[D_test.city == 'sj'].copy()
D_test_iq = D_test[D_test.city == 'iq'].copy()

D_train_sj.fillna(method='ffill', inplace=True)
D_train_iq.fillna(method='ffill', inplace=True)

D_test_sj.fillna(method='ffill', inplace=True)
D_test_iq.fillna(method='ffill', inplace=True)

D_train_sj.drop('city', axis=1, inplace=True)
D_train_sj.drop('week_start_date', axis=1, inplace=True)

D_test_sj.drop('city', axis=1, inplace=True)
D_test_sj.drop('week_start_date', axis=1, inplace=True)

D_train_iq.drop('city', axis=1, inplace=True)
D_train_iq.drop('week_start_date', axis=1, inplace=True)

D_test_iq.drop('city', axis=1, inplace=True)
D_test_iq.drop('week_start_date', axis=1, inplace=True)

NgBin = RL.Learner('NegativeBinomial')
R = 10 ** np.arange(-1, -10, -0.2, dtype=np.float64)
# BB, acc = NgBin.OptimizeModelParameters(D_train_sj, {'alpha': R})

features, accuracy = NgBin.OptimizeFeatures(D_train_sj)



