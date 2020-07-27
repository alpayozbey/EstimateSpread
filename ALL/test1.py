import pandas as pd
import temporr2 as RL
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

BestParams_sj = {'n_estimators': 70, 'random_state': 120}
BestParams_iq = {'n_estimators': 40, 'random_state': 70}


RF1 = RL.Learner('RandomForest', BestParams_sj)
Features_sj, acc = RF1.OptimizeFeatures(D_train_sj)

#%%
features = ['reanalysis_specific_humidity_g_per_kg',
            'reanalysis_dew_point_temp_k',
            'station_avg_temp_c',
            'station_min_temp_c',
            list(D_train_sj.keys())[-1]]

RF1 = RL.Learner('RandomForest')
A = np.arange(10, 200, 10, dtype=np.int)
B = np.arange(10, 200, 10, dtype=np.int)
BestParams, acc = RF1.OptimizeModelParameters(D_train_sj[features],
                                              {'n_estimators': A,
                                               'random_state': B})
#%%

#%%
RF2 = RL.Learner('RandomForest')
A = np.arange(10, 200, 10, dtype=np.int)
B = np.arange(10, 200, 10, dtype=np.int)
BestParams, acc = RF2.OptimizeModelParameters(D_train_iq[features],
                                              {'n_estimators': A,
                                               'random_state': B})
#%%
BestParams_sj = {'n_estimators': 70, 'random_state': 120}
BestParams_iq = {'n_estimators': 40, 'random_state': 70}


RF1 = RL.Learner('RandomForest', BestParams_sj)
Features_sj, acc = RF1.OptimizeFeatures(D_train_sj)

#%%
MLP = RL.Learner('MLP')
MLP.TrainModel(D_train_sj)
predict_sj = MLP.Predict(D_test_sj)

#%%
MRF1 = RL.Learner('MultipleRandomForest')
# A = np.arange(10, 200, 10, dtype=np.int)
# B = np.arange(10, 200, 10, dtype=np.int)
# BB, acc = NgBin.OptimizeModelParameters(D_train_sj[features], {'n_estimators': A, 'random_state': B})
MRF1.TrainModel(D_train_sj)
predict_sj = MRF1.Predict(D_test_sj)

#%%

MRF2 = RL.Learner('MultipleRandomForest')
MRF2.TrainModel(D_train_iq)
predict_iq = MRF2.Predict(D_test_iq)


# features, accuracy = NgBin.OptimizeFeatures(D_train_sj)

#%%
sj_out = pd.DataFrame()
sj_out['year'] = D_test_sj['year'].astype(int)
sj_out['weekofyear'] = D_test_sj['weekofyear'].astype(int)
sj_out['city'] = 'sj'
sj_out['total_cases'] = predict_sj.astype(int)
sj_out = sj_out[['city', 'year', 'weekofyear', 'total_cases']]

iq_out = pd.DataFrame()
iq_out['year'] = D_test_iq['year'].astype(int)
iq_out['weekofyear'] = D_test_iq['weekofyear'].astype(int)
iq_out['city'] = 'iq'
iq_out['total_cases'] = predict_iq.astype(int)
iq_out = iq_out[['city', 'year', 'weekofyear', 'total_cases']]

DF = pd.concat([sj_out, iq_out])
DF.to_csv("Submission_out.csv", index=False)

