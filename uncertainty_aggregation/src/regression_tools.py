from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import numpy as np
import keras
import keras.backend as K


n_ensemble = 10


def shallow_net_stddev(y_true, y_pred):
    return K.sqrt(keras.losses.mean_squared_error(y_true, y_pred))


def shallow_net_model(num_variables=1, l2_reg_lambda=0.005):
    reg = regularizers.l2(l2_reg_lambda)

    model = Sequential()
    model.add(Dense(units=61, activation='relu',
              kernel_regularizer=reg, input_dim=num_variables))
    model.add(Dense(units=61, activation='relu', kernel_regularizer=reg))
    model.add(Dense(units=1, kernel_regularizer=reg, activation='linear'))

    return model


def meta_classifications(md_moped, gauge=0.5, threshs=np.arange(0.05, 0.95, 0.1)):
    logistic_df = md_moped.thresh_classification(
        thresholds=threshs, method='logistic', num_ensemble=n_ensemble)
    d, n = md_moped.classification_best_gb_parameters(
        threshold=gauge, num_ensemble=5)
    gradient_boost_df = md_moped.thresh_classification(
        thresholds=threshs, method='gradient_boost', num_ensemble=n_ensemble, gb_depth=d, gb_n_estimators=n)

    return logistic_df, gradient_boost_df


def meta_regressions(md_moped):
    methods = ['linear', 'shallow_nn', 'gradient_boost']
    results = []

    for method in methods:
        if method == 'gradient_boost':
            d, n = md_moped.regression_best_gb_parameters(num_ensemble=5)
            results.append(md_moped.regression(
                method=method, num_ensemble=n_ensemble, gb_depth=d, gb_n_estimators=n))
        else:
            results.append(md_moped.regression(
                method=method, num_ensemble=n_ensemble))

    return dict(zip(methods, results))
