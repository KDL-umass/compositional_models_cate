from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from econml.dml import NonParamDML

from econml.dml import DML, CausalForestDML
from econml.dr import DRLearner
from econml.metalearners import XLearner, TLearner, SLearner
from econml._cate_estimator import LinearCateEstimator
import pandas as pd
from catenets.models.jax import TNet, SNet, DRNet, SNet1, SNet2
import numpy as np


from sklearn.preprocessing import StandardScaler

def random_forest(df, covariates, treatment, outcome, X_test=None):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    T = df[treatment].values
    Y = df[outcome].values

    # Normalize the features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Fit the random forest models
    est_0 = RandomForestRegressor(n_estimators=100, max_depth=5)
    est_1 = RandomForestRegressor(n_estimators=100, max_depth=5)

    # Split the data based on treatment
    X_T_0 = X_normalized[T == 0]
    X_T_1 = X_normalized[T == 1]
    Y_0 = Y[T == 0]
    Y_1 = Y[T == 1]

    # Train the models
    est_0.fit(X_T_0, Y_0)
    est_1.fit(X_T_1, Y_1)

    if X_test is None:
        X_test_normalized = X_normalized
    else:
        X_test_normalized = scaler.transform(X_test)

    # Predict the outcomes
    y1 = est_1.predict(X_test_normalized)
    y0 = est_0.predict(X_test_normalized)

    return y1, y0


def random_forest_changed_plans(df, covariates, treatment, outcome, X1_test = None, X0_test = None):

    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    T = df[treatment].values
    Y = df[outcome].values

    # # Fit the causal forest model
    # est = CausalForestDML()
    # # Or specify hyperparameters
    est_0 = RandomForestRegressor(n_estimators=100, max_depth=5)
    est_1 = RandomForestRegressor(n_estimators=100, max_depth=5)
    # est = RandomForestRegressor(n_estimators=100, max_depth=5)
    # combine X and T
    X_T_0 = X[T == 0]
    X_T_1 = X[T == 1]
    Y_0 = Y[T == 0]
    Y_1 = Y[T == 1]
    est_0.fit(X_T_0, Y_0)
    est_1.fit(X_T_1, Y_1)
    # X_T = np.concatenate([X, T.reshape(-1, 1)], axis=1)
    # est.fit(X_T, Y)

    # if X_test is None:
    #     X_test_1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    #     X_test_0 = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
    # else:
    #     X_test_1 = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=1)
    #     X_test_0 = np.concatenate([X_test, np.zeros((X_test.shape[0], 1))], axis=1)
    
    # print(X_1[:5], X_0[:5])
    y1 = est_1.predict(X1_test)
    y0 = est_0.predict(X0_test)
    return y1, y0


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_model(input_dim, output_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_dim=input_dim, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_dim)
    ])
    
    optimizer = tf.keras.optimizers.Adam(0.01)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model

def neural_network(df, covariates, treatment, outcome, X_test=None, batch_size=32, epochs=10):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    T = df[treatment].values
    Y = df[outcome].values
    
    # Normalize the features and outcome
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_normalized = scaler_X.fit_transform(X)
    Y_normalized = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()
    
    # Concatenate normalized X and T
    input_dim = X_normalized.shape[1]
    X_T_normalized = np.concatenate([X_normalized, T.reshape(-1, 1)], axis=1)
    
    model = create_model(input_dim+1, 1)
    
    # Train the model with min-batch training
    model.fit(X_T_normalized, Y_normalized, epochs=epochs, batch_size=batch_size)
    
    if X_test is None:
        X_test_normalized = X_normalized
    else:
        X_test_normalized = scaler_X.transform(X_test)
    
    X_test_1 = np.concatenate([X_test_normalized, np.ones((X_test_normalized.shape[0], 1))], axis=1)
    X_test_0 = np.concatenate([X_test_normalized, np.zeros((X_test_normalized.shape[0], 1))], axis=1)
    
    # Predict the normalized log-transformed outputs
    y1_log_normalized = model.predict(X_test_1)
    y0_log_normalized = model.predict(X_test_0)
    
    # Inverse transform the normalized predictions to get the log-transformed outputs
    y1_log = scaler_Y.inverse_transform(y1_log_normalized.reshape(-1, 1)).flatten()
    y0_log = scaler_Y.inverse_transform(y0_log_normalized.reshape(-1, 1)).flatten()
    
    print(y1_log[:5], y0_log[:5])
    
    return y1_log, y0_log

def neural_network_changed_plans(df, covariates, treatment, outcome, X1_test=None, X0_test=None, batch_size=32, epochs=10):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    T = df[treatment].values
    Y = df[outcome].values

    # Normalize the features and outcome
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_normalized = scaler_X.fit_transform(X)
    Y_normalized = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()

    # Split the data into treatment and control groups
    X_T_0 = X_normalized[T == 0]
    X_T_1 = X_normalized[T == 1]
    Y_0 = Y_normalized[T == 0]
    Y_1 = Y_normalized[T == 1]

    # Create separate models for treatment and control groups
    input_dim = X_normalized.shape[1]
    model_0 = create_model(input_dim, 1)
    model_1 = create_model(input_dim, 1)

    # Train the models with min-batch training
    model_0.fit(X_T_0, Y_0, epochs=epochs, batch_size=batch_size)
    model_1.fit(X_T_1, Y_1, epochs=epochs, batch_size=batch_size)

    if X1_test is None:
        X1_test = X_normalized
    else:
        X1_test = scaler_X.transform(X1_test)

    if X0_test is None:
        X0_test = X_normalized
    else:
        X0_test = scaler_X.transform(X0_test)

    # Predict the normalized log-transformed outputs
    y1_log_normalized = model_1.predict(X1_test)
    y0_log_normalized = model_0.predict(X0_test)

    # Inverse transform the normalized predictions to get the log-transformed outputs
    y1_log = scaler_Y.inverse_transform(y1_log_normalized.reshape(-1, 1)).flatten()
    y0_log = scaler_Y.inverse_transform(y0_log_normalized.reshape(-1, 1)).flatten()

    print(y1_log[:5], y0_log[:5])
    return y1_log, y0_log





def non_param_DML(df, covariates, treatment, outcome, X_test = None):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    T = df[treatment].values
    Y = df[outcome].values
    

    # # Fit the causal forest model
    # est = CausalForestDML()
    # # Or specify hyperparameters
    est = NonParamDML(model_y=RandomForestRegressor(),
                  model_t=RandomForestRegressor(),
                  model_final=RandomForestRegressor())
    est.fit(Y, T, X=X)

    if X_test is None:
        X_test = X
    else:
        X_test = X_test
    
    causal_effect_estimates = est.effect(X_test, T0=0, T1=1)

    return causal_effect_estimates

def s_learner(df, covariates, treatment, outcome, X_test = None):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    T = df[treatment].values
    Y = df[outcome].values
    
    # # Fit the causal forest model
    # est = CausalForestDML()
    # # Or specify hyperparameters
    est = SLearner(overall_model=RandomForestRegressor())
    est.fit(Y, T, X=X)

    if X_test is None:
        X_test = X
    else:
        X_test = X_test
    
    causal_effect_estimates = est.effect(X_test)

    return causal_effect_estimates


def x_learner(df, covariates, treatment, outcome, X_test = None):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    T = df[treatment].values
    Y = df[outcome].values
   
    
    # # Fit the causal forest model
    # est = CausalForestDML()
    # # Or specify hyperparameters
    est = XLearner(models=[RandomForestRegressor(), RandomForestRegressor()])
    est.fit(Y, T, X=X)
    if X_test is None:
        X_test = X
    else:
        X_test = X_test
    
    
    causal_effect_estimates = est.effect(X_test)
    
    return causal_effect_estimates

def t_learner(df, covariates, treatment, outcome, X_test = None):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    T = df[treatment].values
    Y = df[outcome].values
    
    # # Fit the causal forest model
    # est = CausalForestDML()
    # # Or specify hyperparameters
    est = TLearner(models=[RandomForestRegressor(), RandomForestRegressor()])
    est.fit(Y, T, X=X)

    if X_test is None:
        X_test = X
    else:
        X_test = X_test
    causal_effect_estimates = est.effect(X_test)

    return causal_effect_estimates


def run_catenets(df, covariates, treatment, outcome, X_test = None, baseline=None):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    T = df[treatment].values
    Y = df[outcome].values

    # Normalize the features and outcome
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_normalized = scaler_X.fit_transform(X)
    Y_normalized = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()

    # Fit the causal forest model
    if baseline == "tnet":
        est = TNet()
    elif baseline == "snet":
        est = SNet()
    elif baseline == "snet1":
        est = SNet1()
    elif baseline == "snet2":
        est = SNet2()
    elif baseline == "drnet":
        est = DRNet()
    est.fit(X_normalized,Y_normalized,T)

    if X_test is None:
        X_test = X_normalized
    else:
        X_test = scaler_X.transform(X_test)

    try:
        causal_effect_estimates, y0, y1 = est.predict(X_test, return_po=True)
        y0 = scaler_Y.inverse_transform(y0.reshape(-1, 1)).flatten()
        y1 = scaler_Y.inverse_transform(y1.reshape(-1, 1)).flatten()
        return causal_effect_estimates, y1, y0
    except:
        causal_effect_estimates = est.predict(X_test)
        return causal_effect_estimates

