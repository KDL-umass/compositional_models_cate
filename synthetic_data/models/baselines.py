from econml.dml import NonParamDML
from sklearn.ensemble import RandomForestRegressor
from econml.metalearners import XLearner, TLearner, SLearner
from catenets.models.jax import TNet, SNet, DRNet, SNet1, SNet2
import pandas as pd
import numpy as np
import tensorflow as tf

def x_learner(df, covariates, treatment, outcome, X_test = None, output_scaler=None):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    
    if len(covariates) == 1:
        X = X.reshape(-1, 1)
    
    T = df[treatment].values
    Y = df[outcome].values   
    est = XLearner(models=[RandomForestRegressor(n_estimators=100), RandomForestRegressor(n_estimators=100)])
    est.fit(Y, T, X=X)

    # if X_test is None, then evaluate on the training data
    if X_test is None:
        X_test = X
    else:
    # otherwise evaluate on the test data
        X_test = X_test[covariates].values
        if len(covariates) == 1:
            X_test = X_test.reshape(-1, 1)
    causal_effect_estimates = est.effect(X_test)

    # scale back the output if output_scaler is provided
    if output_scaler is not None:
        causal_effect_estimates = output_scaler.inverse_transform(causal_effect_estimates.reshape(-1, 1)).flatten()
    
    return causal_effect_estimates

def s_learner(df, covariates, treatment, outcome, X_test = None, output_scaler=None):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    
    if len(covariates) == 1:
        X = X.reshape(-1, 1)
    
    T = df[treatment].values
    Y = df[outcome].values   
    est = XLearner(models=RandomForestRegressor(n_estimators=100))
    est.fit(Y, T, X=X)

    # if X_test is None, then evaluate on the training data
    if X_test is None:
        X_test = X
    else:
    # otherwise evaluate on the test data
        X_test = X_test[covariates].values
        if len(covariates) == 1:
            X_test = X_test.reshape(-1, 1)
    causal_effect_estimates = est.effect(X_test)

    # scale back the output if output_scaler is provided
    if output_scaler is not None:
        causal_effect_estimates = output_scaler.inverse_transform(causal_effect_estimates.reshape(-1, 1)).flatten()
    
    return causal_effect_estimates

def t_learner(df, covariates, treatment, outcome, X_test = None, output_scaler=None):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    
    if len(covariates) == 1:
        X = X.reshape(-1, 1)
    
    T = df[treatment].values
    Y = df[outcome].values   
    est = TLearner(models=[RandomForestRegressor(n_estimators=100), RandomForestRegressor(n_estimators=100)])
    est.fit(Y, T, X=X)

    # if X_test is None, then evaluate on the training data
    if X_test is None:
        X_test = X
    else:
    # otherwise evaluate on the test data
        X_test = X_test[covariates].values
        if len(covariates) == 1:
            X_test = X_test.reshape(-1, 1)
    causal_effect_estimates = est.effect(X_test)

    # scale back the output if output_scaler is provided
    if output_scaler is not None:
        causal_effect_estimates = output_scaler.inverse_transform(causal_effect_estimates.reshape(-1, 1)).flatten()
    
    return causal_effect_estimates

def non_param_DML(df, covariates, treatment, outcome, X_test = None, output_scaler=None):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    T = df[treatment].values
    Y = df[outcome].values

    est = NonParamDML(model_y=RandomForestRegressor(n_estimators=100, max_depth=10),
                  model_t=RandomForestRegressor(n_estimators=100, max_depth=10),
                  model_final=RandomForestRegressor(n_estimators=100, max_depth=10))
    est.fit(Y, T, X=X)

    # if X_test is None, then evaluate on the training data
    if X_test is None:
        X_test = X
    else:
    # otherwise evaluate on the test data
        X_test = X_test[covariates].values
    
    causal_effect_estimates = est.effect(X_test, T0=0, T1=1)
    # scale back the output if output_scaler is provided
    if output_scaler is not None:
        causal_effect_estimates = output_scaler.inverse_transform(causal_effect_estimates.reshape(-1, 1)).flatten()

    # this method only returns the effect estimates
    return causal_effect_estimates

def random_forest(df, covariates, treatment, outcome, X_test=None, output_scaler=None):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    T = df[treatment].values
    Y = df[outcome].values
    print(X.shape, T.shape, Y.shape)

    # concatenate the features and treatment
    X_T = np.concatenate([X, T[:, None]], axis=1)
    # Fit the random forest models
    est = RandomForestRegressor(n_estimators=100, max_depth=10)
    est.fit(X_T, Y)

    if X_test is None:
        X_test = X
    else:
        X_test = X_test[covariates].values
    X_0 = np.concatenate([X_test, np.zeros((X_test.shape[0], 1))], axis=1)
    X_1 = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=1)
    # Predict the outcomes
    y1 = est.predict(X_1)
    y0 = est.predict(X_0)

    if output_scaler is not None:
        y1 = output_scaler.inverse_transform(y1.reshape(-1, 1)).flatten()
        y0 = output_scaler.inverse_transform(y0.reshape(-1, 1)).flatten()
    causal_effect_estimates = y1 - y0

    return causal_effect_estimates, y1, y0



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
    
    # Concatenate normalized X and T
    input_dim = X.shape[1]
    X_T = np.concatenate([X, T.reshape(-1, 1)], axis=1)
    
    model = create_model(input_dim+1, 1)
    # Train the model with min-batch training
    model.fit(X_T, Y, epochs=epochs, batch_size=batch_size)
    
    if X_test is None:
        X_test = X
    else:
        X_test = X_test
    
    X_test_1 = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=1)
    X_test_0 = np.concatenate([X_test, np.zeros((X_test.shape[0], 1))], axis=1)
    
    # Predict the normalized log-transformed outputs
    y1 = model.predict(X_test_1)
    y0 = model.predict(X_test_0)
    return y1, y0

