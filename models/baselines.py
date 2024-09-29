from econml.dml import NonParamDML
from sklearn.ensemble import RandomForestRegressor
from econml.metalearners import XLearner
from catenets.models.jax import TNet, SNet, DRNet, SNet1, SNet2
import pandas as pd
import numpy as np
import tensorflow as tf

def create_model(input_dim, output_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_dim=input_dim, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_dim)
    ])
    
    optimizer = tf.keras.optimizers.Adam(0.01)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model

def random_forest(df, covariates, treatment, outcome, X_test=None):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    T = df[treatment].values
    Y = df[outcome].values

    # concatenate the features and treatment
    X_T = np.concatenate([X, T[:, None]], axis=1)
    # Fit the random forest models
    est = RandomForestRegressor(n_estimators=100, max_depth=5)
    est.fit(X, Y)

    if X_test is None:
        X_test = X
    else:
        X_test = X_test
    X_0 = np.concatenate([X_test, np.zeros((X_test.shape[0], 1))], axis=1)
    X_1 = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=1)
    # Predict the outcomes
    y1 = est.predict(X_1)
    y0 = est.predict(X_0)

    return y1, y0

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

# TODO: Add preprocessing steps before running the model
# Non-parametric DML
def non_param_DML(df, covariates, treatment, outcome, X_test = None):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    T = df[treatment].values
    Y = df[outcome].values

    est = NonParamDML(model_y=RandomForestRegressor(),
                  model_t=RandomForestRegressor(),
                  model_final=RandomForestRegressor())
    est.fit(Y, T, X=X)

    # if X_test is None, then evaluate on the training data
    if X_test is None:
        X_test = X
    else:
    # otherwise evaluate on the test data
        X_test = X_test
    
    causal_effect_estimates = est.effect(X_test, T0=0, T1=1)

    # this method only returns the effect estimates
    return causal_effect_estimates

# X-learner
def x_learner(df, covariates, treatment, outcome, X_test = None):
    # Split the data into features, treatment, and outcome
    X = df[covariates].values
    
    if len(covariates) == 1:
        X = X.reshape(-1, 1)
    
    T = df[treatment].values
    Y = df[outcome].values   
    est = XLearner(models=[RandomForestRegressor(), RandomForestRegressor()])
    est.fit(Y, T, X=X)

    # if X_test is None, then evaluate on the training data
    if X_test is None:
        X_test = X
    else:
    # otherwise evaluate on the test data
        X_test = X_test
        if len(covariates) == 1:
            X_test = X_test.reshape(-1, 1)
    causal_effect_estimates = est.effect(X_test)
    
    return causal_effect_estimates

# CATENETS
