import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import os
import json
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import seaborn as sns
from utils import observational_sampling
import sys


# set relative path 
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = "{}/data/csvs".format(ROOT_DIR)
plot_dir = "{}/plots".format(ROOT_DIR)

# CONFIG = {
#     "data_folder_name": "data_manjaro",
#     "main_research_dir": "/Users/ppruthi/research/novelty_accommodation/",
#     "modeling_dir": "/Users/ppruthi/research/novelty_accommodation/modeling"
# }

expressions = [
            "(A + B) * SVD(C)[0] * LU(D)[1] * QR(E)[1] * (F - G) * inv(H) * dot(I, J) * trace(K * L) * det(M) * norm(N, 'fro')",
            "SVD(O)[2] * (P * LU(Q)[0]) * QR(R)[0] * (S + T) * inv(U) * dot(V, W) * trace(X * Y) * det(Z) * norm(A, 'fro')",
            "(B + C) * (D * SVD(E)[1]) * LU(F)[1] * (G * QR(H)[1]) * inv(I) * dot(J, K) * trace(L * M) * det(N) * norm(O, 'fro')",
            "QR(P)[0] * SVD(Q)[0] * (R * LU(S)[0]) * (T + U) * inv(V) * dot(W, X) * trace(Y * Z) * det(A) * norm(B, 'fro')",
            "(C + D) * (E * SVD(F)[2]) * LU(G)[1] * QR(H)[1] * (I - J) * inv(K) * dot(L, M) * trace(N * O) * det(P) * norm(Q, 'fro')",
            "SVD(R)[1] * LU(S)[0] * (T * QR(U)[0]) * (V + W) * inv(X) * dot(Y, Z) * trace(A * B) * det(C) * norm(D, 'fro')",
            "(E + F) * (G * SVD(H)[0]) * LU(I)[1] * QR(J)[1] * (K - L) * inv(M) * dot(N, O) * trace(P * Q) * det(R) * norm(S, 'fro')",
            "QR(T)[0] * SVD(U)[2] * LU(V)[0] * (W * X) * inv(Y) * dot(Z, A) * trace(B * C) * det(D) * norm(E, 'fro')",
            "(F + G) * (H * SVD(I)[1]) * LU(J)[1] * QR(K)[1] * (L - M) * inv(N) * dot(O, P) * trace(Q * R) * det(S) * norm(T, 'fro')",
            "SVD(U)[0] * LU(V)[0] * QR(W)[0] * (X + Y) * inv(Z) * dot(A, B) * trace(C * D) * det(E) * norm(F, 'fro')",
            "(A + B) * C * TR(D) * dot(E, F) * inv(G) * (H + I) * J * TR(K) * dot(L, M) * inv(N) * (O + P) * Q * TR(R) * dot(S, T) * inv(U) * (V + W)",
            "2 * trace(X * Y) * dot(Z, A) * inv(B) * (C + D) * E * TR(F) * dot(G, H) * inv(I) * (J + K) * L * TR(M) * dot(N, O) * inv(P) * (Q + R) * S",
            "det(T) * (U - V) * dot(W, X) * TR(Y) * norm(Z, 'fro') * (A + B) * C * TR(D) * dot(E, F) * inv(G) * (H + I) * J * TR(K) * dot(L, M) * inv(N) * (O + P)",
            "(Q + R) * dot(S, T) * inv(U) * (V * TR(W)) * trace(X * Y) * 3 * (Z + A) * dot(B, C) * inv(D) * (E * TR(F)) * norm(G, 'fro') * (H + I) * J * TR(K) * dot(L, M) * inv(N)",
            "det(O) * (P - Q) * dot(R, S) * TR(T) * (U + V) * dot(W, X) * inv(Y) * (Z * TR(A)) * trace(B * C) * 4 * (D + E) * dot(F, G) * inv(H) * (I * TR(J)) * norm(K, 'fro') * (L + M)",
            "(N * TR(O)) * dot(P, Q) * inv(R) * (S + T) * U * TR(V) * dot(W, X) * inv(Y) * (Z + A) * B * TR(C) * dot(D, E) * inv(F) * (G + H) * I * TR(J) * dot(K, L) * inv(M) * (N + O)",
            "5 * trace(P * Q) * dot(R, S) * inv(T) * (U + V) * W * TR(X) * dot(Y, Z) * inv(A) * (B + C) * D * TR(E) * dot(F, G) * inv(H) * (I + J) * K * TR(L) * dot(M, N) * inv(O) * (P + Q)",
            "det(R) * (S - T) * dot(U, V) * TR(W) * norm(X, 'fro') * (Y + Z) * A * TR(B) * dot(C, D) * inv(E) * (F + G) * H * TR(I) * dot(J, K) * inv(L) * (M + N) * O * TR(P) * dot(Q, R) * inv(S) * (T + U)",
            "(V + W) * dot(X, Y) * inv(Z) * (A * TR(B)) * trace(C * D) * 6 * (E + F) * dot(G, H) * inv(I) * (J * TR(K)) * norm(L, 'fro') * (M + N) * O * TR(P) * dot(Q, R) * inv(S) * (T + U) * V * TR(W) * dot(X, Y) * inv(Z)",
            "det(A) * (B - C) * dot(D, E) * TR(F) * (G + H) * dot(I, J) * inv(K) * (L * TR(M)) * trace(N * O) * 7 * (P + Q) * dot(R, S) * inv(T) * (U * TR(V)) * norm(W, 'fro') * (X + Y) * Z * TR(A) * dot(B, C) * inv(D) * (E + F)",
            "(G + H) * dot(I, J) * inv(K) * (L * TR(M)) * trace(N * O) * 8 * (P + Q) * dot(R, S) * inv(T) * (U * TR(V)) * norm(W, 'fro') * (X + Y) * Z * TR(A) * dot(B, C) * inv(D) * (E + F) * G * TR(H) * dot(I, J) * inv(K)",
            "det(L) * (M - N) * dot(O, P) * TR(Q) * (R + S) * dot(T, U) * inv(V) * (W * TR(X)) * trace(Y * Z) * 9 * (A + B) * dot(C, D) * inv(E) * (F * TR(G)) * norm(H, 'fro') * (I + J) * K * TR(L) * dot(M, N) * inv(O) * (P + Q)",
            "(R + S) * dot(T, U) * inv(V) * (W * TR(X)) * trace(Y * Z) * 10 * (A + B) * dot(C, D) * inv(E) * (F * TR(G)) * norm(H, 'fro') * (I + J) * K * TR(L) * dot(M, N) * inv(O) * (P + Q) * R * TR(S) * dot(T, U) * inv(V)",
            "det(W) * (X - Y) * dot(Z, A) * TR(B) * (C + D) * dot(E, F) * inv(G) * (H * TR(I)) * trace(J * K) * 11 * (L + M) * dot(N, O) * inv(P) * (Q * TR(R)) * norm(S, 'fro') * (T + U) * V * TR(W) * dot(X, Y) * inv(Z) * (A + B)",
            "(C + D) * dot(E, F) * inv(G) * (H * TR(I)) * trace(J * K) * 12 * (L + M) * dot(N, O) * inv(P) * (Q * TR(R)) * norm(S, 'fro') * (T + U) * V * TR(W) * dot(X, Y) * inv(Z) * (A + B) * C * TR(D) * dot(E, F) * inv(G)",
            "det(H) * (I - J) * dot(K, L) * TR(M) * (N + O) * dot(P, Q) * inv(R) * (S * TR(T)) * trace(U * V) * 13 * (W + X) * dot(Y, Z) * inv(A) * (B * TR(C)) * norm(D, 'fro') * (E + F) * G * TR(H) * dot(I, J) * inv(K) * (L + M)",
            "(N + O) * dot(P, Q) * inv(R) * (S * TR(T)) * trace(U * V) * 14 * (W + X) * dot(Y, Z) * inv(A) * (B * TR(C)) * norm(D, 'fro') * (E + F) * G * TR(H) * dot(I, J) * inv(K) * (L + M) * N * TR(O) * dot(P, Q) * inv(R)"]

# T_names = ["data_test", "data_test_manjaro"]

module_files = os.listdir(data_dir)
operators = [file.split(".")[0].split("_")[1] for file in module_files if file.endswith(".csv") and "high_level" not in file]

    
def get_operator_df(operator_name):
    df = pd.read_csv(f'{data_dir}/module_{operator_name}.csv')
    df["matrix_size"] = df["query_id"].apply(lambda x: int(x.split("_")[1]))
    # fill NA values with 0
    df.fillna(0, inplace=True)
    print(operator_name, df.shape)
    return df

def get_high_level_df():
    df = pd.read_csv(f'{data_dir}/maths_evaluation_data_high_level_features.csv')
    return df

# get observational dataset by filtering treatment ids
def get_obs_df(query_treatment_ids, operator_name):
    df = get_operator_df(operator_name)    
    query_ids = query_treatment_ids["query_id"].unique()
    treatment_ids = query_treatment_ids["treatment_id"].unique()
    query_treatment_ids["query_id_treatment_id"] = query_treatment_ids["query_id"].astype(str) + "_" + query_treatment_ids["treatment_id"].astype(str)

    df = df[(df["query_id"].isin(query_ids)) & (df["treatment_id"].isin(treatment_ids))]
    df = df.sort_values(["query_id", "treatment_id"])
    df = df.reset_index(drop=True)

    df_sampled = df.copy()
    df_sampled["query_id_treatment_id"] = df_sampled["query_id"].astype(str) + "_" + df_sampled["treatment_id"].astype(str)
    df_obs = df_sampled[df_sampled["query_id_treatment_id"].isin(query_treatment_ids["query_id_treatment_id"])]
    df_cf = df_sampled[~df_sampled["query_id_treatment_id"].isin(query_treatment_ids["query_id_treatment_id"])]
    return df, df_obs, df_cf
   
def generate_high_level_observational_dataset(treatment_ids=[0, 1], sampling="random", prob_value=0.5, biasing_covariate="num_matmul", bias_strength=1, plot_folder=None):
    df = get_high_level_df()
    df.sort_values(by=["query_id", "treatment_id"], inplace=True)

    # Filter treatment IDs and complex ops
    df = df[df["treatment_id"].isin(treatment_ids)]
    df = df.reset_index(drop=True)
    

    # Have only query IDs that have all treatment IDs
    query_ids = df.groupby("query_id").filter(lambda x: len(x) == len(treatment_ids))["query_id"].unique()
    df = df[df["query_id"].isin(query_ids)]
    df = df.reset_index(drop=True)
    

    # Randomly sample from the treatment IDs per query ID
    if treatment_ids is not None:
        if sampling == "random":
            df_sampled = df.groupby("query_id").sample(n=1, random_state=42)
            df_cf_sampled = df[~df.index.isin(df_sampled.index)]
            # rename treatment_id to assigned_treatment_id
        else:
            if sampling == "random_prob":
                df_sampled, df_cf_sampled = random_sampling(df, prob_value=prob_value, treatment_ids=treatment_ids)
            elif sampling == "observational":
                df_sampled, df_cf_sampled = observational_sampling(df, biasing_covariate=biasing_covariate, bias_strength=bias_strength, treatment_ids=treatment_ids, plot_folder=plot_folder)

        df_sampled = df_sampled.reset_index(drop=True)
        df_cf_sampled = df_cf_sampled.reset_index(drop=True)
    else:
        df_sampled = df
        df_cf_sampled = None

    return df, df_sampled, df_cf_sampled

# # try generating high level observational data
if __name__ == "__main__":
    treatment_ids = [0, 1]
    sampling = "observational"
    biasing_covariate = "matrix_size"
    bias_strength = 1
    plot_folder = plot_dir
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    df, df_sampled, df_cf_sampled = generate_high_level_observational_dataset(treatment_ids=treatment_ids, sampling=sampling, biasing_covariate=biasing_covariate, bias_strength=bias_strength, plot_folder=plot_folder)
    print(df_sampled.shape)
    print(df_cf_sampled.shape)
    print(df.shape)
