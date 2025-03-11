from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
# import plot
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
pd.options.mode.chained_assignment = None
def observational_sampling(df, biasing_covariate=None, bias_strength=1, treatment_ids=[0, 3], plot_folder=None, features_dim=3):
    df.sort_values(by=["query_id", "treatment_id"], inplace=True)
    cov = df[biasing_covariate].values
    
    cov = cov[::2]
    ecdf = ECDF(cov)
    cov_ecdf = ecdf(cov)
    cov_ecdf = cov_ecdf - np.mean(cov_ecdf)
    coefficients = np.repeat(bias_strength, len(cov))
    prob_values = 1 / (1 + np.exp(-coefficients * cov_ecdf))
    prob_values = np.clip(prob_values, 0.001, 0.999)
    assigned_treatment_ids = np.random.binomial(1, prob_values)
    assigned_treatment_ids = np.where(assigned_treatment_ids == 1, treatment_ids[0], treatment_ids[1])
    assigned_treatment_ids = np.repeat(assigned_treatment_ids, 2)
    df["assigned_treatment_id"] = assigned_treatment_ids

    if plot_folder is not None:
        plt.figure(figsize=(10, 10))
        plt.scatter(cov, prob_values)
        plt.xlabel(biasing_covariate)
        plt.ylabel("prob_values")
        plot_dir = "{}/{}".format(plot_folder, "prob_values_vs_covariate_values")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/prob_values_vs_covariate_values_bias_strength_{bias_strength}.png")

    df_sampled = df[df["treatment_id"] == df["assigned_treatment_id"]]
    df_cf_sampled = df[~df.index.isin(df_sampled.index)]
    # drop treatment_id from df_sampled and df_cf_sampled
    df_sampled.drop(columns=["treatment_id"], inplace=True)
    df_cf_sampled.drop(columns=["treatment_id"], inplace=True)
    # rename assigned_treatment_id to treatment_id
    df_sampled.rename(columns={"assigned_treatment_id": "treatment_id"}, inplace=True)
    df_cf_sampled.rename(columns={"assigned_treatment_id": "treatment_id"}, inplace=True)

    # drop assigned_treatment_id from df
    df.drop(columns=["assigned_treatment_id"], inplace=True)
    return df_sampled, df_cf_sampled