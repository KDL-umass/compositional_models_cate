import argparse
import json
import os
import warnings
import sys
import pandas as pd
from sklearn.metrics import r2_score
from domains.synthetic_data_sampler import SyntheticDataSampler
from models.MoE import *
from models.utils import *


warnings.filterwarnings('ignore')

def parse_arguments(jupyter=False):
    parser = argparse.ArgumentParser(description="Train modular neural network architectures and baselines for causal effect estimation")
    parser.add_argument("--domain", type=str, default="synthetic_data", help="Domain")
    parser.add_argument("--biasing_covariate", type=str, default="feature_sum", help="Biasing covariate")
    parser.add_argument("--bias_strength", type=float, default=0, help="Bias strength")
    parser.add_argument("--scale", type=bool, default=False, help="Scale data")
    parser.add_argument("--num_modules", type=int, default=10, help="Number of modules")
    parser.add_argument("--num_feature_dimensions", type=int, default=1, help="Number of feature dimensions")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--composition_type", type=str, default="hierarchical", help="Composition type")
    parser.add_argument("--resample", type=bool, default=True, help="Resample data")
    parser.add_argument("--seed", type=int, default=45, help="Seed for reproducibility")
    parser.add_argument("--fixed_structure", type=bool, default=False, help="Fixed structure flag")
    parser.add_argument("--data_dist", type=str, default="uniform", help="Data distribution")
    parser.add_argument("--heterogeneity", type=float, default=1.0, help="Heterogeneity")
    parser.add_argument("--split_type", type=str, default="ood", help="Split type")
    # hidden_dim
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    # epochs
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    # batch_size
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    # output_dim
    parser.add_argument("--output_dim", type=int, default=1, help="Output dimension")
    # covariates_shared
    parser.add_argument("--covariates_shared", type=bool, default=False, help="Covariates shared")
    # model_class
    parser.add_argument("--underlying_model_class", type=str, default="MLP", help="Model class")
    # run_env
    parser.add_argument("--run_env", type=str, default="local", help="Run environment")
    # use_subset_features
    parser.add_argument("--use_subset_features", type=bool, default=False, help="Use subset features")
    # generate trees systematically for creating OOD data
    parser.add_argument("--systematic", type=bool, default=True, help="Generate trees systematically")
    parser.add_argument("--test_size", type=float, default=0.8, help="Test size")
    parser.add_argument("--model_misspecification", type=bool, default=False, help="Model misspecification")
    if jupyter:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args

def setup_directories(args):
    base_dir = "/Users/ppruthi/research/compositional_models/compositional_models_cate/domains" if args.run_env == "local" else "/work/pi_jensen_umass_edu/ppruthi_umass_edu/compositional_models_cate/domains"
    main_dir = f"{base_dir}/{args.domain}"
    csv_path = f"{main_dir}/csvs/fixed_structure_{args.fixed_structure}_outcomes_{args.composition_type}_systematic_{args.systematic}"
    obs_data_path = f"{main_dir}/observational_data/fixed_structure_{args.fixed_structure}_outcomes_{args.composition_type}_systematic_{args.systematic}"
    scaler_path = f"{obs_data_path}/{args.biasing_covariate}_{args.bias_strength}/{args.split_type}/scalers"
    return main_dir, csv_path, obs_data_path, scaler_path

def simulate_and_prepare_data(args, sampler, csv_path, obs_data_path, scaler_path, num_train_modules=1, test_on_last_depth=False):
    if args.resample:
        sampler.simulate_data()
    sampler.create_observational_data(biasing_covariate=args.biasing_covariate, bias_strength=args.bias_strength)
    sampler.create_iid_ood_split(split_type=args.split_type, num_train_modules=num_train_modules, test_on_last_depth=test_on_last_depth)
        
    data = pd.read_csv(f"{csv_path}/{args.domain}_data_high_level_features.csv")
    df_sampled = pd.read_csv(f"{obs_data_path}/{args.biasing_covariate}_{args.bias_strength}/df_sampled.csv")
    
    return data, df_sampled

def load_train_test_qids(csv_path, args):
    with open(f"{csv_path}/{args.split_type}/train_test_split_qids.json", "r") as f:
        train_test_qids = json.load(f)
    train_qids, test_qids = train_test_qids["train"], train_test_qids["test"]
    return train_qids, test_qids

def load_train_test_data(csv_path, args, df_sampled):
    train_qids, test_qids = load_train_test_qids(csv_path, args)
    train_df = df_sampled[df_sampled["query_id"].isin(train_qids)]
    test_df = df_sampled[df_sampled["query_id"].isin(test_qids)]
    return train_df, test_df, train_qids, test_qids

def train_and_evaluate_model(model, train_df, test_df, covariates, treatment, outcome, epochs, batch_size, num_modules, num_feature_dimensions, train_qids, test_qids,plot=False, model_name="MoE",scheduler_flag=False):
    model, _, _ = train_model(model, train_df, covariates, treatment, outcome, epochs, batch_size, num_modules, num_feature_dimensions, plot=plot, model_name=model_name,scheduler_flag=scheduler_flag)
    
    train_estimates = predict_model(model, train_df, covariates, num_modules, num_feature_dimensions, return_effect=True, model_name=model_name)
    test_estimates = predict_model(model, test_df, covariates, num_modules, num_feature_dimensions, return_effect=True, model_name=model_name)
    train_df.loc[:, "estimated_effect"] = train_estimates
    test_df.loc[:, "estimated_effect"] = test_estimates
    
    estimated_effects_train = get_estimated_effects(train_df, train_qids)
    estimated_effects_test = get_estimated_effects(test_df, test_qids)
    
    return estimated_effects_train, estimated_effects_test

def train_and_evaluate_catenets(train_df, test_df, covariates, treatment, outcome, train_qids, test_qids):
    model = train_catenets(train_df, covariates, treatment, outcome)
    train_estimates = predict_catenets(model, train_df, covariates)
    test_estimates = predict_catenets(model, test_df, covariates)
    train_df.loc[:, "estimated_effect"] = train_estimates
    test_df.loc[:, "estimated_effect"] = test_estimates
    estimated_effects_train = get_estimated_effects(train_df, train_qids)
    estimated_effects_test = get_estimated_effects(test_df, test_qids)
    return estimated_effects_train, estimated_effects_test

def calculate_metrics(gt_effects, estimated_effects):
    pehe_score = pehe(gt_effects, estimated_effects)
    r2_score_val = r2_score(gt_effects, estimated_effects)
    return pehe_score, r2_score_val

def decompose_module_errors(module_csvs, num_modules):
    module_dfs = {}
    module_wise_pehe = []
    cov = {}
    print(module_csvs.keys())

    for module_file, module_csv in module_csvs.items():
        module_id = int(module_file.split("_")[-1])
        print(f"Module {module_id}")
        module_dfs[module_id] = module_csv
        pehe_score = pehe(module_csv['ground_truth_effect'], module_csv['estimated_effect'])
        a = module_csv['ground_truth_effect']
        b = module_csv['estimated_effect']
        module_wise_pehe.append(pehe_score)
        cov[module_id] = a - b

    total_cov = []
    for i in range(1, num_modules + 1):
        for j in range(1, num_modules + 1):
            if i != j:
                total_cov.append(np.mean(cov[i] * cov[j]))

    return  sum(module_wise_pehe), sum(total_cov)


def process_shared_covariates_row_wise(train_df, test_df, args):
    num_modules = [col for col in train_df.columns if col.startswith('num_module_')]
    if args.covariates_shared and args.composition_type == "parallel":
        def process_row(row):
            # Find non-zero modules for this row
            num_modules = [col for col in row.index if col.startswith('num_module_')]
            non_zero_modules = [int(col.split('_')[-1]) for col in num_modules if row[col] != 0]
            first_module = non_zero_modules[0]
            first_module_features = [col for col in row.index if col.startswith(f'module_{first_module}_feature_')]
            if not non_zero_modules:
                raise ValueError(f"No non-zero modules found for row with query_id {row['query_id']}")
            
            # Create a new row with renamed features
            new_row = {}
            for feature in first_module_features:
                feature_id = int(feature.split('_')[-1])
                new_row[f'feature_{feature_id}'] = row[feature]
            
            # Add other necessary columns
            for col in ['query_id', 'treatment_id', 'tree_depth', 'query_output'] + num_modules:
                new_row[col] = row[col]
            
            return pd.Series(new_row)

        # don't include header row
        train_df_processed = train_df.apply(process_row, axis=1)
        test_df_processed = test_df.apply(process_row, axis=1)
        print(f"Train data shape: {train_df_processed.shape}")
        print(f"Test data shape: {test_df_processed.shape}")
        
        
        # Ensure consistent column ordering
        columns_order = [col for col in train_df_processed.columns if col.startswith('feature_')] + ['query_id', 'treatment_id', 'tree_depth', 'query_output'] + num_modules
        train_df_processed = train_df_processed[columns_order]
        test_df_processed = test_df_processed[columns_order]
        
        return train_df_processed, test_df_processed
    else:
        # If covariates are not shared, return the original dataframes
        return train_df, test_df

def combine_model_effects(gt_effects, model_effects, additive_combined_df):
    """
    Combine ground truth effects with estimated effects from all models into a single DataFrame.
    
    :param gt_effects: Dictionary of ground truth effects
    :param model_effects: Dictionary of estimated effects for each model
    :param additive_combined_df: DataFrame with additive model results
    :return: Combined DataFrame with effects from all models
    """
    # Start with ground truth effects
    combined_df = pd.DataFrame({
        'query_id': gt_effects.keys(),
        'ground_truth_effect': gt_effects.values()
    })
    
    # Add effects from each model
    for model_name, effects in model_effects.items():
        combined_df[f'{model_name}_effect'] = [effects[qid] for qid in combined_df['query_id']]
    
    # Add effects from the additive model
    additive_effects = additive_combined_df.set_index('query_id')['estimated_effect']
    additive_gt_effects = additive_combined_df.set_index('query_id')['ground_truth_effect']
    combined_df['Additive_effect'] = combined_df['query_id'].map(additive_effects)
    combined_df['Additive_gt_effect'] = combined_df['query_id'].map(additive_gt_effects)
    
    return combined_df