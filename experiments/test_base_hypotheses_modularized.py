import argparse
import json
import os
import warnings
import sys
sys.path.append('../')

import pandas as pd
from sklearn.metrics import r2_score

from domains.samplers import SyntheticDataSampler
from models.MoE import MoE, MoELinear
from models.MoE import *
from models.utils import *
from models.additive_parallel_comp_model import get_additive_model_effects
from experiments.exp_utils import parse_arguments, setup_directories, simulate_and_prepare_data, load_train_test_data, train_and_evaluate_model, calculate_metrics, decompose_module_errors
warnings.filterwarnings('ignore')

def main():
    args = parse_arguments()
    print(args)
    main_dir, csv_path, obs_data_path, scaler_path = setup_directories(args)
    module_function_types = ["linear", "quadratic", "mlp", "logarithmic", "exponential", "sigmoid"]
    sampler = SyntheticDataSampler(args.num_modules, args.num_feature_dimensions, args.composition_type, 
                                    args.fixed_structure, args.num_modules, args.num_samples, args.seed, 
                                    args.data_dist, module_function_types=module_function_types, resample=args.resample,
                                    heterogeneity=args.heterogeneity, covariates_shared=args.covariates_shared, 
                                    use_subset_features=args.use_subset_features, systematic=args.systematic, 
                                    run_env=args.run_env)

    data, df_sampled = simulate_and_prepare_data(args, sampler, csv_path, obs_data_path, scaler_path)
    train_df, test_df, train_qids, test_qids = load_train_test_data(csv_path, args, df_sampled)

    train_df, test_df = process_shared_covariates_row_wise(train_df, test_df, args)
    covariates = [x for x in train_df.columns if "feature" in x]


    # covariates = [x for x in train_df.columns if "module_1_feature" in x] if args.covariates_shared else [x for x in train_df.columns if "feature" in x]
    if args.systematic:
        covariates += [x for x in train_df.columns if "num" in x]

    treatment, outcome = "treatment_id", "query_output"

    gt_effects_test = get_ground_truth_effects(data, test_qids)
    gt_effects_train = get_ground_truth_effects(data, train_qids)


    input_dim = len(covariates)
    models = {
        "Baseline": BaselineModel if args.underlying_model_class == "MLP" else BaselineLinearModel,
        "MoE": MoE if args.underlying_model_class == "MLP" else MoELinear
    }

    results = {}
    for model_name, model_class in models.items():
        print(f"Training {model_name} Model")
        if model_name == "Baseline":
            model = model_class(input_dim + 1, (input_dim + 1) * 2, args.output_dim)
        else:
            model = model_class(input_dim + 1, (input_dim + 1) * 2, args.output_dim, args.num_modules)
        
        estimated_effects_train, estimated_effects_test = train_and_evaluate_model(
            model, train_df, test_df, covariates, treatment, outcome, args.epochs, args.batch_size, train_qids, test_qids
        )
        gt_effects_train_values, gt_effects_test_values = np.array(list(gt_effects_train.values())), np.array(list(gt_effects_test.values()))
        estimated_effects_train_values, estimated_effects_test_values = np.array(list(estimated_effects_train.values())), np.array(list(estimated_effects_test.values()))
        results[f"{model_name}_train"] = calculate_metrics(gt_effects_train_values, estimated_effects_train_values)
        results[f"{model_name}_test"] = calculate_metrics(gt_effects_test_values, estimated_effects_test_values)

    # Catenets
    print("Training Catenets Model")
    estimated_effects_train, estimated_effects_test = train_and_evaluate_catenets(train_df, test_df, covariates, treatment, outcome, train_qids, test_qids)
    gt_effects_train_values, gt_effects_test_values = np.array(list(gt_effects_train.values())), np.array(list(gt_effects_test.values()))
    estimated_effects_train_values, estimated_effects_test_values = np.array(list(estimated_effects_train.values())), np.array(list(estimated_effects_test.values()))
    results["Catenets_train"] = calculate_metrics(gt_effects_train_values, estimated_effects_train_values)
    results["Catenets_test"] = calculate_metrics(gt_effects_test_values, estimated_effects_test_values)
        
    print("Training Additive Model")
    additive_combined_train_df, additive_combined_test_df, module_csvs_train, module_csvs_test = get_additive_model_effects(
        csv_path, obs_data_path, train_qids, test_qids, hidden_dim=args.hidden_dim, epochs=args.epochs, 
        batch_size=args.batch_size, output_dim=args.output_dim, underlying_model_class=args.underlying_model_class, scale=args.scale, scaler_path=scaler_path
    )
    results["Additive_train"] = calculate_metrics(additive_combined_train_df["ground_truth_effect"], additive_combined_train_df["estimated_effect"])
    results["Additive_test"] = calculate_metrics(additive_combined_test_df["ground_truth_effect"], additive_combined_test_df["estimated_effect"])

    module_pehe_sum_train, module_cov_train = decompose_module_errors(module_csvs_train, args.num_modules)
    module_pehe_sum_test, module_cov_test = decompose_module_errors(module_csvs_test, args.num_modules)
    results["Module_PEHE_decomposition_test"] = module_pehe_sum_test, module_cov_test
    results["Module_PEHE_decomposition_train"] = module_pehe_sum_train, module_cov_train

    # Save results and CSVs
    results_path = f"{main_dir}/results/results_{args.data_dist}_{args.module_function_type}_{args.composition_type}_covariates_shared_{args.covariates_shared}_underlying_model_{args.underlying_model_class}_use_subset_features_{args.use_subset_features}_systematic_{args.systematic}"
    os.makedirs(results_path, exist_ok=True)
    with open(f"{results_path}/results_{args.num_modules}_{args.num_feature_dimensions}_scale_{args.scale}.json", "w") as f:
        json.dump(results, f)

    

    results_csv_folder = f"{results_path}/csvs_{args.num_modules}_{args.num_feature_dimensions}_scale_{args.scale}"
    os.makedirs(results_csv_folder, exist_ok=True)
    # Save combined_df_test and module_csvs here
    additive_combined_test_df.to_csv(f"{results_csv_folder}/additive_combined_test_df.csv", index=False)
    for module_file, module_csv in module_csvs_test.items():
        module_csv.to_csv(f"{results_csv_folder}/{module_file}.csv", index=False)

    

    print(results)
    print(f"Results saved at {results_path}")
    print(f"CSVs saved at {results_csv_folder}")
    print("Done!")

if __name__ == "__main__":
    main()