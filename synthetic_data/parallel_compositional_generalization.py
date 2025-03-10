import argparse
import json
import os
import warnings
import sys
sys.path.append('../')
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from models.modular_compositional_models import *
from data_generator.synthetic_data_sampler import SyntheticDataSampler
from models.end_to_end_modular_models import *
from models.MoE import *
from models.utils import *
from models.nn_architectures import *
from models.baselines import *
from exp_utils import *
warnings.filterwarnings('ignore')

args = parse_arguments()


all_results = {}
all_results_train_size = {}
noise = 1
num_train_modules = list(range(2, args.num_modules + 1))
exp = "CG"

for variable in num_train_modules:
    if exp == "noise":
        train_modules = 3
        noise = variable
    else:
        noise = 0
        train_modules = variable
    all_results[str(variable)] = {}
    all_results_train_size[str(variable)] = []

    main_dir, csv_path, obs_data_path, scaler_path = setup_directories(args)
    
    module_function_types = ["polyval"] * args.num_modules
    sampler = SyntheticDataSampler(args.num_modules, args.num_feature_dimensions, args.composition_type, 
                                    args.fixed_structure, args.num_modules, args.num_samples, args.seed, 
                                    args.data_dist, module_function_types=module_function_types, resample=args.resample,
                                    heterogeneity=args.heterogeneity, covariates_shared=args.covariates_shared, 
                                    use_subset_features=args.use_subset_features, systematic=args.systematic, 
                                    run_env=args.run_env, noise = noise)

    # first sample the data and prepare it
    data, df_sampled = simulate_and_prepare_data(args, sampler, csv_path, obs_data_path, scaler_path, num_train_modules=train_modules, test_on_last_depth=True)
    
    # Load train and test qids
    train_qids, test_qids = load_train_test_qids(csv_path, args)

    covariates = [x for x in df_sampled.columns if "feature_0" in x] if args.covariates_shared else [x for x in df_sampled.columns if "feature" in x]
    module_feature_covariates = [x for x in df_sampled.columns if "feature_0" in x] if args.covariates_shared else [x for x in df_sampled.columns if "feature" in x]
    
        
    if args.systematic:
        covariates += [x for x in df_sampled.columns if "num" in x]

    # Scale data
    if args.scale:
        data, df_sampled = scale_df(data, df_sampled, scaler_path, csv_path, composition_type=args.composition_type, covariates=covariates,data_dist=args.data_dist)

    # Load train and test data
    train_df, test_df, train_qids, test_qids = load_train_test_data(csv_path, args, df_sampled)
    train_df, test_df = process_shared_covariates_row_wise(train_df, test_df, args)
    
    # covariates = [x for x in train_df.columns if "feature" or "order" in x and "output" not in x]
    T0_trees = {}
    # index trees by query_id
    for tree in sampler.processed_trees_0:
        T0_trees[tree["query_id"]] = tree

    T1_trees = {}
    # index trees by query_id
    for tree in sampler.processed_trees_1:
        T1_trees[tree["query_id"]] = tree

    # As we have same structure across the query_id, we need only one tree to get the structure.

    print(f"Training data shape: {train_df.shape}")
    print("Covariates: ", covariates)
    treatment, outcome = "treatment_id", "query_output"

    gt_effects_test = get_ground_truth_effects(data, test_qids)
    gt_effects_train = get_ground_truth_effects(data, train_qids)


    model_effects_train = {}
    model_effects_test = {}

    input_dim = len(covariates)
    if input_dim < args.hidden_dim and args.covariates_shared:
        hidden_dim = args.hidden_dim 
    else:
        hidden_dim = (input_dim + 1)*2

    baseline_models = {
        "XLearner": None,
        "TLearner": None,
        "SLearner": None,
        "RandomForest": None,
        "NonParamDML": None
    }
    results = {}
    for model_name, model_class in baseline_models.items():
        if model_name == "XLearner":
            print(f"Training {model_name} Model")
            estimated_effects_train_values = x_learner(train_df, covariates, treatment, outcome, X_test=train_df)
            estimated_effects_test_values = x_learner(train_df, covariates, treatment, outcome, X_test=test_df)
        elif model_name == "TLearner":
            print(f"Training {model_name} Model")
            estimated_effects_train_values = t_learner(train_df, covariates, treatment, outcome, X_test=train_df)
            estimated_effects_test_values = t_learner(train_df, covariates, treatment, outcome, X_test=test_df)
        elif model_name == "SLearner":
            print(f"Training {model_name} Model")
            estimated_effects_train_values = s_learner(train_df, covariates, treatment, outcome, X_test=train_df)
            estimated_effects_test_values = s_learner(train_df, covariates, treatment, outcome, X_test=test_df)
        elif model_name == "RandomForest":
            print(f"Training {model_name} Model")
            estimated_effects_train_values, _, _ = random_forest(train_df, covariates, treatment, outcome, X_test=train_df)
            estimated_effects_test_values, _, _ = random_forest(train_df, covariates, treatment, outcome, X_test=test_df)
        elif model_name == "NonParamDML":
            print(f"Training {model_name} Model")
            estimated_effects_train_values = non_param_DML(train_df, covariates, treatment, outcome, X_test=train_df)
            estimated_effects_test_values = non_param_DML(train_df, covariates, treatment, outcome, X_test=test_df)
        gt_effects_test_values = np.array(list(gt_effects_test.values()))
        gt_effects_train_values = np.array(list(gt_effects_train.values()))
        results[f"{model_name}_train"] = calculate_metrics(gt_effects_train_values, estimated_effects_train_values)
        results[f"{model_name}_test"] = calculate_metrics(gt_effects_test_values, estimated_effects_test_values)

    nn_models = {
        # # high-level model
        "Baseline": BaselineModel if args.underlying_model_class == "MLP" else BaselineLinearModel,
        # # high-level model wirh same number of modules
        "MoE": MoE if args.underlying_model_class == "MLP" else MoELinearModel,
        "MoEknownCov": MoEknownCov,
    }
    
    for model_name, model_class in nn_models.items():
        print(f"Training {model_name} Model")
        if model_name == "Baseline":
            model = model_class(input_dim + 1, hidden_dim, args.output_dim)
        elif model_name == "MoE":
            model = model_class(input_dim + 1, hidden_dim, args.output_dim, args.num_modules)
        elif model_name == "MoEknownCov":
            model = model_class(input_dim + 1, hidden_dim, args.output_dim, args.num_modules, args.num_feature_dimensions)
        
        estimated_effects_train, estimated_effects_test = train_and_evaluate_model(
            model, train_df, test_df, covariates, treatment, outcome, args.epochs, args.batch_size, args.num_modules, args.num_feature_dimensions, train_qids, test_qids, plot=False, model_name=model_name
        )
        gt_effects_train_values, gt_effects_test_values = np.array(list(gt_effects_train.values())), np.array(list(gt_effects_test.values()))
        estimated_effects_train_values, estimated_effects_test_values = np.array(list(estimated_effects_train.values())), np.array(list(estimated_effects_test.values()))
        results[f"{model_name}_train"] = calculate_metrics(gt_effects_train_values, estimated_effects_train_values)
        results[f"{model_name}_test"] = calculate_metrics(gt_effects_test_values, estimated_effects_test_values)
        model_effects_train[model_name] = estimated_effects_train
        model_effects_test[model_name] = estimated_effects_test
        # save model as .pth file
        # torch.save(model.state_dict(), f"{main_dir}/results/systematic_{args.systematic}/{model_name}_model.pth")
        


    print("Training Catenets Model")
    estimated_effects_train, estimated_effects_test = train_and_evaluate_catenets(train_df, test_df, covariates, treatment, outcome, train_qids, test_qids)
    gt_effects_train_values, gt_effects_test_values = np.array(list(gt_effects_train.values())), np.array(list(gt_effects_test.values()))
    estimated_effects_train_values, estimated_effects_test_values = np.array(list(estimated_effects_train.values())), np.array(list(estimated_effects_test.values()))
    results["Catenets_train"] = calculate_metrics(gt_effects_train_values, estimated_effects_train_values)
    results["Catenets_test"] = calculate_metrics(gt_effects_test_values, estimated_effects_test_values)
    # Add Catenets effects
    model_effects_train["Catenets"] = estimated_effects_train
    model_effects_test["Catenets"] = estimated_effects_test
    print(results)
    print("Training Additive composition Model with known covariates")
    # Save results and CSVs
    results_path = f"{main_dir}/results/systematic_{args.systematic}/results_{args.data_dist}_{args.composition_type}_covariates_shared_{args.covariates_shared}_use_subset_features_{args.use_subset_features}_numbert_of_modules_{args.num_modules}_combined_results_sequential.json"
    

    evaluate_train = True
    print("Training Additive Model")
    additive_combined_train_df, additive_combined_test_df, module_csvs_train, module_csvs_test, module_test_sizes = get_additive_model_effects(
        csv_path, obs_data_path, train_qids, test_qids, hidden_dim=args.hidden_dim, epochs=args.epochs, 
        batch_size=args.batch_size, output_dim=args.output_dim, underlying_model_class=args.underlying_model_class, scale=args.scale, scaler_path=scaler_path, bias_strength=args.bias_strength, num_modules=args.num_modules, num_feature_dimensions=args.num_feature_dimensions
    )
    print(results)
    results["Additive_known_cov_train"] = calculate_metrics(additive_combined_train_df["ground_truth_effect"], additive_combined_train_df["estimated_effect"])
    results["Additive_known_cov_test"] = calculate_metrics(additive_combined_test_df["ground_truth_effect"], additive_combined_test_df["estimated_effect"])

    
    # now have additive model version without known covariates

    module_pehe_sum_train, module_cov_train = decompose_module_errors(module_csvs_train, args.num_modules)
    module_pehe_sum_test, module_cov_test = decompose_module_errors(module_csvs_test, args.num_modules)
    results["Module_known_cov_PEHE_decomposition_test"] = module_pehe_sum_test, module_cov_test
    results["Module_PEHE_known_cov_decomposition_train"] = module_pehe_sum_train, module_cov_train


    additive_combined_train_df, additive_combined_test_df, module_csvs_train, module_csvs_test, module_test_sizes = get_additive_model_effects(
        csv_path, obs_data_path, train_qids, test_qids, hidden_dim=args.hidden_dim, epochs=args.epochs, 
        batch_size=args.batch_size, output_dim=args.output_dim, underlying_model_class=args.underlying_model_class, scale=args.scale, scaler_path=scaler_path, bias_strength=args.bias_strength, hl_train_df=train_df, hl_test_df=test_df, hl_covariates=module_feature_covariates, use_high_level_features=True, num_modules=args.num_modules, num_feature_dimensions=args.num_feature_dimensions
        )
    results["Additive_unknown_cov_train"] = calculate_metrics(additive_combined_train_df["ground_truth_effect"], additive_combined_train_df["estimated_effect"])
    results["Additive_unknown_cov_test"] = calculate_metrics(additive_combined_test_df["ground_truth_effect"], additive_combined_test_df["estimated_effect"])

    module_pehe_sum_train, module_cov_train = decompose_module_errors(module_csvs_train, args.num_modules)
    module_pehe_sum_test, module_cov_test = decompose_module_errors(module_csvs_test, args.num_modules)
    results["Module_unknown_cov_PEHE_decomposition_test"] = module_pehe_sum_test, module_cov_test
    results["Module_PEHE_unknown_cov_decomposition_train"] = module_pehe_sum_train, module_cov_train
    
   
    all_results[str(variable)] = results
    
    # # save all_results
    with open(f"{main_dir}/results/systematic_{args.systematic}/all_results_{args.data_dist}_{args.composition_type}_covariates_shared_{args.covariates_shared}_use_subset_features_{args.use_subset_features}_exp_{exp}_number_of_modules_{args.num_modules}.json", "w") as f:
        json.dump(all_results, f)


    