import os
from models.utils import *
from models.train_nn_architectures import train_model, evaluate_model
import pandas as pd
import pickle


def setup_directories(domain, fixed_structure, outcomes_parallel, run_env="local"):
    if run_env == "local":
        main_dir = "/Users/ppruthi/research/compositional_models/compositional_models_cate/domains"
    else:
        main_dir = "/work/pi_jensen_umass_edu/ppruthi_umass_edu/compositional_models_cate/domains"

    
    subdirs = ["csvs", "jsons", "plots", "results", "config"]
    
    if domain == "synthetic_data":
        dirs = {subdir: f"{main_dir}/{domain}/{subdir}/fixed_structure_{fixed_structure}_outcomes_parallel_{outcomes_parallel}" for subdir in subdirs}
    else:
        dirs = {subdir: f"{main_dir}/{domain}/{subdir}" for subdir in subdirs}
    
    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    return dirs

def run_comp_experiment(args, dirs):
    # Load data
    if args.test_run == 1:
        load_all = False
        print("Running test run")
    # Load sampled data
    query_id_to_treatment = load_query_id_to_treatment(dirs, args.biasing_covariate, args.bias_strength)
    query_id_depth_info = load_query_id_depth_info(dirs, args.domain)

    # read module_features from config file
    with open(f"{dirs['config']}/{args.domain}_config.json", "r") as f:
        config = json.load(f)
        module_feature_count_dict = config["module_feature_counts"]
    
    results = []
    for split in args.splits:
        for trial in range(args.num_trials):
            train_test_split_info = load_train_test_split_info(dirs, split)
            input_scalers, output_scalers = get_input_output_scalers(dirs, args.biasing_covariate, args.bias_strength, split)
            
            tree_groups_0 = load_dataset(domain=args.domain, data_folder=f"{dirs['jsons']}/data_0", outcomes_parallel=args.outcomes_parallel, single_outcome=args.single_outcome, load_all=load_all, input_scalers=input_scalers, output_scalers=output_scalers)
            tree_groups_1 = load_dataset(domain=args.domain, data_folder=f"{dirs['jsons']}/data_1", outcomes_parallel=args.outcomes_parallel, single_outcome=args.single_outcome, load_all=load_all, input_scalers=input_scalers, output_scalers=output_scalers)

            train_dataset_tree_groups_0, train_dataset_tree_groups_1, test_dataset = train_test_split_tree_groups(tree_groups_0, tree_groups_1, train_test_split_info)
            # Sample and create batch
            sampled_tree_groups = get_observational_tree_groups(train_dataset_tree_groups_0, train_dataset_tree_groups_1, query_id_to_treatment)
            print(f"Sampled dataset size: {len(sampled_tree_groups)}")

            # # get input and output scalers per module
            # input_scalers, output_scalers = get_input_output_scalers(sampled_tree_groups)
            train_dataset = create_batch(sampled_tree_groups,query_id_depth_info, batch_size=args.batch_size)
            print(f"Running split {split} trial {trial}")
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Test dataset size: {len(test_dataset)}")

            # Now how to do scaling?
            # As we have information about the train/test split, we can just scale the train data csvs and store the scalers
            # then we can use the scalers to scale both train and test data here before passing it to the model
            # Train and evaluate model
            nn_model_architecture = f"{args.nn_architecture_type}{args.nn_architecture_base}"
            model = train_model(args.domain, train_dataset, test_dataset, epochs=args.epochs, nn_model_architecture=nn_model_architecture, hidden_size=args.hidden_size, outcomes_parallel=args.outcomes_parallel, single_outcome=args.single_outcome, module_feature_count_dict=module_feature_count_dict)
            test_predictions, test_outputs = evaluate_model(model, test_dataset, single_outcome=args.single_outcome, outcomes_parallel=args.outcomes_parallel)

            # Create test dataframe and plot results
            test_df = create_test_df(test_predictions, test_outputs, single_outcome=args.single_outcome, outcomes_parallel=args.outcomes_parallel)
            pehe, r2 = plot_and_save_results(test_df, dirs['plots'], dirs['results'], split, args.sampling, args.biasing_covariate, args.bias_strength, args.nn_architecture_type, args.single_outcome, experiment_type="cate")
            
            results.append({
                "split": split,
                "bias_strength": args.bias_strength,
                "trial": trial,
                "pehe": pehe,
                "r2": r2,
                "nn_architecture": nn_model_architecture,
            })
    
    return results