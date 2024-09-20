from models.MoE import *
from models.utils import *
import pandas as pd
import numpy as np
import os

# Additive Baseline
# This model trains a separate model for each module and sums the estimated effects.
def get_additive_model_effects(csv_path, obs_data_path, train_qids, test_qids, hidden_dim=32, epochs=100, batch_size=64, output_dim=1, underlying_model_class="MLP"):
    module_files = os.listdir(f"{csv_path}/")
    print(module_files)
    module_files = [x for x in module_files if "module" in x]

    # read all the module files
    module_data = {}
    for module_file in module_files:
        module_data[module_file] = pd.read_csv(f"{csv_path}/{module_file}")
    

    query_id_treatment_id_json_path = f"{obs_data_path}/feature_sum_0/treatment_assignments.json"
    with open(query_id_treatment_id_json_path, "r") as f:
        query_id_treatment_id = json.load(f)
    train_data = {}
    test_data = {}
    for module_file in module_files:
        module_df = module_data[module_file]
        module_df["assigned_treatment_id"] = module_df["query_id"].apply(lambda x: query_id_treatment_id[str(x)])
        module_df = module_df[module_df["treatment_id"] == module_df["assigned_treatment_id"]]
        # drop the assigned treatment id
        module_df.drop("assigned_treatment_id", axis=1, inplace=True)
        # sort the data by query_id
        module_df = module_df.sort_values("query_id")
        # split the data into train and test
        train_data[module_file] = module_df[module_df["query_id"].isin(train_qids)]
        test_data[module_file] = module_df[module_df["query_id"].isin(test_qids)]
        
    for module_file in module_files:
        train_df = train_data[module_file]
        test_df = test_data[module_file]
        covariates = [x for x in train_df.columns if "feature" in x]
        treatment = "treatment_id"
        outcome = "output"
        input_dim = len(covariates)
        if underlying_model_class == "MLP":
            expert_model = BaselineModel(input_dim + 1, (input_dim + 1)*2, output_dim)
        else:
            expert_model = BaselineLinearModel(input_dim + 1, output_dim)
        
        expert_model, train_losses, val_losses = train_model(expert_model, train_df, covariates, treatment, outcome, epochs, batch_size)
        causal_effect_estimates = predict_model(expert_model, test_df, covariates)
        test_df["estimated_effect"] = causal_effect_estimates
        test_data[module_file] = test_df

    # now for each module, get the ground truth and estimated effects
    additive_ground_truth_effects = {}
    additive_estimated_effects = {}
    modules_csvs = {}
    for module_file in module_files:
        module_name = module_file.split(".")[0]
        train_df = train_data[module_file]
        test_df = test_data[module_file]
        module_causal_effect_dict_test = get_ground_truth_effects(module_data[module_file], test_qids, treatment_col="treatment_id", outcome_col="output")
        module_estimated_effects = get_estimated_effects(test_df, test_qids)
        # have a combined df with ground truth and estimated effects based on the query_id
        module_gt_effect_df = pd.DataFrame.from_dict(module_causal_effect_dict_test, orient="index", columns=["ground_truth_effect"])
        # add the estimated effects based on query ids with same order
        module_estimated_effects_df = pd.DataFrame.from_dict(module_estimated_effects, orient="index", columns=["estimated_effect"])
        # merge 
        module_combined_df = pd.concat([module_gt_effect_df, module_estimated_effects_df], axis=1)
        modules_csvs[module_name] = module_combined_df

        
        if len(additive_estimated_effects) == 0:
            additive_ground_truth_effects = module_causal_effect_dict_test
            additive_estimated_effects = module_estimated_effects
        else:
            # add the effects
            # handle the case where the query ids are not the same, check if k is in the other dict oyherwise add 0
            for k, v in module_causal_effect_dict_test.items():
                if k not in additive_ground_truth_effects:
                    additive_ground_truth_effects[k] = v
                else:
                    additive_ground_truth_effects[k] += v
                if k not in additive_estimated_effects:
                    additive_estimated_effects[k] = v
                else:
                    additive_estimated_effects[k] += v

            
    additive_gt_effect_df = pd.DataFrame.from_dict(additive_ground_truth_effects, orient="index", columns=["ground_truth_effect"])
    additive_estimated_effect_df = pd.DataFrame.from_dict(additive_estimated_effects, orient="index", columns=["estimated_effect"])
    additive_combined_df = pd.concat([additive_gt_effect_df, additive_estimated_effect_df], axis=1)

    return additive_combined_df, modules_csvs