import os
import pandas as pd
from run_compositional_models import run_comp_experiment, setup_directories
from baselines import run_baseline


def run_baselines(args, df, covariates, treatment, outcome, X_test=None):
    baseline_results = {}
    for baseline in args.baselines:
        print(f"Running baseline: {baseline}")
        if baseline == "neural_network":
            y1, y0 = run_baseline(baseline, df, covariates, treatment, outcome, X_test, 
                                  batch_size=args.batch_size, epochs=args.epochs)
        else:
            y1, y0 = run_baseline(baseline, df, covariates, treatment, outcome, X_test)
        
        baseline_results[baseline] = {"y1": y1, "y0": y0}
    return baseline_results

def run_experiments(args):
    dirs = setup_directories(args.domain, args.fixed_structure, args.outcomes_parallel)
    
    all_results = []
    
    if args.experiment_type == "cate":
        for bias_strength in args.bias_strengths:
            exp_results = run_comp_experiment(args, dirs, bias_strength=bias_strength)
            all_results.extend(exp_results)
            
            if args.baselines:
                df = pd.read_csv(f"{dirs['csvs']}/obs_data/df_sampled_{args.sampling}_{args.biasing_covariate}_{bias_strength}.csv")
                covariates = [col for col in df.columns if col not in [args.treatment, args.outcome]]
                baseline_results = run_baselines(args, df, covariates, args.treatment, args.outcome)
                
                for baseline, results in baseline_results.items():
                    all_results.extend([{
                        "model": baseline,
                        "split": "all",
                        "bias_strength": bias_strength,
                        "trial": 0,
                        "y1": results["y1"],
                        "y0": results["y0"]
                    }])
    
    elif args.experiment_type == "sample_size":
        for sample_size in args.sample_sizes:
            exp_results = run_comp_experiment(args, dirs, sample_size=sample_size)
            all_results.extend(exp_results)
            
            if args.baselines:
                df = pd.read_csv(f"{dirs['csvs']}/obs_data/df_sampled_{args.sampling}_{args.biasing_covariate}_{sample_size}.csv")
                covariates = [col for col in df.columns if col not in [args.treatment, args.outcome]]
                baseline_results = run_baselines(args, df, covariates, args.treatment, args.outcome)
                
                for baseline, results in baseline_results.items():
                    all_results.extend([{
                        "model": baseline,
                        "split": "all",
                        "sample_size": sample_size,
                        "trial": 0,
                        "y1": results["y1"],
                        "y0": results["y0"]
                    }])
    
    else:
        all_results.extend(run_experiment(args, dirs))
    
    # Save results
    results_folder = f"{dirs['results']}/{'pehe' if args.experiment_type != 'sample_size' else 'r2'}"
    os.makedirs(results_folder, exist_ok=True)
    
    pd.DataFrame(all_results).to_csv(
        f"{results_folder}/{args.experiment_type}_results_{args.sampling}_{args.biasing_covariate}_{args.nn_architecture_type}_single_outcome_{args.single_outcome}_trials_{args.num_trials}.csv",
        index=False
    )