import argparse
import random
from run_all_models import run_experiments

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train modular neural network architectures and baselines for causal effect estimation")
    parser.add_argument("--domain", type=str, required=True, help="Domain to run the experiments on")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train the model")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials per bias strength and split")
    parser.add_argument("--nn_architecture_type", type=str, default="modular", help="Neural network architecture type")
    parser.add_argument("--nn_architecture_base", type=str, default="LSTM", help="Base neural network architecture")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--sampling", type=str, default="random", help="Sampling type")
    parser.add_argument("--splits", type=str, nargs='+', default=["iid", "ood"], help="Splits for training")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--outcomes_parallel", type=int, default=0, help="Outcomes parallel")
    parser.add_argument("--fixed_structure", type=int, default=0, help="Fixed structure flag")
    parser.add_argument("--single_outcome", type=int, default=1, help="Single outcome flag")
    parser.add_argument("--experiment_type", type=str, default="cate", help="Type of experiment to run")
    parser.add_argument("--bias_strengths", type=float, nargs='+', default=[0, 20], help="Bias strengths to test")
    parser.add_argument("--sample_sizes", type=int, nargs='+', default=[100, 500, 1000, 5000, 10000], help="Sample sizes to test")
    parser.add_argument("--baselines", type=str, nargs='+', default=["random_forest"], 
                        choices=["random_forest", "neural_network", "non_param_DML", "x_learner", "tnet", "snet", "snet1", "snet2", "drnet"],
                        help="Baseline models to run")
    return parser.parse_args()

def main():
    args = parse_arguments()
    random.seed(args.seed)
    run_experiments(args)

if __name__ == "__main__":
    main()