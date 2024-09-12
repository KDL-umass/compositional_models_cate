import argparse
import random
from models.run_compositional_models import run_comp_experiment, setup_directories

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train modular neural network architectures and baselines for causal effect estimation")
    parser.add_argument("--domain", type=str, required=True, help="Domain to run the experiments on")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train the model")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials per bias strength and split")
    parser.add_argument("--nn_architecture_type", type=str, default="modular", help="Neural network architecture type")
    parser.add_argument("--nn_architecture_base", type=str, default="MLP", help="Base neural network architecture")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    #add hidden_size argument
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for training")
    parser.add_argument("--sampling", type=str, default="random", help="Sampling type")
    parser.add_argument("--splits", type=str, nargs='+', default=["iid"], help="Splits for training")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--outcomes_parallel", type=int, default=0, help="Outcomes parallel")
    parser.add_argument("--fixed_structure", type=int, default=0, help="Fixed structure flag")
    parser.add_argument("--single_outcome", type=int, default=0, help="Single outcome flag")
    parser.add_argument("--bias_strength", type=float, default=10, help="Bias strength to test")
    parser.add_argument("--biasing_covariate", type=str, default="x0", help="Biasing covariate")
    #add test_run flag
    parser.add_argument("--test_run", type=int, default=1, help="Test run flag")
    return parser.parse_args()

def main():
    args = parse_arguments()
    random.seed(args.seed)
    dirs = setup_directories(args.domain, args.fixed_structure, args.outcomes_parallel)
    run_comp_experiment(args, dirs)

if __name__ == "__main__":
    main()