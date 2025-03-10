# Compositional Models for Estimating Causal Effects
Repository providing benchmarks and code to reproduce experiments of CLeaR 25 paper: [Compositional Models for Estimating Causal Effects](https://arxiv.org/abs/2406.17714), to appear in Causal Learning and Reasoning Conference, 2025. 

**Summary:** We introduce a novel compositional framework to estimate conditional average treatment effects (CATE) for compositional systems with structured units. We introduce three novel and realistic evaluation environments to evaluate compositional approaches for causal effect estimation — (1) query execution in relational databases, (2) matrix processing on different types of computer hardware, and (3) simulated manufacturing assembly line data based on a realistic simulator. We provide data and code to generate data from the three benchmarks and synthetic data used in the paper. We find that the compositional approach provides accurate causal effect estimation for structured units, increased sample efficiency, improved overlap between treatment and control groups, and compositional generalization to units with unseen combinations of components.

## Data generation and benchmark creation
### Synthetic data
We generate synthetic compositional data with various characteristics -- composition structures (sequential and parallel), data distribution (uniform and normal), functional forms of response functions (linear, non-linear, polynomial), systematic data generation of increasing tree-depths vs. sequential tree generation with exactly same composition structure across units. For more details, see ```synthetic_data/data_generator/synthetic_data_sampler.py``` file. 

**Usage:**
To generate synthetic data, use the below code (with root_dir ```synthetic_data/```).
```python
from data_generator.synthetic_data_sampler import SyntheticDataSampler
num_modules = 10
module_function_types = ["polyval"] * num_modules

# simulate data for both treatments (experimental data )
sampler = SyntheticDataSampler( num_modules = num_modules, 
                                num_feature_dimensions = 1, 
                                composition_type = "sequential", 
                                fixed_structure = False, 
                                max_depth=num_modules, 
                                num_samples=1000, 
                                seed=42, 
                                data_dist = "uniform", module_function_types=module_function_types, resample=False)

# create observational data by introducing observational bias
sampler.create_observational_data(biasing_covariate="feature_sum",      bias_strength=1)

# split units into train/test systematically (IID: Random split, OOD: split on varying tree-depths) and indicate if models are evaluated on the maximum tree-depth (for OOD split)
sampler.create_iid_ood_split(split_type="ood", 
                            num_train_modules=train_modules, test_on_last_depth=True)
```

## Experiment results 
In order to reproduce experiment results, currently we have separate codebase for each domain. Run the code in the respective folder to reproduce experiment results. 
### Synthetic data
- cd ```synthetic_data```

- Run ```./base_experiments.sh``` in ```synthetic_data/``` folder to generate results for compositional generalization experiment for sequential and parallel compositional structures. This will generate ```results/``` folder in ```synthetic_data/``` with json files consisting of $R^2$ and PEHE metrics for CATE estimation task.

- Use ```notebooks/plot_results.ipynb``` to reproduce the results of Figure 3. 

### Manufacturing domain

### Query execution domain 

### Matrix operations processing


