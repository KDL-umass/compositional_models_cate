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

### Manufacturing assembly data generation
1. cd ```manufacturing_assembly``` 
2. Run ```factoryScenarioGenerator.ipynb``` notebook to generate various manufacturing assembly line layouts with hierarchical structures and initial factory conditions which determine how much raw material is available and specifies the product demand. 
- Running this notebook will generate ```factory_scenario``` folder and ```initial_conditions``` folder. 
3. The number of workers and their skill distribution is specified in ```workers/workers_00.json``` and ```workers/workers_01.json``` folder. This specifies the binary treatment for hierarchical assembly layouts (units). 
3. Run ```simulate_factories-time-dynamics.ipynb``` notebook which uses simpy (an event simulator) to generate potential outcomes for different treatments. It takes a factory scenario (hierarchical structure representing instance-specific composition) as an input, runs it with a set of factory workers with multiple skill levels to calculate total rework, scrap and products produced by each station and the whole factory (component-level and unit-level potential ouctomes.)

### Matrix operations processing
- Run ```matrix_operations/generate_matrix_expressions_data.py``` to generate data for a set of matrix expressions (units) on a given computer hardware (treatment), and obtain the run-times (potential outcomes) for each operation (component) as well as the overall expression. 
- Already generated expression data on two different computer hardware is provided in the JSONs and CSVs formats here:  [Google Drive Link](https://drive.google.com/drive/u/0/folders/1Z6tOgRAP3LNkC1hLJI3nbgGWn98YFSt-).  

- **Note:** To generate the matrix operations data set, each matrix operation is evaluated independently of other operations, and overall run-time of the expression is the sum of the run-times of the individual operations, thus this data set satisfy additive parallel compositional assumption. Hence, we use additive parallel compositional model for this data set as explained below. 

## Experiment results 
In order to reproduce experiment results, currently we have separate codebase for each domain. Run the code in the respective folder to reproduce experiment results. 
### Synthetic data
- cd ```synthetic_data```.

- Run ```./base_experiments.sh``` in ```synthetic_data/``` folder to generate results for compositional generalization experiment for sequential and parallel compositional structures. This will generate ```results/``` folder in ```synthetic_data/``` with json files consisting of $R^2$ and ```PEHE``` metrics for CATE estimation task.

- Use ```notebooks/plot_results.ipynb``` to reproduce the results of Figure 3. 

### Manufacturing domain

#### Unitary model training and evaluation
- Run ```manufacturing_assembly/highLevelModelTraining.ipynb``` for CATE estimation using unitary models. 

#### Compositional model training and evaluation
- First, run ```manufacturing_assembly/LowLevelModels.ipynb``` to train component-level models for potential outcomes estimation. 

- Then, run ```manufacturing_assembly/LowLevelModels-aggregation.ipynb``` to aggregate the component-level estimates to obtain unit-level CATE estimate using the compositional approach. 

### Matrix operations processing
- First, make sure that ```matrix_operations/data/csvs``` contain the CSV files for the components and units (```maths_evaluation_datahigh_levelfeatures.csv```), consisting of covariates, treatment and outcomes (run-time) for both the treatments. Download the prepared data from [Google Drive Link](https://drive.google.com/drive/u/0/folders/1tJrB_i5Us8YA0JxHAIskV5f8H-yhnX-0)

- Run ```matrix_operations/run_math_evaluation_baselines.py``` to run the standard CATE baselines (unitary approach) on experimental (bias_strength = 0) and observational (bias_strength = 1 - 20) data. 

- Run ```matrix_operations/run_parallel_additive_model_maths_baseline.py``` to run the additive parallel compositional model. 

### Query execution domain 




