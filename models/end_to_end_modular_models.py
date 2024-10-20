import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from models.modular_compositional_models import split_modular_data
import numpy as np

# this creates a separate model for each module 
class ModuleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ModuleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for child output or 0
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    # forward function takes input x and child_output
    def forward(self, x, child_output):
        # Ensure x is 2D: [batch_size, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Ensure child_output is 2D: [batch_size, 1]
        if child_output.dim() == 0:
            child_output = child_output.unsqueeze(0).unsqueeze(1)
        elif child_output.dim() == 1:
            child_output = child_output.unsqueeze(1)
        
        combined_input = torch.cat([x, child_output], dim=1)
        return self.net(combined_input)


# this creates a model with multiple modules
class ModularModel(nn.Module):
    def __init__(self, module_configs):
        super(ModularModel, self).__init__()
        self.modules = {}
        for module_id, config in module_configs.items():
            self.modules[module_id] = ModuleNet(
                config['input_dim'], 
                config['hidden_dim'], 
                config['output_dim']
            )

    # and forward function takes input x and json_structure
    def forward(self, x, json_structure):
        
        def forward_recursive(node):
            # Get module_id and input for this node
            module_id = str(node['module_id'])
            module_input = x[module_id]
            
            # Get output from child node
            child_output = torch.zeros(1, 1)
            if node['children'] is not None and len(node['children']) > 0:
                child_output = forward_recursive(node['children'][0])
            
            # Return output from this node
            return self.modules[module_id](module_input, child_output)

        return forward_recursive(json_structure)

def create_modular_model(train_data, jsons_0, jsons_1, scale=False, use_high_level_features=False, train_df=None, covariates=None):
    
    module_configs = {}
    scalers = {}
    output_scaler = StandardScaler()
    
    def process_node(node):
        module_id = str(node['module_id'])
        if module_id not in module_configs:
            if use_high_level_features:
                input_dim = len(covariates) + 1  # +1 for treatment
            else:
                module_file = f"module_{module_id}.csv"
                df = train_data[module_file]
                module_covariates = [x for x in df.columns if "feature" in x]
                input_dim = len(module_covariates) + 1  # +1 for treatment
            
            # Create and fit scaler
            scaler = StandardScaler()
            if scale:
                if use_high_level_features:
                    scaler.fit(train_df[covariates])
                else:
                    scaler.fit(df[module_covariates])
            scalers[module_id] = scaler
            
            hidden_dim = max(32, (input_dim + 1) * 2)
            output_dim = 1
            
            module_configs[module_id] = {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'output_dim': output_dim
            }
        
        if node['children'] is not None and len(node['children']) > 0:
            process_node(node['children'][0])
    
    for json_structure in jsons_0.values():
        process_node(json_structure['json_tree'])
    for json_structure in jsons_1.values():
        process_node(json_structure['json_tree'])
    
    # Fit output scaler on all output values
    if scale:
        all_outputs = []
        for module_file in train_data.values():
            all_outputs.extend(module_file['output'].values)
        output_scaler.fit(np.array(all_outputs).reshape(-1, 1))
    
    return ModularModel(module_configs), scalers, output_scaler


def prepare_data_for_training(train_data, jsons_0, jsons_1, train_qids, scalers, output_scaler, scale=False, use_high_level_features=False, train_df=None, covariates=None):
    X = {qid: {} for qid in train_qids}
    y = []
    json_structures = []
    
    def get_features_recursive(node, query_id):
        module_id = str(node['module_id'])
        if use_high_level_features:
            row = train_df[train_df['query_id'] == query_id].iloc[0]
            features = row[covariates].values.reshape(1, -1)
        else:
            module_file = f"module_{module_id}.csv"
            df = train_data[module_file]
            row = df[df['query_id'] == query_id].iloc[0]
            module_covariates = [x for x in df.columns if "feature" in x]
            features = row[module_covariates].values.reshape(1, -1)
        
        if scale:
            features_scaled = scalers[module_id].transform(features).flatten()
        else:
            features_scaled = features.flatten()
        # Add treatment as a feature
        features_scaled = np.append(features_scaled, row['treatment_id'])
            
        X[query_id][module_id] = features_scaled
        if node['children'] is not None and len(node['children']) > 0:
            get_features_recursive(node['children'][0], query_id)
    
    for query_id in train_qids:
        treatment = train_data[f"module_{jsons_0[query_id]['json_tree']['module_id']}.csv"][train_data[f"module_{jsons_0[query_id]['json_tree']['module_id']}.csv"]['query_id'] == query_id]['treatment_id'].iloc[0]
        json_data = jsons_1[query_id] if treatment == 1 else jsons_0[query_id]
        get_features_recursive(json_data['json_tree'], query_id)
        json_structures.append(json_data['json_tree'])
        root_module_id = str(json_data['json_tree']['module_id'])
        root_module_file = f"module_{root_module_id}.csv"
        output = train_data[root_module_file][train_data[root_module_file]['query_id'] == query_id]['output'].iloc[0]
        if scale:
            output = output_scaler.transform([[output]])[0][0]
        y.append(output)

    return X, torch.tensor(y, dtype=torch.float32).unsqueeze(1), json_structures

class ModularDataset(Dataset):
    def __init__(self, X, y, json_structures):
        self.X = X
        self.y = y
        self.json_structures = json_structures
        self.qids = list(X.keys())

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        qid = self.qids[idx]
        return {
            'X': {k: torch.tensor(v, dtype=torch.float32) for k, v in self.X[qid].items()},
            'y': self.y[idx],
            'json_structure': self.json_structures[idx]
        }

def custom_collate(batch):
    X_batch = {qid: item['X'] for qid, item in enumerate(batch)}
    y_batch = torch.stack([item['y'] for item in batch])
    json_structures = [item['json_structure'] for item in batch]
    return X_batch, y_batch, json_structures


def train_end_to_end_modular_model(model, X, y, json_structures, epochs, batch_size, lr=0.001):
    
    parameters = []
    for module in model.modules.values():
        parameters.extend(module.parameters())
    
    optimizer = optim.Adam(parameters, lr=lr)
    loss_fn = nn.MSELoss()

    dataset = ModularDataset(X, y, json_structures)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y, batch_structures in dataloader:
            optimizer.zero_grad()
            outputs = torch.cat([model(x, structure).unsqueeze(0) for x, structure in zip(batch_X.values(), batch_structures)])
            # squeeze outputs to match batch_y shape
            outputs = outputs.squeeze(1)
            # print(outputs.shape, batch_y.shape)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    return model

def predict_end_to_end_modular_model(model, test_data, jsons_0, jsons_1, scalers, output_scaler, scale=False, use_high_level_features=False, test_df=None, covariates=None):
    model.eval()
    predictions = {}
    
    def get_features_recursive(node, query_id, treatment):
        module_id = str(node['module_id'])
        if use_high_level_features:
            row = test_df[test_df['query_id'] == query_id].iloc[0]
            features = row[covariates].values.reshape(1, -1)
        else:
            module_file = f"module_{module_id}.csv"
            df = test_data[module_file]
            row = df[df['query_id'] == query_id].iloc[0]
            module_covariates = [x for x in df.columns if "feature" in x]
            features = row[module_covariates].values.reshape(1, -1)
        
        if scale:
            features_scaled = scalers[module_id].transform(features).flatten()
        else:
            features_scaled = features.flatten()
        # Add treatment as a feature
        features_scaled = np.append(features_scaled, treatment)
        return features_scaled

    query_ids = []
    if use_high_level_features:
        query_ids = test_df['query_id'].unique()
    else:
        for module_id, data in test_data.items():
            query_ids.extend(data['query_id'].values)
        query_ids = list(set(query_ids))
    
    for query_id in query_ids:
        X_0 = {str(node['module_id']): torch.tensor([get_features_recursive(node, query_id, 0)], dtype=torch.float32) 
                for node in traverse_json_tree(jsons_0[query_id]['json_tree'])}
        X_1 = {str(node['module_id']): torch.tensor([get_features_recursive(node, query_id, 1)], dtype=torch.float32)
                for node in traverse_json_tree(jsons_1[query_id]['json_tree'])}
        y_0 = model(X_0, jsons_0[query_id]['json_tree'])
        y_1 = model(X_1, jsons_1[query_id]['json_tree'])
        
        if scale:
            y_0 = output_scaler.inverse_transform(y_0.detach().numpy())[0][0]
            y_1 = output_scaler.inverse_transform(y_1.detach().numpy())[0][0]
        else:
            y_0 = y_0.item()
            y_1 = y_1.item()
        
        predictions[query_id] = y_1 - y_0
    return predictions


def traverse_json_tree(node):
    yield node
    if node['children'] is not None:
        for child in node['children']:
            yield from traverse_json_tree(child)

def get_ground_truth_effects_jsons(jsons_0, jsons_1, qids):
    jsons_0 = {k:v for k,v in jsons_0.items() if k in qids}
    jsons_1 = {k:v for k,v in jsons_1.items() if k in qids}
    ground_truth_0 = {k:v["query_output"] for k,v in jsons_0.items()}
    ground_truth_1 = {k:v["query_output"] for k,v in jsons_1.items()}
    return {k: ground_truth_1[k] - ground_truth_0[k] for k in qids}
    

def prepare_results(qids, predictions, ground_truth_data):
    results = []
    
    for qid in qids:
        results.append({
            "query_id": qid,
            "ground_truth_effect": ground_truth_data[qid],
            "estimated_effect": predictions[qid]
        })
    return pd.DataFrame(results)

def get_end_to_end_modular_model_effects(csv_path, obs_data_path, train_qids, test_qids, jsons_0, jsons_1, hidden_dim=32, epochs=100, batch_size=64, output_dim=1, underlying_model_class="MLP", scale=True, scaler_path=None, bias_strength=0, domain="synthetic_data", model_misspecification=False, composition_type="hierarchical", evaluate_train=False, train_df=None, test_df=None, covariates=None, use_high_level_features=False):
    
    if use_high_level_features:
        assert train_df is not None and test_df is not None and covariates is not None, "train_df, test_df, and covariates must be provided when use_high_level_features is True"
       
    train_data, test_data, module_files = split_modular_data(csv_path, obs_data_path, train_qids, test_qids, scaler_path, scale, bias_strength, composition_type)
    
    # Create the model and get scalers
    model, scalers, output_scaler = create_modular_model(train_data, jsons_0, jsons_1, scale, use_high_level_features, train_df, covariates)
    
    # Prepare data for training
    X, y, json_structures = prepare_data_for_training(train_data, jsons_0, jsons_1, train_qids, scalers, output_scaler, scale, use_high_level_features, train_df, covariates)
    
    # Train the model
    train_end_to_end_modular_model(model, X, y, json_structures, epochs, batch_size)
    
    # Make predictions
    if evaluate_train:
        train_predictions = predict_end_to_end_modular_model(model, train_data, jsons_0, jsons_1, scalers, output_scaler, scale, use_high_level_features, train_df, covariates)
        train_ground_truth = get_ground_truth_effects_jsons(jsons_0, jsons_1, train_qids)
        train_results = prepare_results(train_qids, train_predictions, train_ground_truth)
    
    test_predictions = predict_end_to_end_modular_model(model, test_data, jsons_0, jsons_1, scalers, output_scaler, scale, use_high_level_features, test_df, covariates)
    
    # Get ground truth effects
    test_ground_truth = get_ground_truth_effects_jsons(jsons_0, jsons_1, test_qids)
    
    # Prepare and return results
    test_results = prepare_results(test_qids, test_predictions, test_ground_truth)
    
    if evaluate_train:
        return train_results, test_results
    else:
        return test_results