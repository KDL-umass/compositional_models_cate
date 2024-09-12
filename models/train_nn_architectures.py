import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.nn_architectures import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import random
import json
    
def train_model(domain, train_dataset, test_dataset, epochs=50, nn_model_architecture = "modularLSTM", hidden_size=64, outcomes_parallel=0, single_outcome=True, module_feature_count_dict=None, module_features_dict=None):
    
    # based on the domain, we will have different tree structures --- binary or unary 
    # TODO: For synthetic domain, experiment with binary trees as well
    # Set up model parameters based on domain
    tree_type = "unary" if domain == "synthetic" else "binary"
    # for unary trees we have 2 extra dimensions, for binary trees we have 3 extra dimensions
    # for unary, one dimension is the treatment dimension, the other is the child input dimension
    # for binary, one dimension is the treatment dimension, the other is the left child input dimension, the third is the right child input dimension
    # This is only for MLP, have to figure out stuff for LSTM. How does the information propogate in LSTM for parse-tree like data. Do we just pass the representation of the 
    # left, right child nodes to the parent node or do we pass the parent node representation to the child nodes as well?
    extra_dim = 2 if tree_type == "unary" else 3 

    # Model selection
    model_classes = {
        "unary": {
            "recursiveLSTM": RecursiveLSTMPredictor,
            "modularLSTM": ModularLSTMPredictor,
            "modularMLP": ModularMLPPredictor
        },
        "binary": {
            "recursiveLSTM": RecursiveTreeLSTMPredictor,
            "modularLSTM": ModularTreeLSTMPredictor,
            "modularMLP": ModularTreeMLPPredictor
        }
    }
    model = model_classes[tree_type][nn_model_architecture](module_feature_count_dict, extra_dim, hidden_size)
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Shuffle the training dataset
        random.shuffle(train_dataset)

        for i, batch in enumerate(train_dataset):
            query_ids, trees, treatment_ids, outputs = batch
            predicted_outputs = [model(tree, treatment) for tree, treatment in zip(trees, treatment_ids)]

            if single_outcome:
                if outcomes_parallel:
                    # predicted_outputs = torch.stack([output for outputs in predicted_outputs for output in outputs]).squeeze(1).squeeze(1)
                    # take sum of the predicted outputs and then stack them
                    predicted_outputs = torch.stack([torch.stack(outputs).sum(dim=0) for outputs in predicted_outputs]).squeeze(1).squeeze(1)
                     
                    target_outputs = torch.tensor([output for output in outputs], dtype=torch.float32)
                    # calculate the sum of the outputs
                else:
                    predicted_outputs = torch.stack([output[0] for output in predicted_outputs]).squeeze(1)
                    target_outputs = torch.tensor(outputs, dtype=torch.float32).unsqueeze(1)
                  

            else:
                predicted_outputs = torch.stack([output for outputs in predicted_outputs for output in outputs]).squeeze(1).squeeze(1)
                # Flatten the target outputs if it's a list of lists, and convert to tensor
                target_outputs = torch.tensor([output for outputs in outputs for output in outputs], dtype=torch.float32)
           
            loss = criterion(predicted_outputs, target_outputs)
            total_loss += loss.item()

            # # Scale the gradients
            # loss = loss / len(train_dataset)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch: {epoch}, Train Loss: {avg_loss}")
    return model

    
def evaluate_model(model, test_dataset, single_outcome=True, outcomes_parallel=False):
    model.eval()
    test_predictions = {}
    test_outputs = {}
    print("Evaluating on test dataset")
    with torch.no_grad():
        for (query_id, tree_0, outputs_0, tree_1, outputs_1) in test_dataset:
            pred_0 = model(tree_0, 0)
            pred_1 = model(tree_1, 1)
            if single_outcome:
                if outcomes_parallel:
                    pred_0 = torch.stack(pred_0).sum(dim=0).squeeze().numpy()
                    pred_1 = torch.stack(pred_1).sum(dim=0).squeeze().numpy()
                else:
                    pred_0 = pred_0[0].squeeze().numpy()
                    pred_1 = pred_1[0].squeeze().numpy()
            else:
                if outcomes_parallel:
                    pred_0 = torch.stack(pred_0).sum(dim=0).squeeze().numpy()
                    pred_1 = torch.stack(pred_1).sum(dim=0).squeeze().numpy()
                    outputs_0 = np.sum(outputs_0)
                    outputs_1 = np.sum(outputs_1) 
                else:
                    pred_0 = torch.stack(pred_0).squeeze().numpy()
                    pred_1 = torch.stack(pred_1).squeeze().numpy()
                   
            test_predictions[query_id] = {"y0": pred_0, "y1": pred_1}
            test_outputs[query_id] = {"y0": outputs_0, "y1": outputs_1}   
    return test_predictions, test_outputs

