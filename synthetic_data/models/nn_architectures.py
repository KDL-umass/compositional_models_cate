import numpy as np
import torch
import torch.nn as nn
import json
import os
from sklearn.model_selection import train_test_split
import random
import scipy.linalg as linalg

# Base architecture for the LSTM model.
class LSTM(nn.Module):
    # initialize the TreeLSTM model
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.gate_size = 5 * hidden_size
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc_gate = nn.Linear(input_size + hidden_size, self.gate_size)

    # forward pass of the TreeLSTM model
    def forward(self, input, left_hidden, left_cell):
        input_combined = torch.cat((input, left_hidden), dim=1)
        gate_output = self.fc_gate(input_combined)
        gate_output = gate_output.unsqueeze(0)
        i_gate, f_gate, o_gate, cell_hat, hidden_hat = torch.split(gate_output, self.hidden_size, dim=2)
        i_gate = self.sigmoid(i_gate)
        f_gate = self.sigmoid(f_gate)
        o_gate = self.sigmoid(o_gate)
        cell_hat = self.tanh(cell_hat)
        cell = i_gate * cell_hat + f_gate * left_cell
        hidden = o_gate * self.tanh(cell)
        return hidden.squeeze(0), cell.squeeze(0)

# LSTM model for predicting the treatment effect.
class RecursiveLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, operators, features_dim):
        super(RecursiveLSTMPredictor, self).__init__()
        self.lstm = LSTM(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.operators = operators
        self.features_dim = features_dim

    # Shared function for recursively evaluating the expression tree
    def recursive_eval(self, node, treatment_id, left_hidden, left_cell, outputs):
        operator, left_child = (node["module_name"], None) if node["children"] is None else (node["module_name"], node["children"][0])
        if left_child is not None:
            left_hidden, left_cell, outputs = self.recursive_eval(left_child, treatment_id, left_hidden, left_cell, outputs)
        operator_tensor = torch.tensor([self.operators.index(operator)], dtype=torch.float32).unsqueeze(0)
        # convert treatment_id to tensor
        treatment_tensor = torch.tensor([treatment_id], dtype=torch.float32).unsqueeze(0)
        features = torch.tensor(node["features"][:self.features_dim], dtype=torch.float32).unsqueeze(0)
        input_tensor = torch.cat([features, operator_tensor], dim=1)
        input_tensor = torch.cat([input_tensor, treatment_tensor], dim=1)
        hidden, cell = self.tree_lstm(input_tensor, left_hidden, left_cell)
        output = self.output_layer(hidden)
        outputs.insert(0, output)
        return hidden, cell, outputs

    # forward pass of the LSTM model
    def forward(self, expr_tree, treatment_id):
        left_hidden = torch.zeros(1, self.tree_lstm.hidden_size)
        left_cell = torch.zeros(1, self.tree_lstm.hidden_size)
        _, _, outputs = self.recursive_eval(expr_tree, treatment_id, left_hidden, left_cell, [])
        return outputs

# Modular TreeLSTM model for predicting the treatment effect.
class ModularLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, operators, features_dim):
        super(ModularLSTMPredictor, self).__init__()
        self.operators = operators
        self.features_dim = features_dim
        self.module_lstms = nn.ModuleList([LSTM(input_size, hidden_size) for _ in operators])
        self.output_layers = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in operators])

    # Shared function for recursively evaluating the expression tree
    def recursive_eval(self, node, treatment_id, left_hidden, left_cell, outputs):
        # get the operator and left child of the current node
        operator, left_child = (node["module_name"], None) if node["children"] is None else (node["module_name"], node["children"][0])
        # recursively evaluate the left child
        if left_child is not None:
            left_hidden, left_cell, outputs = self.recursive_eval(left_child, treatment_id, left_hidden, left_cell, outputs)
        # get the index of the operator
        operator_index = self.operators.index(operator)
        # get the LSTM and output layer for the operator
        lstm = self.module_lstms[operator_index]
        output_layer = self.output_layers[operator_index]
        # convert treatment_id to tensor
        treatment_tensor = torch.tensor([treatment_id], dtype=torch.float32).unsqueeze(0)
        # operator_tensor = torch.tensor([operator_index], dtype=torch.float32).unsqueeze(0)
        # get the features of the current node
        features = torch.tensor(node["features"][:self.features_dim], dtype=torch.float32).unsqueeze(0)
        input_tensor = torch.cat([features, treatment_tensor], dim=1)

        # input_tensor = torch.cat([input_tensor, treatment_tensor], dim=1)
        hidden, cell = lstm(input_tensor, left_hidden, left_cell)
        # get the output of the current node
        output = output_layer(hidden)
        # insert the output into the outputs list
        outputs.insert(0, output)
        return hidden, cell, outputs

    # forward pass of the Modular TreeLSTM model
    def forward(self, expr_tree, treatment_id):
        left_hidden = torch.zeros(1, self.module_lstms[0].hidden_size)
        left_cell = torch.zeros(1, self.module_lstms[0].hidden_size)
        _, _, outputs = self.recursive_eval(expr_tree, treatment_id, left_hidden, left_cell, [])
        return outputs

# Base architecture for the MLP model.
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, input_tensor):
        x = torch.relu(self.fc1(input_tensor))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Modular MLP model for predicting the treatment effect.
class ModularMLPPredictor(nn.Module):
    def __init__(self, module_feature_dict, extra_dim, hidden_size):
        super(ModularMLPPredictor, self).__init__()
        self.module_feature_dict = module_feature_dict
        self.module_names = list(module_feature_dict.keys())
        self.module_mlps = nn.ModuleList([MLP(module_feature_dict[module_name] + extra_dim, hidden_size) for module_name in self.module_names])

    def recursive_eval(self, node, treatment_id, outputs):
        # get the operator and left child of the current node
        module_name, left_child = (node.module_name, None) if node.left_child is None else (node.module_name, node.left_child)
        # recursively evaluate the left child
        if left_child is not None:
            outputs = self.recursive_eval(left_child, treatment_id, outputs)
        # get the index of the operator and the MLP for the operator
        module_index = self.module_names.index(module_name)
        mlp = self.module_mlps[module_index]
        # convert treatment_id to tensor
        treatment_tensor = torch.tensor([treatment_id], dtype=torch.float32).unsqueeze(0)
        # get the output of the child node if it exists but don't pop it from the outputs list
        left_output = outputs[0] if left_child is not None else torch.tensor([0.0], dtype=torch.float32).unsqueeze(0)
        features = torch.tensor(node.features, dtype=torch.float32).unsqueeze(0)
        # append left_output to the features tensor if it exists
        input_tensor = torch.cat([features, left_output], dim=1) 
        input_tensor = torch.cat([input_tensor, treatment_tensor], dim=1)
        output = mlp(input_tensor)
        outputs.insert(0, output)
        return outputs

    def forward(self, expr_tree, treatment_id):
        outputs = self.recursive_eval(expr_tree, treatment_id, [])
        return outputs
        
# Base architecture for the TreeLSTM model.
class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.gate_size = 5 * hidden_size
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc_gate = nn.Linear(input_size + 2*hidden_size, self.gate_size)

    def forward(self, input, left_hidden, left_cell, right_hidden, right_cell):
        input_combined = torch.cat((input, left_hidden, right_hidden), dim=1)
        # print(input_combined.shape)
        gate_output = self.fc_gate(input_combined)
        gate_output = gate_output.unsqueeze(0)
        i_gate, f_left_gate, f_right_gate, o_gate, cell_hat = torch.split(gate_output, self.hidden_size, dim=2)
        i_gate = self.sigmoid(i_gate)
        f_left_gate = self.sigmoid(f_left_gate)
        f_right_gate = self.sigmoid(f_right_gate)
        o_gate = self.sigmoid(o_gate)
        cell_hat = self.tanh(cell_hat)
        cell = i_gate * cell_hat + f_left_gate * left_cell + f_right_gate * right_cell
        hidden = o_gate * self.tanh(cell)
        return hidden.squeeze(0), cell.squeeze(0)

# Recursive TreeLSTM model for predicting the treatment effect.
class RecursiveTreeLSTMPredictor(nn.Module):
    def __init__(self, module_feature_dict, extra_dim, hidden_size):
        super(RecursiveMatrixOperationPredictor, self).__init__()
        self.module_names = list(module_feature_dict.keys())
        # input_size is maximum of all the module_feature_dict values
        self.input_size = max(module_feature_dict.values()) + extra_dim
        self.tree_lstm = TreeLSTM(self.input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
       

    def recursive_eval(self, node, treatment_id, left_hidden, left_cell, right_hidden, right_cell, outputs,feature_scaler, output_scaler):
        module_name, left_child, right_child = node["module_name"], None, None
        if node["left_child"] is not None:
            left_child = node["left_child"] 
        if node["right_child"] is not None:
            right_child = node["right_child"] 

        if left_child is not None:
            left_hidden, left_cell, outputs = self.recursive_eval(
                left_child, treatment_id, left_hidden, left_cell, right_hidden, right_cell, outputs,feature_scaler, output_scaler
            )
        else:
            left_hidden, left_cell = torch.zeros(1, self.tree_lstm.hidden_size), torch.zeros(1, self.tree_lstm.hidden_size)
        if right_child is not None:
            right_hidden, right_cell, outputs = self.recursive_eval(
                right_child, treatment_id, left_hidden, left_cell, right_hidden, right_cell, outputs,feature_scaler, output_scaler
            )
        else:
            right_hidden, right_cell = torch.zeros(1, self.tree_lstm.hidden_size), torch.zeros(1, self.tree_lstm.hidden_size)
      

        module_name_tensor = torch.tensor([self.module_names.index(module_name)], dtype=torch.float32).unsqueeze(0)
        # convert treatment_id to tensor
        treatment_tensor = torch.tensor([treatment_id], dtype=torch.float32).unsqueeze(0)
        features = torch.tensor(node["features"][:self.features_dim], dtype=torch.float32).unsqueeze(0)
        input_tensor = torch.cat([features, module_name_tensor], dim=1)
        input_tensor = torch.cat([input_tensor, treatment_tensor], dim=1)
        hidden, cell = self.tree_lstm(input_tensor, left_hidden, left_cell, right_hidden, right_cell)
        output = self.output_layer(hidden)
        outputs.insert(0, output)
        return hidden, cell, outputs

    def forward(self, expr_tree, treatment_id, feature_scaler, output_scaler):
        left_hidden = torch.zeros(1, self.tree_lstm.hidden_size)
        left_cell = torch.zeros(1, self.tree_lstm.hidden_size)
        right_hidden = torch.zeros(1, self.tree_lstm.hidden_size)
        right_cell = torch.zeros(1, self.tree_lstm.hidden_size)
        _, _, outputs = self.recursive_eval(expr_tree, treatment_id, left_hidden, left_cell, right_hidden, right_cell, [], feature_scaler, output_scaler) 
        return outputs

# Modular TreeLSTM model for predicting the treatment effect.
class ModularTreeLSTMPredictor(nn.Module):
    def __init__(self, module_feature_dict, extra_dim, hidden_size):
        super(ModularLSTMPredictor, self).__init__()
        self.module_feature_dict = module_feature_dict
        self.module_names = list(module_feature_dict.keys())
        self.module_lstms = nn.ModuleList([TreeLSTM(self.module_feature_dict[module_name] + extra_dim, hidden_size) for module_name in self.module_names])
        self.output_layers = nn.ModuleList([nn.Linear(hidden_size, 1) for module_name in self.module_names])
       

    def recursive_eval(self, node, treatment_id, left_hidden, left_cell, right_hidden, right_cell, outputs,feature_scaler, output_scaler):
        module_name, left_child, right_child = node["module_name"], None, None
        if node["left_child"] is not None:
            left_child = node["left_child"] 
        if node["right_child"] is not None:
            right_child = node["right_child"] 

        if left_child is not None:
            left_hidden, left_cell, outputs = self.recursive_eval(
                left_child, treatment_id, left_hidden, left_cell, right_hidden, right_cell, outputs,feature_scaler, output_scaler
            )
        else:
            left_hidden, left_cell = torch.zeros(1, self.tree_lstm.hidden_size), torch.zeros(1, self.tree_lstm.hidden_size)
        if right_child is not None:
            right_hidden, right_cell, outputs = self.recursive_eval(
                right_child, treatment_id, left_hidden, left_cell, right_hidden, right_cell, outputs,feature_scaler, output_scaler
            )
        else:
            right_hidden, right_cell = torch.zeros(1, self.tree_lstm.hidden_size), torch.zeros(1, self.tree_lstm.hidden_size)
        
        module_index = self.module_names.index(module_name)
        module_tensor = torch.tensor([module_index], dtype=torch.float32).unsqueeze(0)
        # convert treatment_id to tensor
        treatment_tensor = torch.tensor([treatment_id], dtype=torch.float32).unsqueeze(0)
        features = torch.tensor(node["features"], dtype=torch.float32).unsqueeze(0)
        input_tensor = torch.cat([features, treatment_tensor], dim=1)
        module_tree_lstm = self.module_lstms[module_index]
        module_output_layer = self.output_layers[module_index]

        hidden, cell = module_tree_lstm(input_tensor, left_hidden, left_cell, right_hidden, right_cell)
        output = module_output_layer(hidden)
        outputs.insert(0, output)
        return hidden, cell, outputs

    def forward(self, expr_tree, treatment_id, feature_scaler, output_scaler):
        left_hidden = torch.zeros(1, self.tree_lstm.hidden_size)
        left_cell = torch.zeros(1, self.tree_lstm.hidden_size)
        right_hidden = torch.zeros(1, self.tree_lstm.hidden_size)
        right_cell = torch.zeros(1, self.tree_lstm.hidden_size)
        _, _, outputs = self.recursive_eval(expr_tree, treatment_id, left_hidden, left_cell, right_hidden, right_cell, [], feature_scaler, output_scaler) 
        return outputs


# let's first work on the modular mlp predictor architecture in which each module has its own mlp.
# And modules are connected in a tree structure.
# each module uses the features of the current node, the output of the left child, the output of the right child, and the treatment id to predict the output of the current node.
class ModularTreeMLPPredictor(nn.Module):
    def __init__(self, module_feature_dict, extra_dim, hidden_size):
        super(ModularTreeMLPPredictor, self).__init__()
        self.module_feature_dict = module_feature_dict
        self.module_names = list(module_feature_dict.keys())
        self.module_mlps = nn.ModuleList([MLP(module_feature_dict[module_name] + extra_dim, hidden_size) for module_name in self.module_names])

    def recursive_eval(self, node, treatment_id, outputs):
    
        module_name, left_child, right_child = node.module_name, node.left_child, node.right_child
        if left_child is not None:
            left_child_output, outputs = self.recursive_eval(left_child, treatment_id, outputs)
        else:
            left_child_output = torch.tensor([0.0], dtype=torch.float32).unsqueeze(0)

        if right_child is not None:
            right_child_output, outputs = self.recursive_eval(right_child, treatment_id, outputs)
        else:
            right_child_output = torch.tensor([0.0], dtype=torch.float32).unsqueeze(0)
        module_index = self.module_names.index(module_name)
        mlp = self.module_mlps[module_index]
        treatment_tensor = torch.tensor([treatment_id], dtype=torch.float32).unsqueeze(0)
    
        features = torch.tensor(node.features, dtype=torch.float32).unsqueeze(0)
        
        input_tensor = torch.cat([features, left_child_output, right_child_output], dim=1)
        input_tensor = torch.cat([input_tensor, treatment_tensor], dim=1)
        output = mlp(input_tensor)
        outputs.insert(0, output)
        return output, outputs

    def forward(self, expr_tree, treatment_id):
        current_output, outputs = self.recursive_eval(expr_tree, treatment_id, [])
        return outputs
# class ModularMatrixOperationPredictor(nn.Module):
#     def __init__(self, input_size, hidden_size, operators):
#         super(ModularMatrixOperationPredictor, self).__init__()
#         self.operators = ["det","elementwise_add", "elementwise_mul", "elementwise_sub", "inverse", "LU", "matmul", "norm", "QR", "SVD", "TR",  "trace"]
#         self.ops_dir = {
#         "-": "elementwise_sub",
#         "+": "elementwise_add",
#         "*": "matmul",
#         "dot": "elementwise_mul",
#         "TR": "TR",
#         "inv": "inverse",
#         "det": "det",
#         "trace": "trace"}
#         self.module_lstms = nn.ModuleList([TreeLSTMMatrixOperation(input_size, hidden_size) for _ in operators])
#         self.output_layers = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in operators])


#     def recursive_eval(self, node, treatment_id, left_hidden, left_cell, right_hidden, right_cell, outputs,feature_scaler, output_scaler): 
#         operator, left_child, right_child = (
#             self.ops_dir.get(node["operator_name"],node["operator_name"]),
#             None,
#             None,
#         ) 

#         operator_index = self.operators.index(operator)
#         lstm = self.module_lstms[operator_index]
#         output_layer = self.output_layers[operator_index]

#         if node["left_child"] is not None:
#             left_child = node["left_child"] 
#         if node["right_child"] is not None:
#             right_child = node["right_child"] 

#         if left_child is not None:
#             left_hidden, left_cell, outputs = self.recursive_eval(
#                 left_child, treatment_id, left_hidden, left_cell, right_hidden, right_cell, outputs, feature_scaler, output_scaler) 
            
#         else:
#             left_hidden, left_cell = torch.zeros(1, self.module_lstms[0].hidden_size), torch.zeros(1, self.module_lstms[0].hidden_size)
#         if right_child is not None:
#             right_hidden, right_cell, outputs = self.recursive_eval(
#                 right_child, treatment_id, left_hidden, left_cell, right_hidden, right_cell, outputs,feature_scaler, output_scaler) 
            
#         else:
#             right_hidden, right_cell = torch.zeros(1, self.module_lstms[0].hidden_size), torch.zeros(1, self.module_lstms[0].hidden_size)
#         node["features"] = np.array([node["left_child_input_shape"], node["right_child_input_shape"] if node["right_child_input_shape"] is not None else 0, node["output_shape"]])
#         node["features"] = np.squeeze(feature_scaler.transform(node["features"].reshape(1,-1)))
#         operator_tensor = torch.tensor([operator_index], dtype=torch.float32).unsqueeze(0)
#         treatment_tensor = torch.tensor([treatment_id], dtype=torch.float32).unsqueeze(0)
#         features = torch.tensor(node["features"], dtype=torch.float32).unsqueeze(0)
#         input_tensor = torch.cat([features, operator_tensor], dim=1)
#         input_tensor = torch.cat([input_tensor, treatment_tensor], dim=1)
#         hidden, cell = lstm(input_tensor, left_hidden, left_cell, right_hidden, right_cell)
#         output = output_layer(hidden)
#         outputs.insert(0, output)
#         return hidden, cell, outputs


#     def forward(self, expr_tree, treatment_id, feature_scaler, output_scaler):
        
#         left_hidden = torch.zeros(1, self.module_lstms[0].hidden_size)
#         left_cell = torch.zeros(1, self.module_lstms[0].hidden_size)
#         right_hidden = torch.zeros(1, self.module_lstms[0].hidden_size)
#         right_cell = torch.zeros(1, self.module_lstms[0].hidden_size)
#         _, _, outputs = self.recursive_eval(expr_tree, treatment_id, left_hidden, left_cell, right_hidden, right_cell, [],feature_scaler, output_scaler)    
#         return outputs


# class ParallelAdditiveModel()