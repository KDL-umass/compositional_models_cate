import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from catenets.models.jax import TNet, SNet, DRNet, SNet1, SNet2
# set the seed
torch.manual_seed(42)


# Define the baseline model that just takes the covariates and treatment as input and predicts the outcome
class BaselineModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaselineModel, self).__init__()
        self.ff_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.ff_network(x)

def train_catenets(train_df, covariates, treatment, outcome, baseline="tnet"):
    X = train_df[covariates].values
    T = train_df[treatment].values
    Y = train_df[outcome].values
    if baseline == "tnet":
        est = TNet()
    elif baseline == "snet":
        est = SNet()
    elif baseline == "snet1":
        est = SNet1()
    elif baseline == "snet2":
        est = SNet2()
    elif baseline == "drnet":
        est = DRNet()
    est.fit(X,Y,T)

    return est

def predict_catenets(model, test_df, covariates):
    X = test_df[covariates].values
    causal_effect_estimates = model.predict(X)
    return causal_effect_estimates
    
class MoE(nn.Module):
    # define the __init__ method
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MoE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList([create_expert_model(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = create_gate_model(input_dim, num_experts)
        # define an aggregation model that combines the outputs of the experts

    # define the forward method
    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # squeeze the expert outputs to remove the last dimension
        # gate_probs = self.gate(x)
        final_output = torch.sum(expert_outputs, dim=1)
        return final_output

class MoEknownCov(nn.Module):
    # define the __init__ method
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, feature_dim):
        super(MoEknownCov, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.expert_input_dim = feature_dim + 1
        
        self.experts = nn.ModuleList([create_expert_model(self.expert_input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = create_gate_model(input_dim, num_experts)
        # define an aggregation model that combines the outputs of the experts

    # define the forward method
    def forward(self, x):
        expert_outputs = []
        for i in range(self.num_experts):
            
            expert_input = x[:, i, :]
            
            expert_output = self.experts[i](expert_input)
            # append the expert output to the expert outputs list
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        # squeeze the expert outputs to remove the last dimension
        # gate_probs = self.gate(x)
        final_output = torch.sum(expert_outputs, dim=1)
        return final_output

# write a function that creates the expert model
def create_expert_model(input_dim, hidden_dim, output_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )
    return model

# write a function that creates the gate model
def create_gate_model(input_dim, num_experts):
    model = nn.Sequential(
        nn.Linear(input_dim, num_experts),
        nn.Softmax(dim=1)
    )
    return model

def train_model(model, train_df, covariates, treatment, outcome, epochs, batch_size, num_modules, num_feature_dimensions, val_df=None, plot=False, model_name="MoE", scheduler_flag=False):
    # use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training the model on {device}")
    model.to(device)

    # shuffle the training data
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    
    
    T = train_df[treatment].values
    Y = train_df[outcome].values
    if model_name in ["Baseline", "MoE"]:
        X = train_df[covariates].values
        X_T = np.concatenate([X, T.reshape(-1, 1)], axis=1)
        X_T = torch.tensor(X_T, dtype=torch.float32).to(device)
    else:
        # get feature of each module
        X_module_wise_array = []
        for i in range(num_modules):
            module_features = [f"module_{i+1}_feature_feature_{j}" for j in range(num_feature_dimensions)]
            X_module_wise = train_df[module_features].values.reshape(-1, num_feature_dimensions)
            X_module_wise_T = np.concatenate([X_module_wise, T.reshape(-1, 1)], axis=1)
            X_module_wise_T = torch.tensor(X_module_wise_T, dtype=torch.float32).to(device)
            X_module_wise_array.append(X_module_wise_T)

    Y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1).to(device)

    loss_fn = nn.MSELoss()
    if scheduler_flag == False:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Slightly lower initial learning rate

        # Cosine annealing with warm restarts
        # T_0 is the number of epochs before first restart
        # T_mult=2 means each restart cycle will be twice as long as the previous one
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # First restart occurs after 10 epochs
            T_mult=2,  # Each restart cycle is twice as long
            eta_min=1e-6  # Minimum learning rate
        )
    
    
    # Prepare validation data if provided
    if val_df is not None:
        if model_name in ["Baseline", "MoE"]:
            X_val = val_df[covariates].values
            X_T_val = np.concatenate([X_val, val_df[treatment].values.reshape(-1, 1)], axis=1)
            X_T_val = torch.tensor(X_T_val, dtype=torch.float32).to(device)
        else:
            X_module_wise_val_array = []
            for i in range(num_modules):
                module_features = [f"module_{i+1}_feature_feature_{j}" for j in range(num_feature_dimensions)]
                X_module_wise_val = val_df[module_features].values
                X_module_wise_T_val = np.concatenate([X_module_wise_val, val_df[treatment].values.reshape(-1, 1)], axis=1)
                X_module_wise_T_val = torch.tensor(X_module_wise_T_val, dtype=torch.float32).to(device)
                X_module_wise_val_array.append(X_module_wise_T_val)
            Y_val = torch.tensor(val_df[outcome].values, dtype=torch.float32).reshape(-1, 1).to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, Y.shape[0], batch_size):
            if model_name in ["Baseline", "MoE"]:
                X_T_batch = X_T[i:i+batch_size]
            else:
                # get list of tensors for each module
                X_T_batch = [X_module_wise_array[j][i:i+batch_size] for j in range(num_modules)] # make it a list of tensors
                X_T_batch = torch.stack(X_T_batch, dim=1) # stack the list of tensors to make a tensor
                
                
            Y_batch = Y[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model(X_T_batch)
            loss = loss_fn(output, Y_batch)
            loss.backward()
            # Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / (Y.shape[0] // batch_size)
        train_losses.append(avg_train_loss)
        if scheduler_flag == True:
            scheduler.step()
        # print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation step
        if val_df is not None:
            model.eval()
            with torch.no_grad():
                val_output = model(X_T_val)
                val_loss = loss_fn(val_output, Y_val)
                val_losses.append(val_loss.item())
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
        # else:
        #     print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Plot losses every 10 epochs or at the end of training
        if (epoch == epochs - 1) and plot == True:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch + 2), train_losses, label='Training Loss')
            if val_df is not None:
                plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.show()
    
    return model, train_losses, val_losses
# Prediction function for both models
def predict_model(model, test_df, covariates, num_modules, num_feature_dimensions, return_effect=True, return_po=False, model_name="MoE"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if model_name in ["Baseline", "MoE"]:
        X = test_df[covariates].values
    
        X_1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        X_0 = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
        X_1 = torch.tensor(X_1, dtype=torch.float32).to(device)
        X_0 = torch.tensor(X_0, dtype=torch.float32).to(device)
    else:
        X_1 = []
        X_0 = []
        for i in range(num_modules):
            module_features = [f"module_{i+1}_feature_feature_{j}" for j in range(num_feature_dimensions)]
            X_module_wise = test_df[module_features].values.reshape(-1, num_feature_dimensions)
            X_module_wise_1 = np.concatenate([X_module_wise, np.ones((X_module_wise.shape[0], 1))], axis=1)
            X_module_wise_0 = np.concatenate([X_module_wise, np.zeros((X_module_wise.shape[0], 1))], axis=1)
            X_module_wise_1 = torch.tensor(X_module_wise_1, dtype=torch.float32).to(device)
            X_module_wise_0 = torch.tensor(X_module_wise_0, dtype=torch.float32).to(device)
            X_1.append(X_module_wise_1)
            X_0.append(X_module_wise_0)

        # convert the list of tensors to a tensor whuch can still be indexed without indexing
        X_1 = torch.stack(X_1, dim=1)
        X_0 = torch.stack(X_0, dim=1)
   

        
    with torch.no_grad():
        predicted_outcome_1 = model(X_1).cpu().numpy()
        predicted_outcome_0 = model(X_0).cpu().numpy()
        predicted_effect = predicted_outcome_1 - predicted_outcome_0

    if return_effect:
        return predicted_effect
    else:
        return predicted_outcome_1, predicted_outcome_0