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
        gate_probs = self.gate(x)
        final_output = torch.sum(gate_probs.unsqueeze(-1) * expert_outputs, dim=1)
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

def train_model(model, train_df, covariates, treatment, outcome, epochs, batch_size, val_df=None, plot=False):
    # use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training the model on {device}")
    model.to(device)

    # shuffle the training data
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    X = train_df[covariates].values
    T = train_df[treatment].values
    Y = train_df[outcome].values
    X_T = np.concatenate([X, T.reshape(-1, 1)], axis=1)
    # print(X_T.shape)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    X_T = torch.tensor(X_T, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # Prepare validation data if provided
    if val_df is not None:
        X_val = val_df[covariates].values
        T_val = val_df[treatment].values
        Y_val = val_df[outcome].values
        X_T_val = np.concatenate([X_val, T_val.reshape(-1, 1)], axis=1)
        X_T_val = torch.tensor(X_T_val, dtype=torch.float32)
        Y_val = torch.tensor(Y_val, dtype=torch.float32).reshape(-1, 1)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, X_T.shape[0], batch_size):
            X_T_batch = X_T[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model(X_T_batch)
            loss = loss_fn(output, Y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / (X_T.shape[0] // batch_size)
        train_losses.append(avg_train_loss)
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
        if ((epoch + 1) % 10 == 0 or epoch == epochs - 1) and plot == True:
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
def predict_model(model, test_df, covariates, return_effect=True, return_po=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X = test_df[covariates].values
    
    X_1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    X_0 = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
    X_1 = torch.tensor(X_1, dtype=torch.float32).to(device)
    X_0 = torch.tensor(X_0, dtype=torch.float32).to(device)

    with torch.no_grad():
        predicted_outcome_1 = model(X_1).cpu().numpy()
        predicted_outcome_0 = model(X_0).cpu().numpy()
        predicted_effect = predicted_outcome_1 - predicted_outcome_0

    if return_effect:
        return predicted_effect
    else:
        return predicted_outcome_1, predicted_outcome_0