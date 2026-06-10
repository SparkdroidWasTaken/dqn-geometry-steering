import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

# --- DATA LOADING ---
path = "Cartpole RL"
with open(path + "states.json") as f: data = json.load(f)
with open(path + "ExperienceData_0.json") as f: exp = json.load(f)
with open(path + "NeuralNetwork_Fixed_Weights.json") as f: net = json.load(f)

states   = np.array([d['state'] for d in data])
rewards  = np.array([d['reward'] for d in data])
episodes = np.array([d['episode'] for d in exp])
traj     = np.array([d['state'] for d in exp])
#perfect state:
#[Math.cos(0.5), Math.sin(0.5),0,0,1,0,0]

#initial state
#[Math.cos(0.5),Math.sin(0.5),0,Math.sin((Math.PI/2)-0.5),Math.cos((Math.PI/2)-0.5),0,0]
class CartpoleManifoldNet(nn.Module):
    def __init__(self, net_json, act_fn_name="sigmoid"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.act_fn = {
            "lrelu": lambda x: F.leaky_relu(x, negative_slope=0.01),
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "idn": lambda x: x
        }.get(act_fn_name, lambda x: x)  # Secure fallback to identity step

        for layer_data in net_json['layers'][:-1]: 
            layer = nn.Linear(layer_data['inputsCount'], layer_data['outputsCount'])
            layer.weight.data = torch.tensor(layer_data['weights'], dtype=torch.float32)
            layer.bias.data = torch.tensor(layer_data['biases'], dtype=torch.float32)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = self.act_fn(layer(x))
        return x

# Switch here: "lrelu", "tanh", "sigmoid", or "idn"
selected_act = "lrelu" 
manifold_model = CartpoleManifoldNet(net, act_fn_name=selected_act)
manifold_model.eval()



with torch.no_grad():
    states_t = torch.tensor(states, dtype=torch.float32)
    traj_t   = torch.tensor(traj, dtype=torch.float32)
    hidden_traj = manifold_model(traj_t).numpy()
        
    # Get 64D activations
    hidden_all = manifold_model(states_t).numpy()
    hidden_traj = manifold_model(traj_t).numpy()


# Define states
perfect_state = np.array([
    np.cos(0.5), np.sin(0.5), 0,
    0, 1, 0, 0
], dtype=np.float32)

initial_state = np.array([
    np.cos(0.5), np.sin(0.5), 0,
    np.sin(np.pi/2 - 0.5), np.cos(np.pi/2 - 0.5), 0, 0
], dtype=np.float32)

# Get activations
with torch.no_grad():
    perfect_hidden = manifold_model(torch.tensor(perfect_state)).numpy()
    initial_hidden = manifold_model(torch.tensor(initial_state)).numpy()

# Linear steering vector
v_linear = perfect_hidden - initial_hidden
v_linear = v_linear / np.linalg.norm(v_linear)
with open(path + "v_linear_relu.json", "w") as f:
    json.dump(v_linear.tolist(), f)
