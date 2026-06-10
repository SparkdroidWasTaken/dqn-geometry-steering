import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DATA LOADING ---
path = "Cartpole RL"
with open(path + "states.json") as f: data = json.load(f)
with open(path + "ExperienceData_0.json") as f: exp = json.load(f)
with open(path + "NeuralNetwork_Fixed_Weights.json") as f: net = json.load(f)

episodes = np.array([d['episode'] for d in exp])
traj     = np.array([d['state'] for d in exp])

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
selected_act = "idn" 
manifold_model = CartpoleManifoldNet(net, act_fn_name=selected_act)
manifold_model.eval()

# --- COMPUTING METRIC TENSORS ALONG TRAJECTORY ---
print("Computing First Fundamental Form (Metric Tensors) via PyTorch Autograd...")

# Convert entire trajectory to tensor with gradient tracking enabled
traj_t = torch.tensor(traj, dtype=torch.float32, requires_grad=True)

# Extract Jacobians up front for every single state vector
jacobians = []
for i in range(len(traj)):
    jac = torch.autograd.functional.jacobian(manifold_model, traj_t[i])
    jacobians.append(jac.numpy())
jacobians = np.array(jacobians) # Shape: (N, 64, 5)

all_stats = []

for ep_num in range(1, episodes.max() + 1):
    mask = episodes == ep_num
    # Skip short/failed trials
    if np.sum(mask) < 4:
        continue
    
    # Extract raw parameters per specific episode to prevent index skewing
    ep_states = traj[mask]       # Shape: (N_steps, 5)
    ep_jacs   = jacobians[mask]  # Shape: (N_steps, 64, 5)
    
    # Compute local state derivatives (Transitions = N_steps - 1)
    ep_dx = np.diff(ep_states, axis=0) 
    ep_jacs_truncated = ep_jacs[:-1] 
    
    # Compute the First Fundamental Form: G = J^T * J
    G = np.matmul(np.transpose(ep_jacs_truncated, (0, 2, 1)), ep_jacs_truncated)
    
    # Calculate Manifold Speed: ds/dt = sqrt( dx^T * G * dx )
    speed_list = []
    for t in range(len(ep_dx)):
        dx = ep_dx[t]
        g_tensor = G[t]
        ds2 = np.dot(dx, np.dot(g_tensor, dx)) 
        speed_list.append(np.sqrt(max(ds2, 1e-8)))
        
    speed = np.array(speed_list)
    
    # --- RIEMANNIAN FRENET-SERRET FRAME ---
    # Tangent vectors: T = J * dx / speed
    T_list = []
    for t in range(len(ep_dx)):
        T_vec = np.dot(ep_jacs_truncated[t], ep_dx[t]) / speed[t]
        T_list.append(T_vec)
    T = np.array(T_list) # Shape: (N_steps - 1, 64)
    
    # Curvature (kappa = |dT/ds|)
    dT = np.diff(T, axis=0)
    dT_ds = dT / speed[:-1, None]
    curvature = np.linalg.norm(dT_ds, axis=1)
    N = dT_ds / (curvature[:, None] + 1e-8)
    
    # Torsion (tau) via Orthogonalized Gram-Schmidt
    torsion_list = []
    for i in range(len(N) - 1):
        t_vec, n_vec = T[i], N[i]
        dn_ds = (N[i+1] - N[i]) / speed[i]
        b_vec = dn_ds - np.dot(dn_ds, t_vec)*t_vec - np.dot(dn_ds, n_vec)*n_vec
        b_norm = np.linalg.norm(b_vec)
        tau = -np.dot(dn_ds, b_vec / b_norm) if b_norm > 1e-6 else 0.0
        torsion_list.append(tau)
    torsion = np.array(torsion_list)
    
    cumulative_arc = np.cumsum(speed)
    all_stats.append({
        'episode': ep_num,
        'arc_length': cumulative_arc[-1],
        'total_curvature': np.trapezoid(curvature, cumulative_arc[:-1]),
        'total_torsion': np.trapezoid(np.abs(torsion), cumulative_arc[:-2]),
    })
    print(f"Ep {ep_num:3d} | FFF_arc={all_stats[-1]['arc_length']:.1f} | κ={all_stats[-1]['total_curvature']:.1f} | τ={all_stats[-1]['total_torsion']:.1f}")

# --- SUMMARY STATISTICS ---
arc_lengths = [s['arc_length'] for s in all_stats]
total_curvatures = [s['total_curvature'] for s in all_stats]
total_torsions = [s['total_torsion'] for s in all_stats]

print(f"\n--- INTRINSIC MANIFOLD AVERAGES ({len(all_stats)} Episodes) ---")
print(f"Arc length (FFF): {np.mean(arc_lengths):.4f} ± {np.std(arc_lengths):.4f}")
print(f"Total curvature:  {np.mean(total_curvatures):.4f} ± {np.std(total_curvatures):.4f}")
print(f"Total torsion:    {np.mean(total_torsions):.4f} ± {np.std(total_torsions):.4f}")