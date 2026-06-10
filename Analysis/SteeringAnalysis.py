import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
path     = "Cartpole RL"
net_path = path + "NeuralNetwork_Fixed_Weights.json"

EPISODES = {
    "Normal":  path + "EpisodeNormal.json",
    "Idn":     path + "EpisodeIdn.json",
    "Lrelu":   path + "EpisodeLrelu.json",
    "Tanh":    path + "EpisodeTanh.json",
    "Sigmoid": path + "EpisodeSigmoid.json",
}

ACT_NAMES = ["idn", "lrelu", "tanh", "sigmoid"]

# =============================================================================
# NEURAL NETWORK
# =============================================================================
class CartpoleManifoldNet(nn.Module):
    def __init__(self, net_json, act_fn_name="idn"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.act_fn = {
            "lrelu":   lambda x: F.leaky_relu(x, negative_slope=0.01),
            "tanh":    torch.tanh,
            "sigmoid": torch.sigmoid,
            "idn":     lambda x: x,
        }.get(act_fn_name, lambda x: x)
        for layer_data in net_json['layers'][:-1]:
            layer = nn.Linear(layer_data['inputsCount'], layer_data['outputsCount'])
            layer.weight.data = torch.tensor(layer_data['weights'], dtype=torch.float32)
            layer.bias.data   = torch.tensor(layer_data['biases'],  dtype=torch.float32)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = self.act_fn(layer(x))
        return x

# =============================================================================
# FRENET-SERRET GEOMETRY
# =============================================================================
def compute_geometry(states, model):
    jacobians = []
    for i in range(len(states)):
        s = torch.tensor(states[i], dtype=torch.float32).requires_grad_(True)
        jac = torch.autograd.functional.jacobian(model, s)
        jacobians.append(jac.detach().numpy())
    jacobians = np.array(jacobians)

    dx   = np.diff(states, axis=0)
    jacs = jacobians[:-1]

    G = np.matmul(np.transpose(jacs, (0, 2, 1)), jacs)
    speed = np.array([
        np.sqrt(max(float(dx[t] @ G[t] @ dx[t]), 1e-8))
        for t in range(len(dx))
    ])

    T = np.array([jacs[t] @ dx[t] / speed[t] for t in range(len(dx))])

    dT    = np.diff(T, axis=0)
    dT_ds = dT / speed[:-1, None]
    kappa = np.linalg.norm(dT_ds, axis=1)
    N_vec = dT_ds / (kappa[:, None] + 1e-8)

    torsion = []
    for i in range(len(N_vec) - 1):
        t_v, n_v = T[i], N_vec[i]
        dn_ds = (N_vec[i+1] - N_vec[i]) / speed[i]
        b_v   = dn_ds - np.dot(dn_ds, t_v)*t_v - np.dot(dn_ds, n_v)*n_v
        b_n   = np.linalg.norm(b_v)
        torsion.append(-np.dot(dn_ds, b_v / b_n) if b_n > 1e-6 else 0.0)
    torsion = np.array(torsion)

    arc = np.cumsum(speed)
    return {
        'arc':       float(arc[-1]),
        'kappa':     float(np.trapezoid(kappa,          arc[:-1])),
        'tau':       float(np.trapezoid(np.abs(torsion), arc[:-2])),
        'kappa_arc': float(np.trapezoid(kappa,          arc[:-1]) / arc[-1]),
    }

# =============================================================================
# MAIN — build results[episode][activation]
# =============================================================================
with open(net_path) as f:
    net = json.load(f)

# Load all episode state arrays + rewards once
ep_data = {}
for ep_name, ep_path in EPISODES.items():
    with open(ep_path) as f:
        ep = json.load(f)
    ep_data[ep_name] = {
        'states':       np.array([d['state']  for d in ep], dtype=np.float32),
        'avg_reward':   float(np.mean([d['reward'] for d in ep])),
        'total_reward': float(np.sum ([d['reward'] for d in ep])),
    }

# results[ep_name][act_name] = geometry dict
results = {ep: {} for ep in EPISODES}

total = len(EPISODES) * len(ACT_NAMES)
done  = 0
for ep_name, edata in ep_data.items():
    for act in ACT_NAMES:
        done += 1
        print(f"[{done}/{total}] {ep_name} × {act} ...", flush=True)
        model = CartpoleManifoldNet(net, act_fn_name=act)
        model.eval()
        results[ep_name][act] = compute_geometry(edata['states'], model)

# =============================================================================
# PRINT FULL 5×4 TABLE
# =============================================================================
print("\n\n" + "="*90)
print("FULL GEOMETRY TABLE  (episode × activation)")
print("="*90)
header = f"{'Episode':<10} {'Act':<8} {'Arc':>10} {'κ':>10} {'τ':>10} {'κ/arc':>8} {'AvgR':>8} {'TotalR':>10}"
print(header)
print("-"*90)
for ep_name, edata in ep_data.items():
    for act in ACT_NAMES:
        g = results[ep_name][act]
        print(f"{ep_name:<10} {act:<8} "
              f"{g['arc']:>10.2f} {g['kappa']:>10.2f} {g['tau']:>10.2f} "
              f"{g['kappa_arc']:>8.4f} "
              f"{edata['avg_reward']:>8.4f} {edata['total_reward']:>10.2f}")
    print()

# =============================================================================
# DELTA TABLE — each steered episode vs Normal, per activation
# =============================================================================
print("="*90)
print("DELTA TABLE  (steered - Normal, per activation)")
print("="*90)
print(f"{'Episode':<10} {'Act':<8} {'Δarc':>10} {'Δκ':>10} {'Δτ':>10} {'Δ(κ/arc)':>10} {'ΔTotalR':>10}")
print("-"*90)
for ep_name in ['Idn', 'Lrelu', 'Tanh', 'Sigmoid']:
    for act in ACT_NAMES:
        g    = results[ep_name][act]
        base = results['Normal'][act]
        print(f"{ep_name:<10} {act:<8} "
              f"{g['arc']-base['arc']:>+10.2f} "
              f"{g['kappa']-base['kappa']:>+10.2f} "
              f"{g['tau']-base['tau']:>+10.2f} "
              f"{g['kappa_arc']-base['kappa_arc']:>+10.4f} "
              f"{ep_data[ep_name]['total_reward']-ep_data['Normal']['total_reward']:>+10.2f}")
    print()