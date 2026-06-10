import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json as json_mod

# =============================================================================
# DATA LOADING
# =============================================================================
path = "Cartpole RL"
with open(path + "states.json")        as f: data    = json.load(f)
with open(path + "ExperienceData_0.json") as f: exp     = json.load(f)
with open(path + "NeuralNetwork_Fixed_Weights.json")   as f: net     = json.load(f)

# Special episodes — each is an independently-trained/evaluated run.
# Slider indices 50-54 respectively. "Normal" = standard training, others = trained
# with that activation function baked in during training (not just at readout).
with open(path + "EpisodeNormal.json")  as f: ep_normal  = json.load(f)
with open(path + "EpisodeIdn.json")     as f: ep_idn     = json.load(f)
with open(path + "EpisodeLrelu.json")   as f: ep_lrelu   = json.load(f)
with open(path + "EpisodeTanh.json")    as f: ep_tanh    = json.load(f)
with open(path + "EpisodeSigmoid.json") as f: ep_sigmoid = json.load(f)

states     = np.array([d['state']   for d in data])
rewards    = np.array([d['reward']  for d in data])
episodes   = np.array([d['episode'] for d in exp])
traj       = np.array([d['state']   for d in exp])
unique_eps = np.unique(episodes)

# Special episode state arrays + avg rewards
SPECIAL_EPS = [
    # (slider_idx, label, colour, json_data)
    # Colour shown in the 3D plot for that episode's trajectory
    (50, "Normal",  "#C8960C", ep_normal),   # gold
    (51, "Idn",     "#9B30FF", ep_idn),      # purple
    (52, "Lrelu",   "#E05C00", ep_lrelu),    # orange
    (53, "Tanh",    "#0099CC", ep_tanh),     # teal
    (54, "Sigmoid", "#CC0066", ep_sigmoid),  # pink
]

# Avg reward per original episode
ep_step_rewards = np.array([d['reward'] for d in exp])
ep_avg_rewards = {}
for ep_num in unique_eps:
    mask = episodes == ep_num
    ep_avg_rewards[int(ep_num)] = float(np.mean(ep_step_rewards[mask]))

# Avg reward per special episode (keyed by label)
special_avg_rewards = {
    label: float(np.mean([d['reward'] for d in jdata]))
    for _, label, _, jdata in SPECIAL_EPS
}

# Slider position NUM_EPISODES (55) = "Space" (full cloud, no trajectory)
NUM_EPISODES = 55   # 0-49 = original eps, 50-54 = specials, 55 = Space


# =============================================================================
# NEURAL NETWORK
# =============================================================================
class CartpoleManifoldNet(nn.Module):
    def __init__(self, net_json, act_fn_name="lrelu"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.act_fn = {
            "lrelu":   lambda x: F.leaky_relu(x, negative_slope=0.01),
            "tanh":    torch.tanh,
            "sigmoid": torch.sigmoid,
            "idn":     lambda x: x,
        }.get(act_fn_name, lambda x: F.leaky_relu(x, negative_slope=0.01))
        for layer_data in net_json['layers'][:-1]:
            layer = nn.Linear(layer_data['inputsCount'], layer_data['outputsCount'])
            layer.weight.data = torch.tensor(layer_data['weights'], dtype=torch.float32)
            layer.bias.data   = torch.tensor(layer_data['biases'],  dtype=torch.float32)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = self.act_fn(layer(x))
        return x

ACT_NAMES = ["idn", "lrelu", "tanh", "sigmoid"]
ACT_LABELS = {"idn": "Identity", "lrelu": "Leaky ReLU", "tanh": "Tanh", "sigmoid": "Sigmoid"}


# =============================================================================
# PRE-COMPUTE PCA FOR EVERY ACTIVATION SPACE
# PCA is fit on states_67.json only; all episodes are projected into that same space.
# =============================================================================
states_t = torch.tensor(states, dtype=torch.float32)
traj_t   = torch.tensor(traj,   dtype=torch.float32)

# Pre-convert special episode state arrays to tensors
special_tensors = {
    label: torch.tensor(np.array([d['state'] for d in jdata]), dtype=torch.float32)
    for _, label, _, jdata in SPECIAL_EPS
}

all_spaces = {}
for act in ACT_NAMES:
    model = CartpoleManifoldNet(net, act_fn_name=act)
    model.eval()
    with torch.no_grad():
        h_all  = model(states_t).numpy()
        h_traj = model(traj_t).numpy()
        h_specials = {label: model(t).numpy() for label, t in special_tensors.items()}

    scaler   = StandardScaler().fit(h_all)
    h_scaled = scaler.transform(h_all)
    pca      = PCA(n_components=3).fit(h_scaled)
    p        = pca.transform(h_scaled)
    pc       = pca.transform(scaler.transform(h_traj))
    pc_specials = {
        label: pca.transform(scaler.transform(h))
        for label, h in h_specials.items()
    }

    all_spaces[act] = {"p": p.tolist(), "pc": pc.tolist(), "pc_specials": {
        label: arr.tolist() for label, arr in pc_specials.items()
    }}
    print(f"  Computed {act}")


# =============================================================================
# BUILD EPISODE LOOKUP TABLE
# Slider indices 0-49  = original episodes
# Slider indices 50-54 = special episodes (Normal/Idn/Lrelu/Tanh/Sigmoid)
# Slider index   55    = Space (handled in JS, no entry needed)
# =============================================================================
# Special episode colour map — passed to JS so it can colour trajectories correctly
special_colours = {label: colour for _, label, colour, _ in SPECIAL_EPS}

js_data = {}
for act in ACT_NAMES:
    p  = np.array(all_spaces[act]["p"])
    pc = np.array(all_spaces[act]["pc"])
    ep_data = {}

    # Original 50 episodes
    for ep_idx in range(50):
        ep_num = int(unique_eps[ep_idx]) if ep_idx < len(unique_eps) else int(unique_eps[-1])
        mask   = episodes == ep_num
        ep_pc  = pc[mask]
        avg_r  = ep_avg_rewards.get(ep_num, 0.0)
        if len(ep_pc) > 0:
            ep_data[str(ep_idx)] = {
                "label": f"Ep {ep_num}", "traj": ep_pc.tolist(),
                "start": ep_pc[0].tolist(), "end": ep_pc[-1].tolist(),
                "avg_reward": round(avg_r, 3), "special": False,
            }
        else:
            ep_data[str(ep_idx)] = {
                "label": f"Ep {ep_num}", "traj": [],
                "start": None, "end": None,
                "avg_reward": round(avg_r, 3), "special": False,
            }

    # Special episodes (indices 50-54)
    for slider_idx, label, colour, _ in SPECIAL_EPS:
        pc_s = np.array(all_spaces[act]["pc_specials"][label])
        ep_data[str(slider_idx)] = {
            "label":      label,
            "traj":       pc_s.tolist(),
            "start":      pc_s[0].tolist(),
            "end":        pc_s[-1].tolist(),
            "avg_reward": round(special_avg_rewards[label], 3),
            "special":    True,
            "colour":     colour,
        }

    js_data[act] = {"p": p.tolist(), "ep": ep_data}

rewards_list = rewards.tolist()
data_json = json_mod.dumps({
    "spaces":  js_data,
    "rewards": rewards_list,
    "acts":    ACT_NAMES,
    "labels":  ACT_LABELS,
})


# =============================================================================
# HTML OUTPUT
# =============================================================================
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CartPole Neural Manifold</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    display: flex;
    height: 100vh;
    background: #FFFBF5;
    font-family: 'Comic Sans MS', 'Comic Sans', cursive, sans-serif;
    color: #1A1A1E;
    overflow: hidden;
  }}

  #sidebar {{
    width: 160px;
    min-width: 160px;
    display: flex;
    flex-direction: column;
    padding: 18px 12px;
    background: #F9EED8;
    border-right: 1px solid #E6DFD3;
    gap: 8px;
    z-index: 10;
    overflow-y: auto;
  }}
  #sidebar h3 {{
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #55555A;
    margin-bottom: 2px;
  }}

  .act-btn {{
    padding: 10px 8px;
    border: 1px solid #B0C4DE;
    border-radius: 6px;
    background: #A2C2E8;
    color: #2C3E50;
    font-size: 12px;
    font-family: 'Comic Sans MS', 'Comic Sans', cursive, sans-serif;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s ease;
    line-height: 1.3;
  }}
  .act-btn:hover  {{ background: #B9D3F3; color: #1A2530; border-color: #8FAECF; }}
  .act-btn.active {{ background: #4A7BB0; color: #FFFFFF; border-color: #355C8A;
                     box-shadow: 0 2px 6px rgba(74,123,176,0.2); }}
  .act-btn .mono  {{ font-size: 10px; color: #4A5D6E; display: block; margin-top: 2px; }}
  .act-btn.active .mono {{ color: #D1E2F4; }}

  /* ---- STATS PANEL ---- */
  #stats-section {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 4px;
    padding-top: 10px;
    border-top: 1px solid #E6DFD3;
  }}
  #stats-ep-label {{
    font-size: 12px;
    font-weight: bold;
    color: #2C3E50;
    margin-bottom: 2px;
  }}
  .stat-row {{
    display: flex;
    flex-direction: column;
    background: #EFE4CC;
    border: 1px solid #D9CDB8;
    border-radius: 5px;
    padding: 5px 7px;
  }}
  /* Special episodes get a left colour bar via border-left */
  .stat-row.special {{
    border-left-width: 3px;
  }}
  .stat-name {{
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7A6E60;
  }}
  .stat-val {{
    font-size: 13px;
    color: #1A1A1E;
    font-weight: bold;
    margin-top: 1px;
  }}
  #stats-placeholder {{
    font-size: 10px;
    color: #9A9490;
    font-style: italic;
    text-align: center;
    padding: 6px 0;
  }}

  /* ---- TIMESTEP SLIDERS ---- */
  #ts-section {{
    display: none;
    flex-direction: column;
    gap: 6px;
    margin-top: 8px;
    padding-top: 10px;
    border-top: 1px solid #E6DFD3;
  }}
  #ts-section h3 {{ margin-bottom: 0; }}
  #ts-label {{
    font-size: 11px;
    color: #2C3E50;
    text-align: center;
  }}
  #ts-reset {{
    padding: 5px 6px;
    border: 1px solid #B0C4DE;
    border-radius: 5px;
    background: #A2C2E8;
    color: #2C3E50;
    font-size: 10px;
    font-family: 'Comic Sans MS', 'Comic Sans', cursive, sans-serif;
    cursor: pointer;
    text-align: center;
  }}
  #ts-reset:hover {{ background: #B9D3F3; }}

  #main  {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
  #plot  {{ flex: 1; }}
</style>
</head>
<body>

<div id="sidebar">
  <h3>Activation</h3>

  <div id="stats-section">
    <h3>Episode Stats</h3>
    <div id="stats-ep-label">—</div>
    <div id="stats-rows">
      <div id="stats-placeholder">Select an episode</div>
    </div>
  </div>

  <div id="ts-section">
    <h3>Timestep</h3>
    <div id="ts-label">— / —</div>
    <label id="ts-from-label" style="font-size:10px;color:#55555A;">From: 1</label>
    <input id="ts-from" type="range" min="1" max="1" value="1">
    <label id="ts-to-label" style="font-size:10px;color:#55555A;">To: 1</label>
    <input id="ts-to" type="range" min="1" max="1" value="1">
    <button id="ts-reset">Show all</button>
  </div>
</div>

<div id="main">
  <div id="plot"></div>
</div>

<script>
// ---- DATA (injected by Python) ----
const RAW        = {data_json};
const spaces     = RAW.spaces;
const rewards    = RAW.rewards;
const ACT_NAMES  = RAW.acts;
const ACT_LABELS = RAW.labels;
const NUM_EPISODES = 55;  // 0-49 = original, 50-54 = specials, 55 = Space

// ---- FRENET-SERRET STATS ----
// Keyed by act → episode identifier (number for originals, label string for specials)
const STATS = {{
  "idn": {{
    1:{{arc:5754.9,kappa:1089.3,tau:1115.3}},2:{{arc:5334.8,kappa:1190.2,tau:1222.9}},
    3:{{arc:5426.7,kappa:1200.4,tau:1222.4}},4:{{arc:5724.5,kappa:1142.7,tau:1161.0}},
    5:{{arc:6395.8,kappa:901.8,tau:932.1}},6:{{arc:5840.0,kappa:1152.0,tau:1181.2}},
    7:{{arc:6033.9,kappa:965.6,tau:993.8}},8:{{arc:5691.9,kappa:1078.3,tau:1102.7}},
    9:{{arc:5725.6,kappa:1120.5,tau:1148.2}},10:{{arc:5448.2,kappa:1203.9,tau:1237.6}},
    11:{{arc:5487.6,kappa:1168.8,tau:1205.5}},12:{{arc:5741.2,kappa:1102.4,tau:1127.6}},
    13:{{arc:5450.5,kappa:1155.9,tau:1186.7}},14:{{arc:5485.6,kappa:1174.5,tau:1198.3}},
    15:{{arc:5410.3,kappa:1191.9,tau:1218.2}},16:{{arc:6009.8,kappa:1142.5,tau:1169.8}},
    17:{{arc:5817.2,kappa:1111.8,tau:1142.1}},18:{{arc:5857.4,kappa:1132.4,tau:1158.4}},
    19:{{arc:5281.5,kappa:1169.6,tau:1199.9}},20:{{arc:5671.2,kappa:1135.1,tau:1164.6}},
    21:{{arc:5495.0,kappa:1188.3,tau:1216.1}},22:{{arc:5328.4,kappa:1096.4,tau:1127.2}},
    23:{{arc:5403.3,kappa:1125.2,tau:1154.8}},24:{{arc:5816.5,kappa:1164.6,tau:1197.8}},
    25:{{arc:5281.0,kappa:1152.7,tau:1184.8}},26:{{arc:5414.7,kappa:1171.3,tau:1203.2}},
    27:{{arc:5787.8,kappa:1136.3,tau:1168.1}},28:{{arc:6433.1,kappa:1000.0,tau:960.2}},
    29:{{arc:6148.4,kappa:970.4,tau:985.4}},30:{{arc:5381.1,kappa:1192.1,tau:1226.5}},
    31:{{arc:94.1,kappa:71.4,tau:17.8}},32:{{arc:6099.3,kappa:1104.1,tau:1134.3}},
    33:{{arc:5444.6,kappa:1191.6,tau:1230.0}},34:{{arc:5350.1,kappa:1136.0,tau:1162.4}},
    35:{{arc:5775.4,kappa:1124.4,tau:1132.3}},36:{{arc:5640.5,kappa:1176.6,tau:1195.0}},
    37:{{arc:5818.3,kappa:1108.5,tau:1135.3}},38:{{arc:5596.9,kappa:1132.1,tau:1156.6}},
    39:{{arc:5383.2,kappa:1179.3,tau:1213.1}},40:{{arc:5706.7,kappa:1154.8,tau:1177.4}},
    41:{{arc:5726.7,kappa:1114.8,tau:1148.0}},42:{{arc:6031.7,kappa:1093.7,tau:1113.7}},
    43:{{arc:5602.8,kappa:1136.2,tau:1151.4}},44:{{arc:5562.8,kappa:1133.1,tau:1152.9}},
    45:{{arc:5804.5,kappa:1135.1,tau:1160.8}},46:{{arc:5416.4,kappa:1190.8,tau:1223.2}},
    47:{{arc:5886.8,kappa:1123.1,tau:1150.7}},48:{{arc:5298.2,kappa:1089.8,tau:1114.3}},
    49:{{arc:5737.5,kappa:1164.0,tau:1182.5}},50:{{arc:5325.5,kappa:1066.9,tau:1087.9}},
    "Normal":{{arc:5987.84,kappa:1109.21,tau:1145.70}},
    "Idn":{{arc:10737.06,kappa:1640.40,tau:1600.79}},
    "Lrelu":{{arc:4848.45,kappa:1135.52,tau:1169.51}},
    "Tanh":{{arc:10699.30,kappa:1686.24,tau:1654.65}},
    "Sigmoid":{{arc:10752.45,kappa:1707.03,tau:1678.86}},
  }},
  "lrelu": {{
    1:{{arc:696.2,kappa:822.4,tau:847.9}},2:{{arc:631.6,kappa:858.9,tau:858.4}},
    3:{{arc:636.6,kappa:871.8,tau:855.4}},4:{{arc:708.7,kappa:859.1,tau:880.6}},
    5:{{arc:776.5,kappa:744.5,tau:831.1}},6:{{arc:698.5,kappa:835.6,tau:855.1}},
    7:{{arc:696.2,kappa:714.0,tau:768.0}},8:{{arc:658.5,kappa:776.6,tau:803.3}},
    9:{{arc:716.0,kappa:827.5,tau:850.6}},10:{{arc:628.0,kappa:852.5,tau:842.6}},
    11:{{arc:635.0,kappa:855.2,tau:859.6}},12:{{arc:677.8,kappa:831.6,tau:867.3}},
    13:{{arc:631.0,kappa:835.4,tau:838.0}},14:{{arc:639.5,kappa:860.7,tau:858.6}},
    15:{{arc:634.1,kappa:864.7,tau:842.7}},16:{{arc:716.1,kappa:840.6,tau:867.6}},
    17:{{arc:675.5,kappa:829.7,tau:845.5}},18:{{arc:717.5,kappa:839.1,tau:865.5}},
    19:{{arc:620.9,kappa:811.9,tau:799.0}},20:{{arc:686.9,kappa:818.0,tau:846.0}},
    21:{{arc:635.2,kappa:859.9,tau:851.7}},22:{{arc:606.0,kappa:763.4,tau:772.6}},
    23:{{arc:625.6,kappa:799.1,tau:803.5}},24:{{arc:707.9,kappa:862.6,tau:887.9}},
    25:{{arc:629.0,kappa:832.2,tau:833.4}},26:{{arc:627.8,kappa:833.5,tau:822.6}},
    27:{{arc:711.7,kappa:835.7,tau:854.5}},28:{{arc:755.4,kappa:784.4,tau:818.8}},
    29:{{arc:714.2,kappa:759.9,tau:808.4}},30:{{arc:623.6,kappa:833.8,tau:831.5}},
    31:{{arc:13.4,kappa:72.7,tau:32.9}},32:{{arc:728.3,kappa:821.0,tau:853.8}},
    33:{{arc:628.9,kappa:856.6,tau:855.6}},34:{{arc:634.1,kappa:823.5,tau:830.7}},
    35:{{arc:663.5,kappa:804.3,tau:816.7}},36:{{arc:693.7,kappa:903.7,tau:913.2}},
    37:{{arc:723.6,kappa:814.4,tau:827.3}},38:{{arc:689.8,kappa:841.4,tau:850.7}},
    39:{{arc:629.7,kappa:840.8,tau:830.9}},40:{{arc:669.5,kappa:842.7,tau:858.3}},
    41:{{arc:724.9,kappa:854.7,tau:878.7}},42:{{arc:732.7,kappa:829.3,tau:853.7}},
    43:{{arc:695.0,kappa:811.6,tau:816.9}},44:{{arc:701.5,kappa:824.9,tau:840.9}},
    45:{{arc:673.5,kappa:836.7,tau:852.6}},46:{{arc:624.2,kappa:837.0,tau:831.2}},
    47:{{arc:668.1,kappa:820.5,tau:844.8}},48:{{arc:609.3,kappa:795.3,tau:799.7}},
    49:{{arc:688.1,kappa:865.7,tau:885.3}},50:{{arc:637.0,kappa:780.1,tau:781.6}},
    "Normal":{{arc:738.93,kappa:829.89,tau:875.69}},
    "Idn":{{arc:1140.51,kappa:1679.56,tau:1644.41}},
    "Lrelu":{{arc:758.08,kappa:875.49,tau:985.63}},
    "Tanh":{{arc:1119.02,kappa:1724.59,tau:1720.60}},
    "Sigmoid":{{arc:1129.25,kappa:1732.13,tau:1740.88}},
  }},
  "tanh": {{
    1:{{arc:508.3,kappa:1266.3,tau:1350.0}},2:{{arc:475.4,kappa:1359.4,tau:1421.1}},
    3:{{arc:486.5,kappa:1363.8,tau:1441.5}},4:{{arc:502.6,kappa:1340.9,tau:1398.7}},
    5:{{arc:406.9,kappa:1068.2,tau:1152.7}},6:{{arc:522.3,kappa:1343.1,tau:1401.0}},
    7:{{arc:433.0,kappa:1176.1,tau:1239.3}},8:{{arc:454.2,kappa:1281.2,tau:1350.1}},
    9:{{arc:506.7,kappa:1302.2,tau:1369.5}},10:{{arc:486.5,kappa:1364.3,tau:1430.8}},
    11:{{arc:477.5,kappa:1347.4,tau:1414.6}},12:{{arc:456.3,kappa:1251.6,tau:1336.3}},
    13:{{arc:476.0,kappa:1327.2,tau:1394.4}},14:{{arc:484.8,kappa:1343.3,tau:1420.6}},
    15:{{arc:475.8,kappa:1356.4,tau:1416.4}},16:{{arc:520.8,kappa:1338.4,tau:1401.2}},
    17:{{arc:467.5,kappa:1290.1,tau:1363.8}},18:{{arc:499.7,kappa:1338.3,tau:1392.2}},
    19:{{arc:469.7,kappa:1354.4,tau:1412.8}},20:{{arc:498.4,kappa:1346.1,tau:1399.4}},
    21:{{arc:488.0,kappa:1371.1,tau:1431.5}},22:{{arc:447.7,kappa:1290.9,tau:1336.9}},
    23:{{arc:460.2,kappa:1316.9,tau:1365.9}},24:{{arc:523.3,kappa:1342.1,tau:1421.9}},
    25:{{arc:459.0,kappa:1325.8,tau:1387.2}},26:{{arc:477.0,kappa:1324.4,tau:1398.2}},
    27:{{arc:521.4,kappa:1312.0,tau:1383.3}},28:{{arc:405.2,kappa:1170.8,tau:1172.1}},
    29:{{arc:437.6,kappa:1161.0,tau:1221.4}},30:{{arc:476.9,kappa:1358.7,tau:1436.2}},
    31:{{arc:5.6,kappa:71.5,tau:30.0}},32:{{arc:516.9,kappa:1307.1,tau:1370.3}},
    33:{{arc:484.0,kappa:1344.8,tau:1424.3}},34:{{arc:454.0,kappa:1319.0,tau:1381.8}},
    35:{{arc:469.3,kappa:1312.7,tau:1378.4}},36:{{arc:510.9,kappa:1351.1,tau:1419.9}},
    37:{{arc:500.5,kappa:1314.6,tau:1368.5}},38:{{arc:488.8,kappa:1318.4,tau:1373.5}},
    39:{{arc:476.9,kappa:1341.5,tau:1414.2}},40:{{arc:507.1,kappa:1346.5,tau:1406.9}},
    41:{{arc:512.8,kappa:1296.2,tau:1364.1}},42:{{arc:506.0,kappa:1309.5,tau:1354.7}},
    43:{{arc:502.2,kappa:1324.0,tau:1385.0}},44:{{arc:498.5,kappa:1329.3,tau:1399.8}},
    45:{{arc:470.1,kappa:1292.8,tau:1377.0}},46:{{arc:481.8,kappa:1369.4,tau:1438.6}},
    47:{{arc:478.0,kappa:1282.1,tau:1365.5}},48:{{arc:463.2,kappa:1279.2,tau:1325.1}},
    49:{{arc:505.2,kappa:1363.0,tau:1418.2}},50:{{arc:463.2,kappa:1258.9,tau:1304.6}},
    "Normal":{{arc:519.44,kappa:1294.60,tau:1383.52}},
    "Idn":{{arc:709.90,kappa:2596.43,tau:2451.18}},
    "Lrelu":{{arc:391.40,kappa:1166.07,tau:1357.58}},
    "Tanh":{{arc:983.92,kappa:1624.61,tau:1583.23}},
    "Sigmoid":{{arc:971.99,kappa:1624.40,tau:1570.75}},
  }},
  "sigmoid": {{
    1:{{arc:59.0,kappa:1140.3,tau:1150.9}},2:{{arc:54.9,kappa:1221.0,tau:1234.2}},
    3:{{arc:56.0,kappa:1236.2,tau:1245.8}},4:{{arc:58.6,kappa:1186.1,tau:1197.4}},
    5:{{arc:64.4,kappa:948.1,tau:971.0}},6:{{arc:59.5,kappa:1181.1,tau:1197.3}},
    7:{{arc:61.3,kappa:1016.5,tau:1040.9}},8:{{arc:58.0,kappa:1127.9,tau:1149.8}},
    9:{{arc:58.6,kappa:1163.4,tau:1178.6}},10:{{arc:56.1,kappa:1238.2,tau:1249.3}},
    11:{{arc:56.5,kappa:1211.0,tau:1226.6}},12:{{arc:58.5,kappa:1147.5,tau:1164.7}},
    13:{{arc:56.0,kappa:1194.4,tau:1206.9}},14:{{arc:56.5,kappa:1221.2,tau:1231.7}},
    15:{{arc:55.7,kappa:1229.3,tau:1234.7}},16:{{arc:60.5,kappa:1169.4,tau:1183.0}},
    17:{{arc:58.8,kappa:1154.5,tau:1173.9}},18:{{arc:58.9,kappa:1165.4,tau:1180.9}},
    19:{{arc:54.3,kappa:1204.7,tau:1212.4}},20:{{arc:57.4,kappa:1166.7,tau:1181.6}},
    21:{{arc:56.6,kappa:1228.7,tau:1239.2}},22:{{arc:54.0,kappa:1137.5,tau:1150.3}},
    23:{{arc:55.0,kappa:1166.7,tau:1178.8}},24:{{arc:59.2,kappa:1192.2,tau:1210.8}},
    25:{{arc:53.7,kappa:1191.7,tau:1204.8}},26:{{arc:55.6,kappa:1206.8,tau:1214.5}},
    27:{{arc:59.3,kappa:1171.9,tau:1185.6}},28:{{arc:64.7,kappa:1034.1,tau:1016.6}},
    29:{{arc:61.8,kappa:1021.1,tau:1028.0}},30:{{arc:55.4,kappa:1232.3,tau:1251.9}},
    31:{{arc:0.6,kappa:57.4,tau:28.4}},32:{{arc:61.0,kappa:1135.8,tau:1149.2}},
    33:{{arc:56.0,kappa:1226.4,tau:1245.5}},34:{{arc:55.2,kappa:1181.7,tau:1195.9}},
    35:{{arc:59.1,kappa:1172.5,tau:1180.7}},36:{{arc:58.6,kappa:1212.6,tau:1223.9}},
    37:{{arc:59.0,kappa:1151.4,tau:1164.7}},38:{{arc:57.5,kappa:1176.2,tau:1185.1}},
    39:{{arc:55.3,kappa:1210.7,tau:1224.2}},40:{{arc:58.3,kappa:1190.3,tau:1203.4}},
    41:{{arc:58.6,kappa:1146.8,tau:1160.5}},42:{{arc:59.9,kappa:1128.5,tau:1133.1}},
    43:{{arc:57.7,kappa:1182.5,tau:1195.5}},44:{{arc:57.2,kappa:1179.9,tau:1197.0}},
    45:{{arc:58.9,kappa:1179.6,tau:1199.1}},46:{{arc:55.7,kappa:1229.9,tau:1249.4}},
    47:{{arc:59.9,kappa:1172.0,tau:1190.9}},48:{{arc:53.8,kappa:1128.3,tau:1128.3}},
    49:{{arc:59.1,kappa:1198.2,tau:1205.8}},50:{{arc:53.8,kappa:1104.6,tau:1110.7}},
    "Normal":{{arc:59.82,kappa:1140.47,tau:1169.10}},
    "Idn":{{arc:118.95,kappa:1653.98,tau:1598.66}},
    "Lrelu":{{arc:48.91,kappa:1202.10,tau:1235.72}},
    "Tanh":{{arc:113.58,kappa:1698.42,tau:1654.00}},
    "Sigmoid":{{arc:115.96,kappa:1717.96,tau:1681.55}},
  }},
}};

// For stats lookup: original eps use numeric key, specials use label string
function getStatsKey(sliderVal, epData) {{
  if (epData.special) return epData.label;   // "Normal", "Idn", etc.
  const m = epData.label.match(/\\d+/);
  return m ? parseInt(m[0]) : null;
}}

function updateStats(act, sliderVal) {{
  const label = document.getElementById('stats-ep-label');
  const rows  = document.getElementById('stats-rows');

  if (sliderVal === NUM_EPISODES) {{
    label.textContent = '\u2014';
    rows.innerHTML = '<div id="stats-placeholder">Select an episode</div>';
    return;
  }}

  const epData   = spaces[act].ep[String(sliderVal)];
  const statsKey = getStatsKey(sliderVal, epData);
  const s        = statsKey !== null && STATS[act] ? STATS[act][statsKey] : null;
  const isSpecial = epData.special;
  const colour    = epData.colour ?? null;
  const avgR      = epData.avg_reward ?? null;

  label.textContent = epData.label;

  // Tile class: special episodes get a coloured left border
  const tileStyle = isSpecial && colour
    ? `class="stat-row special" style="border-left-color:${{colour}}"`
    : `class="stat-row"`;

  const avgTile = avgR !== null
    ? `<div ${{tileStyle}}>
         <span class="stat-name">Avg Reward</span>
         <span class="stat-val">${{avgR.toFixed(3)}}</span>
       </div>` : '';

  if (!s) {{
    rows.innerHTML = avgTile +
      `<div ${{tileStyle}}><span class="stat-name">Arc / \u03ba / \u03c4</span>
       <span class="stat-val" style="font-size:10px;color:#9A9490;font-weight:normal">no data</span></div>`;
    return;
  }}

  rows.innerHTML = avgTile + `
    <div ${{tileStyle}}>
      <span class="stat-name">Arc Length (FFF)</span>
      <span class="stat-val">${{s.arc.toFixed(1)}}</span>
    </div>
    <div ${{tileStyle}}>
      <span class="stat-name">Total Curvature \u03ba</span>
      <span class="stat-val">${{s.kappa.toFixed(1)}}</span>
    </div>
    <div ${{tileStyle}}>
      <span class="stat-name">Total Torsion \u03c4</span>
      <span class="stat-val">${{s.tau.toFixed(1)}}</span>
    </div>`;
}}

let currentAct    = ACT_NAMES[0];
let currentSlider = NUM_EPISODES;  // start on "Space"

const sidebar  = document.getElementById('sidebar');
const tsSection = document.getElementById('ts-section');
const statsSec  = document.getElementById('stats-section');

ACT_NAMES.forEach(act => {{
  const btn = document.createElement('button');
  btn.className   = 'act-btn' + (act === currentAct ? ' active' : '');
  btn.dataset.act = act;
  btn.innerHTML   = ACT_LABELS[act] + '<span class="mono">' + act + '</span>';
  btn.onclick     = () => switchAct(act);
  sidebar.insertBefore(btn, statsSec);
}});

const tsFrom      = document.getElementById('ts-from');
const tsTo        = document.getElementById('ts-to');
const tsFromLabel = document.getElementById('ts-from-label');
const tsToLabel   = document.getElementById('ts-to-label');
const tsLabel     = document.getElementById('ts-label');
const tsReset     = document.getElementById('ts-reset');

let currentFrom = 1;
let currentTo   = 1;

function syncTimestepSlider() {{
  if (currentSlider === NUM_EPISODES) {{
    tsSection.style.display = 'none';
    currentFrom = 1; currentTo = 1;
    return;
  }}
  const ep = spaces[currentAct].ep[String(currentSlider)];
  const maxStep = ep.traj.length;
  tsSection.style.display = 'flex';
  tsFrom.min = tsTo.min = 1;
  tsFrom.max = tsTo.max = maxStep;
  currentFrom = 1;
  currentTo   = maxStep;
  tsFrom.value = currentFrom;
  tsTo.value   = currentTo;
  updateTsLabels(maxStep);
}}

function updateTsLabels(maxStep) {{
  tsFromLabel.textContent = `From: ${{currentFrom}}`;
  tsToLabel.textContent   = `To:   ${{currentTo}}`;
  tsLabel.textContent     = `${{currentFrom}} \u2013 ${{currentTo}} / ${{maxStep}}`;
}}

tsFrom.addEventListener('input', () => {{
  currentFrom = Math.min(parseInt(tsFrom.value), currentTo);
  tsFrom.value = currentFrom;
  const ep = spaces[currentAct].ep[String(currentSlider)];
  updateTsLabels(ep.traj.length);
  Plotly.react(plotDiv, getTraces(currentAct, currentSlider, currentFrom, currentTo), getLayout());
}});
tsTo.addEventListener('input', () => {{
  currentTo = Math.max(parseInt(tsTo.value), currentFrom);
  tsTo.value = currentTo;
  const ep = spaces[currentAct].ep[String(currentSlider)];
  updateTsLabels(ep.traj.length);
  Plotly.react(plotDiv, getTraces(currentAct, currentSlider, currentFrom, currentTo), getLayout());
}});
tsReset.addEventListener('click', () => {{
  const ep = spaces[currentAct].ep[String(currentSlider)];
  currentFrom = 1;
  currentTo   = ep.traj.length;
  tsFrom.value = currentFrom;
  tsTo.value   = currentTo;
  updateTsLabels(ep.traj.length);
  Plotly.react(plotDiv, getTraces(currentAct, currentSlider, currentFrom, currentTo), getLayout());
}});

const plotDiv = document.getElementById('plot');

function getTraces(act, sliderVal, fromStep, toStep) {{
  const space = spaces[act];
  const p = space.p;
  const manifoldOpacity = (sliderVal === NUM_EPISODES) ? 1.0 : 0.15;

  const manifold = {{
    type: 'scatter3d', mode: 'markers', name: 'Manifold',
    x: p.map(r => r[0]), y: p.map(r => r[1]), z: p.map(r => r[2]),
    marker: {{
      size: 1,
      color: rewards,
      colorscale: [[0,'#ff0000'],[0.5,'#ffaa00'],[1,'#00cc44']],
      opacity: manifoldOpacity,
      colorbar: {{ title: 'Reward', thickness: 14, len: 0.5, tickfont: {{color:'#1A1A1E'}} }}
    }},
    hoverinfo: 'skip'
  }};

  if (sliderVal === NUM_EPISODES) {{
    return [manifold,
      {{ type:'scatter3d', mode:'lines+markers', name:'Trajectory', x:[], y:[], z:[],
         line:{{color:'#2C3E50', width:3}}, marker:{{size:3, color:'#2C3E50'}} }},
      {{ type:'scatter3d', mode:'markers', name:'Start', x:[], y:[], z:[],
         marker:{{size:8, color:'lime', symbol:'diamond'}} }},
      {{ type:'scatter3d', mode:'markers', name:'End', x:[], y:[], z:[],
         marker:{{size:8, color:'red', symbol:'cross'}} }},
    ];
  }}

  const ep       = space.ep[String(sliderVal)];
  const fullTraj = ep.traj;
  // Special episodes use their own colour; regular episodes use dark slate
  const trajColor = ep.special ? ep.colour : '#2C3E50';

  const f  = Math.max(0, (fromStep ?? 1) - 1);
  const t  = Math.min(fullTraj.length, toStep ?? fullTraj.length);
  const tr = fullTraj.slice(f, t);

  const startPt = tr.length > 0 ? tr[0]             : null;
  const endPt   = tr.length > 0 ? tr[tr.length - 1] : null;

  return [
    manifold,
    {{ type:'scatter3d', mode:'lines+markers', name:'Trajectory',
       x: tr.map(r=>r[0]), y: tr.map(r=>r[1]), z: tr.map(r=>r[2]),
       line: {{color:trajColor, width:4}},
       marker: {{size:3, color:trajColor, opacity:0.9}} }},
    {{ type:'scatter3d', mode:'markers', name:'Start',
       x: startPt?[startPt[0]]:[], y: startPt?[startPt[1]]:[], z: startPt?[startPt[2]]:[],
       marker: {{size:8, color:'lime', symbol:'diamond', line:{{color:'white', width:1}}}} }},
    {{ type:'scatter3d', mode:'markers', name:'End',
       x: endPt?[endPt[0]]:[], y: endPt?[endPt[1]]:[], z: endPt?[endPt[2]]:[],
       marker: {{size:8, color:'red', symbol:'cross', line:{{color:'white', width:1}}}} }},
  ];
}}

function buildSliderSteps() {{
  const steps = [];
  for (let i = 0; i <= NUM_EPISODES; i++) {{
    const label = i === NUM_EPISODES
      ? 'Space'
      : (spaces[currentAct].ep[String(i)]?.label ?? `Ep ${{i}}`);
    steps.push({{ label, method: 'skip', args: [] }});
  }}
  return steps;
}}

const layout = {{
  paper_bgcolor: '#FFFBF5',
  plot_bgcolor:  '#FFFBF5',
  font: {{ color: '#1A1A1E', family: 'Comic Sans MS, Comic Sans, sans-serif' }},
  title: {{
    text: `Neural Activation Manifold \u2014 <b>${{ACT_LABELS[currentAct]}}</b>`,
    font: {{ size: 15, color: '#1A1A1E' }},
    x: 0.5,
  }},
  scene: {{
    xaxis: {{ title: 'PCA 1', color: '#55555A', gridcolor: '#E6DFD3', showbackground: false }},
    yaxis: {{ title: 'PCA 2', color: '#55555A', gridcolor: '#E6DFD3', showbackground: false }},
    zaxis: {{ title: 'PCA 3', color: '#55555A', gridcolor: '#E6DFD3', showbackground: false }},
  }},
  margin: {{ l:0, r:0, b:90, t:50 }},
  legend: {{ x:0.01, y:0.98, bgcolor:'rgba(255,251,245,0.8)', font:{{color:'#1A1A1E'}} }},
  sliders: [{{
    active: NUM_EPISODES,
    currentvalue: {{ prefix:'Episode: ', font:{{size:13, color:'#55555A'}}, visible:true, xanchor:'center' }},
    pad: {{ t:10, b:10 }},
    bgcolor: '#F9EED8', bordercolor: '#B0C4DE', tickcolor: '#55555A',
    font: {{ color:'#2C3E50', size:9 }},
    steps: buildSliderSteps(),
    len: 0.97, x: 0.015, y: 0,
  }}]
}};

Plotly.newPlot(plotDiv, getTraces(currentAct, currentSlider, currentFrom, currentTo), layout, {{responsive: true}});
updateStats(currentAct, currentSlider);

plotDiv.on('plotly_sliderchange', (e) => {{
  currentSlider = e.slider.active;
  syncTimestepSlider();
  updateStats(currentAct, currentSlider);
  Plotly.react(plotDiv, getTraces(currentAct, currentSlider, currentFrom, currentTo), getLayout());
}});

function getLayout() {{
  return Object.assign({{}}, layout, {{
    title: {{ text:`Neural Activation Manifold \u2014 <b>${{ACT_LABELS[currentAct]}}</b>`, font:{{size:15,color:'#1A1A1E'}}, x:0.5 }},
    sliders: [Object.assign({{}}, layout.sliders[0], {{ active: currentSlider, steps: buildSliderSteps() }})]
  }});
}}

function switchAct(act) {{
  currentAct = act;
  document.querySelectorAll('.act-btn').forEach(b => {{
    b.classList.toggle('active', b.dataset.act === act);
  }});
  syncTimestepSlider();
  updateStats(currentAct, currentSlider);
  Plotly.react(plotDiv, getTraces(currentAct, currentSlider, currentFrom, currentTo), getLayout());
}}
</script>
</body>
</html>"""

out_path = path + "cartpole_manifold_interactive.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Saved to: {out_path}")