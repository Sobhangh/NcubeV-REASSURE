import math
import importlib.util
from pathlib import Path
from time import time

import polytope as pc
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import onnx
from onnx2torch import convert
import torch

from REASSURE.REASSURE.Tools import get_linear_region

def polytope_area_2d(poly):
    verts = np.asarray(pc.extreme(poly), dtype=float)

    if verts.size == 0:
        raise ValueError("No vertices found. Polytope may be empty or unbounded.")
    if verts.shape[1] != 2:
        raise ValueError(f"Expected 2D polytope, got dimension {verts.shape[1]}.")

    # Order vertices counterclockwise
    center = verts.mean(axis=0)
    angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
    verts = verts[np.argsort(angles)]

    # Shoelace formula
    x = verts[:, 0]
    y = verts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area

def plot_polytope(poly, fig,color='blue', alpha=0.3, label=None):
    try:
        # Some polytope versions expose plot directly.
        pc.plot(poly)
    except AttributeError:
        # Fallback: plot polygon boundary from extreme points.
        verts = np.asarray(pc.extreme(poly))
        if verts.ndim == 2 and verts.shape[0] > 0:
            center = verts.mean(axis=0)
            angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
            order = np.argsort(angles)
            polyline = verts[order]
            polyline = np.vstack([polyline, polyline[0]])
            ax = fig.gca()
            ax.plot(polyline[:, 0], polyline[:, 1], linestyle="-", linewidth=2, color=color, label=label)
            ax.fill(polyline[:, 0], polyline[:, 1], alpha=alpha, color=color)
        else:
            return
            raise ValueError("Could not extract vertices from the polytope.")


def any_point_in_polytope(poly):
    # Option 1: center of largest inscribed ball (strictly inside if radius > 0)
    radius, center_raw = pc.cheby_ball(poly)
    center_arr = np.asarray(center_raw)
    center = np.array(
        [v for v in center_arr.reshape(-1)],
        dtype=np.float32,
    )
    center = torch.tensor(center, dtype=torch.float32)
    #print("Center:", center)
    if radius >= 0 and not torch.isnan(center).any():
        return center, radius  # radius > 0 means strictly interior
    #raise RuntimeError("Polytope appears infeasible.")

def allHiddenNeurons(self, x):
    hidden_neurons = []
    x = x.view(2)
    for i, layer in enumerate(self):
        if i == len(self) - 1:
            continue
        if i == 0:
            x = layer(x)
        else:
            x = layer(F.relu(x))
        hidden_neurons.append(x)
    return torch.cat(hidden_neurons, dim=-1)

onnx_model = onnx.load("ppo_acc_bigger_200000_steps.onnx")
torch_model = convert(onnx_model)
torch_model = torch.nn.Sequential(
    torch_model.Gemm_0,
    torch_model.Relu_1,
    torch_model.Gemm_2,
    torch_model.Relu_3,
    torch_model.Gemm_4,
    torch_model.Relu_5,
    torch_model.Gemm_6,
    torch_model.Relu_7,
    torch_model.Gemm_8,
)

input_dim = 2
input_boundary = [np.block([[np.eye(input_dim)], [-np.eye(input_dim)]]),
                      np.block([np.array([100, 200]), np.array([0, 200])])]
#
with open('polytopes-approx-3-bound.pkl', 'rb') as f:
    polytopes = pickle.load(f)

buggy_inputs = torch.stack(list(map(lambda p: any_point_in_polytope(p)[0], list(filter(lambda p: any_point_in_polytope(p) is not None, polytopes)))))

def lin_f(buggy_input):
    buggy_input = torch.tensor([buggy_input[0], buggy_input[1]], dtype=torch.float32)
    buggy_input = buggy_input.view(1, -1)
    buggy_input.requires_grad = True
    return get_linear_region(
            buggy_input, allHiddenNeurons(torch_model,buggy_input).view(-1), input_boundary)

# get_linear_region returns (A, b), so keep regions as a list of tuples
linear_regions_nn = [lin_f(buggy_input) for buggy_input in buggy_inputs]
linear_regions_nn = [pc.Polytope(region[0], region[1]) for region in linear_regions_nn]


#ax = fig.add_subplot(111)
ratios =[]
for i in range(len(polytopes)):
    area_polytope = polytope_area_2d(polytopes[i])
    area_linear_region = polytope_area_2d(linear_regions_nn[i])
    ratio = area_polytope / area_linear_region
    ratios.append(ratio)
    #print(f"Ratio = {ratio:.4f}, Polytope {i}: Area = {area_polytope:.4f}, Linear Region Area = {area_linear_region:.4f}, ")
print(f"Average ratio = {sum(ratios)/len(ratios):.4f}, Max ratio = {max(ratios):.4f}, Min ratio = {min(ratios):.4f}")
#Average ratio = 0.8559, Max ratio = 1.0092, Min ratio = 0.0020

print("check buggy inputs are in polytopes:")
for i in range(len(polytopes)):
    if not (buggy_inputs[i] in polytopes[i]):
        print(f"Buggy input {buggy_inputs[i]} is not in polytope {i}.")
print("check complete")

set_upper_bound = 10
A = np.array([[1.0, 0.0],
              [-1.0, 0.0]])
b = np.array([set_upper_bound, 0.0])
position_bound_poly = pc.Polytope(A, b)

intersected_poly = []
for poly in polytopes:
    isect = poly.intersect(position_bound_poly)
    intersected_poly.append(isect)
    #if not pc.is_empty(isect):
        

# Plot histogram of area ratios
fig_hist = plt.figure(figsize=(8, 5))
ax_hist = fig_hist.gca()
ax_hist.hist(ratios, bins=20, color='steelblue', edgecolor='black', alpha=0.85)
ax_hist.set_title('Histogram of Polytope/Linear-Region Area Ratios')
ax_hist.set_xlabel('Area Ratio')
ax_hist.set_ylabel('Count')
ax_hist.grid(alpha=0.25)
plt.tight_layout()
plt.show()
plt.close(fig_hist)

for i in range(len(buggy_inputs)):
    fig = plt.figure(figsize=(12, 8))
    plot_polytope(intersected_poly[i], fig=fig, alpha=0.3, color='red', label='Intersected Polytope' )
    plot_polytope(polytopes[i], fig=fig, alpha=0.3, color='blue', label='Polytope' )
    #plot_polytope(linear_regions_nn[i], fig=fig, alpha=0.3, color='red', label='Linear Region' )
    ax = fig.gca()
    ax.plot(buggy_inputs[i][0], buggy_inputs[i][1], 'go', markersize=8, label='Buggy Input')
    
    fig.legend()
    fig.suptitle('Polytopes vs Linear Regions')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    

