"""This module tests various functions defined elsewhere."""
import time
import torch
import matplotlib.pyplot as plt
from statistics_functions import sample_velocities
import numeric_functions as func
import config
from copy import deepcopy
from tqdm import tqdm

M = 2
f_initial = []

for _ in range(M):
    mu_x = 10*torch.randn(1)
    mu_y = 10*torch.randn(1)
    sigma_x = 10*torch.rand(1)
    sigma_y = 10*torch.rand(1)
    f_initial.append([1/M, [mu_x, mu_y, sigma_x, sigma_y]])

N = 1_000
momentum_tolerance = 0.000001
energy_tolerance =   0.000001

#############################
# sample_velocities 
#############################
start_time = time.time()    # record the start time

# sample vectors 
tensors = sample_velocities(f_initial = f_initial, N = N)

# record the end time 
end_time = time.time()

# print timing 
print(f"Execution time for sample_velocities with N = {N}: {end_time - start_time} seconds")

# convert the list of tensors into two lists of x and y coordinates
x_coords = [t[0].item() for t in tensors]
y_coords = [t[1].item() for t in tensors]

# plotting
plt.scatter(x_coords, y_coords)  # This plots the points

# Optionally, add labels and title
plt.xlabel('v_x')
plt.ylabel('v_y')
plt.title(f'sample_velocities: mu = {f_initial[:2]}, sigma = {f_initial[2:]}')

#############################
# upper_bound
#############################
print(f"Sigma = {func.upper_bound(config.B, tensors)}")

#############################
# collide
#############################
momentum_failure_counter = 0
energy_failure_counter = 0
counter = 0

# loading bar 
pbar = tqdm(total = N * (N-1)/2)

# check conservation laws
for i in range(N):
    for j in range(i):
        v_i = deepcopy(tensors[i])
        v_j = deepcopy(tensors[j])
        momentum_before = v_i + v_j
        energy_before = 0.5 * (torch.square(torch.norm(v_i)) + torch.square(torch.norm(v_j)))

        v_i_post, v_j_post = func.collide(v_i, v_j)

        momentum_after = v_i_post + v_j_post
        energy_after = 0.5 * (torch.square(torch.norm(v_i_post)) + torch.square(torch.norm(v_j_post)))

        if torch.norm(momentum_before - momentum_after) > momentum_tolerance:
            momentum_failure_counter += 1
        
        if torch.norm(energy_before - energy_after) > energy_tolerance:
            energy_failure_counter += 1

        counter += 1
        pbar.update(1)

print(f"\nN = {N}\nmomentum_tolerance = {momentum_tolerance}\nenergy_tolerance = {energy_tolerance}")
print(f"Conseration of momentum is valid {100 - (float(momentum_failure_counter)/float(counter))}% of the time.")
print(f"Conseration of energy is valid {100 - (float(energy_failure_counter)/float(counter))}% of the time.")

################
# Show the plot
################

if input("Show scatter plot of sampled velocities? (y/n) ") == "y":
    plt.show()
