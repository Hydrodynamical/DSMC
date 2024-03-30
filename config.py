"""Defines parameters of algorithm"""
import torch

#########################################################
# Physical Parameters
#########################################################
# The equation we are solving is 
# \partial_t f = \frac{1}{\epsilon} [ Q_sigma(f,f)]
# Q_sigma = integral min{B(v, v_*), sigma}f(v')f(v_*') - f
epsilon = 1     # Knusen number
rho = 1         # total mass of system 
alpha = 1       # = 1 for hard spheres
C_alpha = 1    # B(v,v*) = C_alpha |v-v*|^alpha

#########################################################
# Simulation Input Parameters
#########################################################
N = 50_000        # number of MC paths drawn
n_total  = 60  # total number of times to run simulation
delta_t = 0.01 # time step size

#########################################################
# Initial Function
#########################################################
M = 3
f_initial = []

for i in range(M):
    # this value is to make sure 2 quartiles are in display range = [-20, 20]
    radians = torch.tensor(float(i)/float(M))
    mu_x = 10*torch.cos(2 * torch.pi * radians)
    mu_y = 10*torch.sin(2 * torch.pi * radians)
    sigma_x = 1
    sigma_y = 1
    f_initial.append([1/M, [mu_x, mu_y, sigma_x, sigma_y]])

def B(v_i, v_j):
    """Defines function which is the cross section. Input is two velocities (dimension = 3). Returns torch.tensor."""
    return C_alpha * ( torch.norm(v_i - v_j) ** alpha)
    
