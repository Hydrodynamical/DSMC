"""Here we do a Direct Simulation Monte Carlo (DSMC) implementation.
See https://en.wikipedia.org/wiki/Direct_simulation_Monte_Carlo for the details.
https://web.archive.org/web/20110717092648/https://www.ipam.ucla.edu/schedule.aspx?pc=kttut 
 """
# import local modules
import config                   # model and simulation parameters
import statistics_functions     # statistical functions 
import numeric_functions        # specific numerical functions
import display_engine           # to display simulation results

# import built in or installed modules
import torch                    # pytorch for data
from copy import deepcopy       # to easily create time series without mutations           
from tqdm import tqdm           # for loading bar

# compute initial velocity of particles
print("Sampling initial velocities with N = ", config.N)
v = statistics_functions.sample_velocities(config.f_initial, config.N)

# first velocity history is initial v
velocity_history = [deepcopy(v)]

# for random time increments
time_counter = 0

# loading bar
print("Running simulation for n_total = ", config.n_total)
print("Time step: ", config.delta_t)
print("Knudsen number: ", config.epsilon)
print("Mass: ", config.rho)
print("Alpha: ", config.alpha)
print("C_alpha: ", config.C_alpha)

pbar = tqdm(total = config.n_total)

# main simulation loop
for n in range(1, config.n_total + 1):
    v = deepcopy(velocity_history[-1])                          # consider last velocity in history
    sigma = numeric_functions.upper_bound(config.B, v)          # see slide #29 of lecture 2 

    # stochastic collision times
    while(time_counter  < n * config.delta_t):                  # DOUBLE CHECK THIS
        i, j = statistics_functions.sample_pair(config.N)       # get random collisional pair
        xi = torch.rand(1)                                      # get random variable in [0,1)
        B_ij = config.B(v[i], v[j])                             # calculate cross section

        # acceptance/rejection for cutoff collision kernel
        if sigma * xi < B_ij:
            v_i_post, v_j_post = numeric_functions.collide(v[i], v[j])
            v[i].data.copy_(v_i_post)
            v[j].data.copy_(v_j_post)
            time_counter += (float(2 * config.epsilon) / float(config.N * config.rho * B_ij))
    velocity_history.append(v)
    pbar.update(1)

# stop loading bar
pbar.close()

# the time series representation is computationally intense
#display_engine.display_time_series(velocity_history)
    
print("Saving simulation as 'solution_animation.gif'.")
display_engine.save_2d_histograms_gif(velocity_history=velocity_history)
print("Simulation completed sucessfully!")