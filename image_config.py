"""Defines parameters of algorithm"""
import torch
from PIL import Image
import numpy as np

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
N = 500_000     # number of MC paths drawn
n_total  = 60 # total number of times to run simulation
delta_t = 0.001 # time step size

#########################################################
# Load image
#########################################################
# Load an image
image = Image.open('frog.jpg')

# Load the pixel map
pixels = image.load()

# Get the dimensions
width, height = image.size

# Convert PIL Image to NumPy array
np_image = np.array(image)

# Perform operations using NumPy (e.g., invert colors)
np_greyscale = np_image.mean(axis = 2)

# print(np_greyscale.shape)
# output = (340, 220)

# # Convert back to PIL Image
# image_greyscale = Image.fromarray(np_greyscale)

# # Show or save the result
# image_greyscale.show()
# print(np_greyscale)
#########################################################
# Initial Function
#########################################################
f_initial = []
height = 340
width = 220

for i in range(height):
    for j in range(width):
        # get coordinates of pixels
        # y = (40 * (height - i)/average_dimension)- 20
        # x = (40 * (width - j)/average_dimension) - 20

        # check if pixel is active 
        if np_greyscale[i,j] < 10:
            mu_x = j
            mu_y = height - i
            sigma_x = 0.3
            sigma_y = 0.3
            f_initial.append([1, [mu_x, mu_y, sigma_x, sigma_y]])

def B(v_i, v_j):
    """Defines function which is the cross section. Input is two velocities (dimension = 3). Returns torch.tensor."""
    return C_alpha * ( torch.norm(v_i - v_j) ** alpha)


    
