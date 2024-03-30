"""This module is responsible for obtaining statistics for DSMC. """
import torch 
import random

def sample_gaussian(gaussian, N):
    """This function samples N velocities according to density gaussian using accept/reject sampling.
    Input:
        gaussian  = [mu_x, mu_y, sigma_x, sigma_y] is a 2d Gaussian with mean mu = [mu_x, mu_y] and variance sigma.
    
    Output:
        List[torch.size = [2]]"""
    velocities = []
    mu_x, mu_y, sigma_x, sigma_y = gaussian 
    for _ in range(N):
        xi_1 = torch.rand(1)
        xi_2 = torch.rand(1)

        # polar coordinate representation
        rho = torch.sqrt(- torch.tensor(2) * torch.log(xi_1))
        theta = 2 * torch.pi * xi_2

        # get velocity sample
        v_x = mu_x + sigma_x * rho * torch.cos(theta)
        v_y = mu_y + sigma_y * rho * torch.sin(theta)
        v = torch.tensor([v_x, v_y])

        # append to velocities 
        velocities.append(v)
    return velocities

def sample_pair(N):
    """Uniformly and independently sample integers i, j from 0, ..., N-1 """
    i = torch.randint(N, [1])
    j = torch.randint(N, [1])
    return i, j

def sample_velocities(f_initial, N):
    """Returns N samples from distribution f_initial.
    Input:
        f_initial = [ [weight0, gaussian0], 
                      [weight1, gaussian1],
                      ...]
        weight: float
        gaussian: [mu_x, mu_y, sigma_x, sigma_y]
        N: int
        
    Output:
        velocities: List"""
    velocities = []
    weights = [parameters[0] for parameters in f_initial]
    gaussians = [parameters[1] for parameters in f_initial]
    for _ in range(N):
        i = random.choices(range(len(weights)), weights=weights)[0]
        velocities.append(sample_gaussian(gaussian=gaussians[i], N=1)[0])
    return velocities

