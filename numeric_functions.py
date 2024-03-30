"""This module contains specific numerical functions used for DSMC."""
import torch

def upper_bound(B, velocities):
    """Computes the upper bound of the cross section given a set of velocities.
    Input:
        B function (the cross section)
        velocities = List[torch.size = [2]]
        
    Output:
         type = torch.tensor"""
    tensor_velocities = torch.stack(velocities, dim = 0)
    #print(f"tensor_velocities = {tensor_velocities}")

    mean_velocity = torch.sum(tensor_velocities, dim = 0)/torch.tensor(len(velocities))
    #print(f"mean_velocity = {mean_velocity}")

    variation = tensor_velocities - mean_velocity
   #print(f"variation = {variation}")

    delta_v = torch.max(variation) 
    #print(f"delta_v = {delta_v}")

    return B(delta_v, -delta_v)

def collide(v_i, v_j):
    """Performs hard sphere collision calculation. Uses random spherical sampling.
    Input:
        v_i = torch.size = [2]
        v_j = torch.size = [2]

    Output:
        v_i_post = torch.size = [2]
        v_j_post = torch.size = [2]
    """
    # obtain a random unit vector omega
    xi = torch.rand(1)
    omega = torch.tensor([ torch.cos( 2 * torch.pi * xi), torch.sin( 2 * torch.pi * xi)])

    # calculate collisions
    sum_velocity = v_i + v_j
    dif_velocity = v_i - v_j
    half_diff_norm = torch.tensor(0.5) * torch.norm(dif_velocity)
    v_i_post = torch.tensor(0.5) * sum_velocity + half_diff_norm * omega
    v_j_post = torch.tensor(0.5) * sum_velocity - half_diff_norm * omega

    return v_i_post, v_j_post
