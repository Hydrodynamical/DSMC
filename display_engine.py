"""For displaying the results of the simulation.
`time_steps` is a list of timesteps
`velocity_history` is a list of lists, where each sublist represents the velocity of all particles at a given timestep.
"""
import matplotlib.pyplot as plt
import os
import imageio
from matplotlib.colors import LinearSegmentedColormap

# Define the colors and their corresponding positions in the colormap
colors = [(1, 1, 1), (0, 0, 0)]  # Black to White
positions = [0, 1]  # Positions for black and white

# Create the LinearSegmentedColormap
bw_cmap = LinearSegmentedColormap.from_list("black_white", list(zip(positions, colors)))

vmin = 0
vmax = 50

def display_time_series(velocity_history):
    """Display the velocity history as a time series, with lines connecting the particles."""
    plt.figure(figsize=(10, 10))
    
    # Assuming each sublist in `velocity_history` has the same number of elements (particles)
    time = len(velocity_history[0])
    
    # Generate a color map to distinguish different particles
    color_map = plt.cm.get_cmap('hsv', time)
    
    for i in range(time):
        # Extract v for the ith particle across all times steps
        v_x = [velocity[i][0] for velocity in velocity_history]
        v_y = [velocity[i][1] for velocity in velocity_history]

        plt.plot(v_x, v_y, '-o', color = color_map(i))
        plt.title('Particle velocities Over Time')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
    plt.show()

def plot_2d_histogram(velocity_history):
    """Display the histogram at each time. """
    time = len(velocity_history)
    for i in range(time):
        plt.figure(figsize=(10, 10))

         #Flatten the list of positions across all timesteps
        x_positions = [v[0].item() for v in velocity_history[i]]
        y_positions = [v[1].item() for v in velocity_history[i]]
        plt.hist2d(x_positions, y_positions, bins=10, cmap='viridis')
        plt.colorbar(label='Number of Particles')
        plt.title('Velocity Distribution')
        plt.xlabel('v_x')
        plt.ylabel('v_y')
        plt.show()

def save_2d_histograms_gif(velocity_history):
    """Save 2d_histograms """
    # make directory to store images
    os.makedirs("frames", exist_ok=True)

    # iterate over time
    time = len(velocity_history)
    for i in range(time):
        plt.figure(figsize=(10, 10))
        #Flatten the list of velocities across all timesteps
        v_x = [v[0].item() for v in velocity_history[i]]
        v_y = [v[1].item() for v in velocity_history[i]]
        plt.hist2d(v_x, v_y, bins=300, cmap=bw_cmap,      # coolwarm for color inverse
                   range = [[0, 1024], [0, 1024]],
                   vmin = vmin, vmax = vmax)
        plt.title(f"Time: {i}")
        plt.colorbar()

        # Save frame
        frame_filename = f"frames/frame_{i:04d}.png"
        plt.savefig(frame_filename)
        plt.close()

    # Filepaths for the frames
    filenames = [f"frames/frame_{i:04d}.png" for i in range(time)]

    # Create a GIF
    gif_path = "solution_animation.gif"
    with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
