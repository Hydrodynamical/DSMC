import imageio
time = 45
# Filepaths for the frames
filenames = [f"frames/frame_{i:04d}.png" for i in range(time,-1, -1)]

# Create a GIF
gif_path = "backwards_animation.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
