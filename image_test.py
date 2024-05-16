from PIL import Image
import numpy as np

# Load an image
# image = Image.open('frog.jpg')
image = Image.open('coffee.jpg')

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

# Convert back to PIL Image
image_greyscale = Image.fromarray(np_greyscale)

# Show or save the result
image_greyscale.show()

