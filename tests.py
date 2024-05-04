import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.random.randn(10000)
y = np.random.randn(10000)

# Create a 2D histogram
plt.hist2d(x, y, bins=30, cmap='inferno')

# Add a colorbar to show the color scale
plt.colorbar()

# Show the plot
plt.show()
