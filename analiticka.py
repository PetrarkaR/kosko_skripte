import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a meshgrid for the plane
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Define the plane's Z-values (flat plane)
Z = np.zeros_like(X)

# Define the hole's radius and center
hole_radius = 1.0
hole_center = (-4, 0)

# Mask out the points within the hole radius
distance_from_center = np.sqrt((X - hole_center[0])**2 + (Y - hole_center[1])**2)
Z[distance_from_center < hole_radius] = np.nan  # Mask the hole with NaN

# Plot the plane with the hole
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, color='blue', edgecolor='grey', alpha=0.8)

# Adjust the view
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Visualization of a Plane with a Hole')

# Set limits for better visualization
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-1, 1])

plt.show()
