import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create figure with explicit black background
fig = plt.figure(figsize=(12, 12), facecolor='black')
ax = fig.add_subplot(111, facecolor='black')
ax.set_aspect('equal')

# Define orbital parameters (adding semi-major and semi-minor axes for paths)
planets = {
    'Earth': {'distance': 1.0, 'period': 1.0, 'color': 'blue', 'size': 100, 'a': 1.0, 'b': 0.99},
    'Mars': {'distance': 1.52, 'period': 1.88, 'color': 'red', 'size': 80, 'a': 1.52, 'b': 1.50},
    'Jupiter': {'distance': 5.2, 'period': 11.86, 'color': 'orange', 'size': 200, 'a': 5.2, 'b': 5.15},
    'Saturn': {'distance': 9.58, 'period': 29.46, 'color': 'yellow', 'size': 180, 'a': 9.58, 'b': 9.50},
    'Uranus': {'distance': 19.18, 'period': 84.01, 'color': 'cyan', 'size': 160, 'a': 19.18, 'b': 19.10}
}

# Create the Sun
sun = plt.Circle((0, 0), 0.2, color='yellow', zorder=10)
ax.add_patch(sun)

# Plot static orbital paths (simplified ellipses)
theta = np.linspace(0, 2*np.pi, 100)
for planet, params in planets.items():
    # Parametric equation for ellipse: x = a*cos(t), y = b*sin(t)
    x = params['a'] * np.cos(theta)
    y = params['b'] * np.sin(theta)
    ax.plot(x, y, color=params['color'], linestyle='--', alpha=0.2, zorder=1)

# Create planet objects and trails
planet_objects = {}
trails = {}
for planet, params in planets.items():
    planet_objects[planet] = plt.Circle((params['distance'], 0),
                                      params['size']/1000,
                                      color=params['color'])
    ax.add_patch(planet_objects[planet])
    trails[planet], = ax.plot([], [], color=params['color'],
                            alpha=0.3, linewidth=1)

# Set plot limits
max_distance = max(p['distance'] for p in planets.values()) + 1
ax.set_xlim(-max_distance, max_distance)
ax.set_ylim(-max_distance, max_distance)

# Store trail data
trail_length = 50
trail_data = {planet: {'x': [], 'y': []} for planet in planets}

def update(frame):
    for planet, params in planets.items():
        # Using circular motion for animation (simplified)
        angle = 2 * np.pi * frame / (params['period'] * 100)
        x = params['distance'] * np.cos(angle)
        y = params['distance'] * np.sin(angle)
        planet_objects[planet].center = (x, y)

        trail_data[planet]['x'].append(x)
        trail_data[planet]['y'].append(y)
        if len(trail_data[planet]['x']) > trail_length:
            trail_data[planet]['x'].pop(0)
            trail_data[planet]['y'].pop(0)
        trails[planet].set_data(trail_data[planet]['x'],
                              trail_data[planet]['y'])

    return list(planet_objects.values()) + list(trails.values())

# Create animation
anim = FuncAnimation(fig, update, frames=1000,
                    interval=30, blit=True)

# Add labels
plt.title("Simplified Solar System Animation with Orbital Paths\n(Earth, Mars, Jupiter, Saturn, Uranus)",
         color='white', pad=20)
ax.text(0, 0, 'Sun', color='white', ha='center', va='center')

# Remove axes
ax.set_axis_off()

# Ensure tight layout
plt.tight_layout()

# Show the animation
plt.show()
