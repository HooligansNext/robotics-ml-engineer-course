"""
Generate a 2D point cloud shaped like a robot arrow, rotate it by a
sequence of angles using a rotation matrix, and plot before/after.
"""
import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix_2d(theta):
    """Returns the 2D rotation matrix for angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

# A "robot" point cloud (arrow shape)
points = np.array([
    [0, 0], [2, 0], [2, 0.5], [3, 0.5], [3, -0.5], [2, -0.5], [2, -1], [0, -1],
])  # shape (8, 2)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
angles = [0, np.pi / 4, np.pi / 2, np.pi]  # 0°, 45°, 90°, 180°

for ax, theta in zip(axes, angles):
    R = rotation_matrix_2d(theta)
    rotated = points @ R.T  # apply rotation to every point at once
    ax.plot(*rotated.T, 'o-')
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect('equal'); ax.grid(True)
    ax.set_title(f"θ = {np.degrees(theta):.0f}°")

plt.tight_layout()
plt.savefig("rotated_robot.png")
plt.show()

print("Verification — rotating by 90° then 90° more should equal 180°:")
R90 = rotation_matrix_2d(np.pi / 2)
R180 = rotation_matrix_2d(np.pi)
test_point = np.array([1, 0])
print(f"R90 @ R90 @ [1, 0] = {R90 @ R90 @ test_point}")
print(f"R180     @ [1, 0] = {R180 @ test_point}")