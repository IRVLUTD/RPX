import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import sys, os

# Function to convert quaternion and position to 4x4 transformation matrix
def pose_to_matrix(position, quaternion):
    # Create rotation matrix from quaternion (x,y,z,w order)
    rot = Rotation.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
    # Use as_matrix() if available, otherwise fall back to as_dcm()
    try:
        rot_matrix = rot.as_matrix()
    except AttributeError:
        rot_matrix = rot.as_dcm()
    
    # Create 4x4 transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rot_matrix
    matrix[:3, 3] = position
    return matrix

# Lists to store data
positions = []
orientations = []

pose_folder = sys.argv[1]
num_poses = len(list(os.listdir(pose_folder)))
# Read all NPZ files
for i in range(num_poses):
    filename = f"{i:05d}.npz"
    file_path = os.path.join(pose_folder, filename)
    try:
        data = np.load(file_path)
        positions.append(data['position'])
        orientations.append(data['orientation'])
        print(f"Loaded {filename}")
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")

# Convert first pose to reference matrix (identity)
ref_matrix = np.eye(4)
ref_inv = np.linalg.inv(ref_matrix)

# Calculate relative poses
relative_positions = []
for pos, ori in zip(positions, orientations):
    # Create transformation matrix for current pose
    current_matrix = pose_to_matrix(pos, ori)
    
    # Calculate relative transformation
    relative_matrix = ref_inv @ current_matrix
    
    # Extract relative position
    relative_positions.append(relative_matrix[:3, 3])

# Convert to numpy array for easier handling
relative_positions = np.array(relative_positions)

# Create figure and 3D axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create color gradient (from blue to red)
colors = np.linspace(0, 1, len(relative_positions))
scatter = ax.scatter(relative_positions[:, 0], 
                    relative_positions[:, 1], 
                    relative_positions[:, 2], 
                    c=plt.cm.coolwarm(colors),  # Blue to red gradient
                    marker='o')

# Add colorbar
plt.colorbar(scatter, label='Progression (Start to End)')

# Set axis limits
ax.set_xlim(np.min(relative_positions[:, 0]), np.max(relative_positions[:, 0]))
ax.set_ylim(np.min(relative_positions[:, 1]), np.max(relative_positions[:, 1]))
ax.set_zlim(np.min(relative_positions[:, 2]), np.max(relative_positions[:, 2]))

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Relative Pose Plot with Color Gradient')

# Add grid
ax.grid(True)

# Show the plot
plt.show()