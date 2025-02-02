import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from gymnasium import Env, spaces
import cv2
import os

class WarehouseRobotEnv(Env):
    def __init__(self, deterministic=True):
        super().__init__()
        
        self.grid_size = (6, 6)
        self.agent_pos = [0, 0]  
        self.package_pos = [2, 2]
        self.dropoff_pos = [0, 4]
        self.obstacles = [[1, 1], [1, 2], [2, 1]]  # Shelves
        self.has_package = False
        self.dropped_off = False
        self.message = "Moving..."  # Dynamic message at the top
        self.trail = []  # To store agent's path
        
        # Define action space: Up, Down, Left, Right, Pickup, Dropoff
        self.action_space = spaces.Discrete(6)
        
        # Observation space: (agent_pos, has_package)
        self.observation_space = spaces.Tuple((
            spaces.MultiDiscrete([6, 6]),
            spaces.Discrete(2)
        ))

        # Store frames for video creation
        self.frames = []

    def reset(self):
        self.agent_pos = [0, 0]
        self.has_package = False
        self.dropped_off = False
        self.message = "Moving..."
        self.trail = []  # Reset trail
        self.frames = []  # Reset frames
        return self._get_obs(), {}

    def step(self, action):
        reward = -1  # Default step penalty
        terminated = False

        # Store the current position in the trail before moving
        self.trail.append(tuple(self.agent_pos))

        # Movement actions (0-3)
        if action < 4:
            new_pos = self._move_agent(action)
            if not self._is_collision(new_pos):
                self.agent_pos = new_pos
            else:
                reward -= 20  # Collision penalty

        # Pickup (4)
        elif action == 4:
            if self.agent_pos == self.package_pos and not self.has_package:
                self.has_package = True
                self.package_pos = None  # Remove the package after pickup
                reward += 25
                self.message = "Package Picked Up!"  # Update message

        # Dropoff (5)
        elif action == 5:
            if self.agent_pos == self.dropoff_pos and self.has_package:
                self.dropped_off = True  # Set the flag for rendering text
                terminated = True
                reward += 100
                self.message = "Package Delivered!"  # Update message

        self.render()  # Save frame instead of showing it

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return (tuple(self.agent_pos), int(self.has_package))

    def _move_agent(self, action):
        moves = {
            0: [-1, 0],  # Up
            1: [1, 0],   # Down
            2: [0, -1],  # Left
            3: [0, 1]    # Right
        }
        delta = moves[action]
        new_pos = [
            self.agent_pos[0] + delta[0],
            self.agent_pos[1] + delta[1]
        ]
        # Clip to grid bounds
        new_pos[0] = np.clip(new_pos[0], 0, 5)
        new_pos[1] = np.clip(new_pos[1], 0, 5)
        return new_pos

    def _is_collision(self, pos):
        return pos in self.obstacles

    def render(self):
        frame_dir = '/Users/nidhi7/Documents/Buffalo/Reinforcement Learning/frames'
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)

        # Draw grid
        for x in range(7):
            ax.axvline(x, color='gray', linewidth=0.5)
        for y in range(7):
            ax.axhline(y, color='gray', linewidth=0.5)

        # Draw the trail (agent's path)
        for pos in self.trail:
            ax.add_patch(Rectangle((pos[1], 5 - pos[0]), 1, 1, facecolor='lightgray', alpha=0.5))

        # Draw agent
        ax.add_patch(Rectangle(
            (self.agent_pos[1], 5 - self.agent_pos[0]), 1, 1,
            facecolor='blue' if self.has_package else 'red'
        ))

        # Draw package only if it hasn’t been picked up
        if self.package_pos:
            ax.add_patch(Rectangle((2, 3), 1, 1, facecolor='green'))  # Package
        
        # Draw drop-off point
        ax.add_patch(Rectangle((4, 5), 1, 1, facecolor='yellow'))  # Dropoff

        # Draw obstacles
        for obs in self.obstacles:
            ax.add_patch(Rectangle(
                (obs[1], 5 - obs[0]), 1, 1, facecolor='black'
            ))

        # Display dynamic message at the top
        ax.text(0, 6.5, self.message, fontsize=10, color='blue', fontweight='bold')

        plt.axis('off')

        # Save frame
        frame_path = f'/Users/nidhi7/Documents/Buffalo/Reinforcement Learning/frames/frame_{len(self.frames)}.png'

        plt.savefig(frame_path, bbox_inches='tight')
        self.frames.append(frame_path)
        plt.close(fig)

    def save_video(self, filename="warehouse_robot_simulation.mp4"):
        """Create a video from the saved frames."""
        if not self.frames:
            print("No frames to save.")
            return
        print(f"Number of frames captured: {len(self.frames)}")

        frame = cv2.imread(self.frames[0])
        if frame is None:
            print("Error: Frames could not be read. Check if they were saved correctly.")
            return

        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        video = cv2.VideoWriter(filename, fourcc, 10, (width, height))

        for frame_path in self.frames:
            video.write(cv2.imread(frame_path))

        video.release()
        print(f"Video saved as {filename}")

        # Clean up frames
        for frame_path in self.frames:
            os.remove(frame_path)

# Example usage:
env = WarehouseRobotEnv()
obs, _ = env.reset()

for step in range(10000):  # Run for at least 10 timesteps
    action = env.action_space.sample()  # Choose a random action
    new_obs, reward, terminated, _, _ = env.step(action)  # Take step

    if terminated:
        break  # Stop if the episode ends

env.save_video("warehouse_robot_simulation.mp4")  # Save video
