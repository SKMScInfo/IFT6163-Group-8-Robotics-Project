# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../dmc_vision_benchmark')))
# import numpy as np
# import random
# # from dmc_vision_benchmark import modules
# from dmc_vision_benchmark.modules import mlp
# # from dmc_vision_benchmark.environments.ant_maze.maze_arenas import make_maze, make_maze_arena
# # from dmc_vision_benchmark.environments.ant_maze.maze_tasks import make_maze_task
# # from dm_control.locomotion.walkers import ant
# from dmc_vision_benchmark.environments.antmaze_env import make_env
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# from dm_control2gym import make

# def main():
#     env = make(
#         domain_name='antmaze',  # or 'walker', etc.
#         task_name='maze',       # adapt if needed
#         # seed=42,
#         visualize_reward=False,
#         # from_pixels=False,  # Set True if you want images
#         # height=84,
#         # width=84,
#         # camera_id=0,
#         # frame_skip=4
#     )

# if __name__ == '__main__':
#     main()

# class ReplayBuffer(Dataset):
#     """Simple dataset to store (state, action) pairs."""
#     def __init__(self):
#         self.states = []
#         self.actions = []

#     def add(self, states, actions):
#         self.states.extend(states)
#         self.actions.extend(actions)

#     def __len__(self):
#         return len(self.states)

#     def __getitem__(self, idx):
#         return self.states[idx], self.actions[idx]

# def compute_variance(ensemble_outputs):
#     # ensemble_outputs: (M, batch_size, action_dim)
#     return torch.var(ensemble_outputs, dim=0).mean(dim=1)  # (batch_size,)

# def select_high_variance(states, ensemble_outputs, top_alpha=0.2):
#     variance = compute_variance(ensemble_outputs)
#     num_select = int(top_alpha * len(states))
#     top_indices = torch.topk(variance, num_select).indices
#     selected_states = [states[i] for i in top_indices]
#     return selected_states

# def query_expert(states, expert_policy):
#     expert_actions = []
#     for state in states:
#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
#         with torch.no_grad():
#             action = expert_policy(state_tensor)
#         expert_actions.append(action.squeeze(0))
#     return expert_actions

# def train_ensemble(models, replay_buffer, batch_size=128, epochs=5):
#     dataloader = DataLoader(replay_buffer, batch_size=batch_size, shuffle=True)
#     for model in models:
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#         for epoch in range(epochs):
#             for states, actions in dataloader:
#                 pred = model(states)
#                 loss = F.mse_loss(pred, actions)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

# def dadagger(env_name, expert_policy, state_dim, action_dim, T=100, n_iters=10, M=5, alpha=0.2):
#     # Create environment
#     env = make_env(
#         seed=42,
#         maze_name="easy7x7a",
#         train_visual_styles=True,
#         random_start_end=True,
#         propagate_seed_to_env=True,
#     )
#     # maze = make_maze(name="easy7x7a")
#     # maze_arena = make_maze_arena(maze)
#     # maze_task = make_maze_task(task_name="shortest_path", walker=ant.Ant(), arena=maze_arena, cameras=)
#     # print(maze_arena)
#     print(env)
#     D = ReplayBuffer()
    
#     # Initialize ensemble of policies
#     models_ensemble = [mlp.MLP(state_dim, action_dim) for _ in range(M)]  # Simple MLPs for now

#     for i in range(n_iters):
#         print(f"Iteration {i+1}/{n_iters}")

#         # Rollout with π_i,1
#         states = []
#         obs = env.reset()
#         print(obs)
#         for t in range(T):
#             state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
#             with torch.no_grad():
#                 action = models_ensemble[0](state_tensor).squeeze(0).numpy()
#             next_obs, reward, done, info = env.step(action)
#             states.append(obs)
#             obs = next_obs
#             if done:
#                 obs = env.reset()

#         # Compute ensemble outputs
#         states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
#         ensemble_outputs = torch.stack([model(states_tensor) for model in models_ensemble])  # (M, T, action_dim)

#         # Select states with highest variance
#         selected_states = select_high_variance(states, ensemble_outputs, top_alpha=alpha)

#         # Query expert
#         expert_actions = query_expert(selected_states, expert_policy)

#         # Aggregate
#         D.add(selected_states, expert_actions)

#         # Retrain all models
#         train_ensemble(models_ensemble, D)
    
#     env.close()

#     return models_ensemble[0]  # Return best π̂ᵢ,₁

# if __name__ == '__main__':
#     dadagger('test', [], [], [])

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dmc_vision_benchmark.environments.antmaze_env import make_env
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from dmc_vision_benchmark.data.load_dmc_vb import load_data
from dmc_vision_benchmark.data.dmc_vb_info import get_action_dim, get_state_dim

# Get the current device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the device
print(device)

env = make_env(
    seed=42,
    maze_name="easy7x7a",
    train_visual_styles=True,
    random_start_end=True,
    propagate_seed_to_env=True,
)

# maze = make_maze(name="easy7x7a")
# maze_arena = make_maze_arena(maze)
# maze_task = make_maze_task(task_name="shortest_path", walker=ant.Ant(), arena=maze_arena, cameras=)
# print(maze_arena)
print(env)

class ModelWithDropout(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ModelWithDropout, self).__init__()
        # Example neural network architecture with dropout layers
        self.fc1 = nn.Linear(state_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout applied
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def process_state(state):
    # Extracting numerical features

    # 'distance_to_target': 1,
    # 'target_position': 3,
    # 'target_vector': 3,
    # 'walker/appendages_pos': 12,
    # 'walker/bodies_pos': 39,
    # 'walker/bodies_quats': 52,
    # 'walker/egocentric_target_vector': 3,
    # 'walker/end_effectors_pos': 12,
    # 'walker/joints_pos': 8,
    # 'walker/joints_vel': 8,
    # 'walker/sensors_accelerometer': 3,
    # 'walker/sensors_gyro': 3,
    # 'walker/sensors_touch': 9,
    # 'walker/sensors_velocimeter': 3,
    # 'walker/world_zaxis': 3,
    # 'walker_position': 3,
    # Flattening image data
    # lowres_top_camera_flat = state['lowres_top_camera'].flatten()
    # walker_egocentric_camera_flat = state['walker/egocentric_camera'].flatten()
    # follow_camera_flat = state['walker/follow_camera'].flatten()
    # overhead_camera_flat = state['walker/overhead_camera'].flatten()

    # Combine all numerical features into one array
    numerical_features = np.concatenate([
        np.array([state['distance_to_target']]),
        state['target_position'],
        state['target_vector'],
        state['walker/appendages_pos'],
        state['walker/bodies_pos'],
        state['walker/bodies_quats'],
        state['walker/egocentric_target_vector'],
        state['walker/end_effectors_pos'],
        state['walker/joints_pos'],
        state['walker/joints_vel'],
        state['walker/sensors_accelerometer'],
        state['walker/sensors_gyro'],
        state['walker/sensors_touch'],
        state['walker/sensors_velocimeter'],
        state['walker/world_zaxis'],
        state['walker_position']
    ])

    return numerical_features

def sample_trajectories(model, env, num_trajectories=10, trajectory_length=100):
    trajectories = []
    for _ in range(num_trajectories):
        state = env.reset()
        # print("STATE:\t", state)
        state = state.observation
        state = process_state(state)
        for _ in range(trajectory_length):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = model(state_tensor).detach().cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)
            print('REWARD:\t', reward)
            trajectory = (state, action, reward)
            state = next_state
            if done:
                break
        trajectories.append(trajectory)
    return trajectories

def get_variance(trajectories, model):
    variances = []
    for state, _, _ in trajectories:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        predictions = []
        # Generate multiple dropout-based predictions
        for _ in range(10):  # Number of dropout samples
            prediction = model(state_tensor).detach().cpu().numpy()[0]
            predictions.append(prediction)
        variance = np.var(predictions, axis=0)
        variances.append((state, variance))
    return variances

def filter_states(variances, alpha=0.1):
    # Sort states based on variance and select the top alpha% with highest variance
    variances.sort(key=lambda x: np.sum(x[1]), reverse=True)
    num_to_keep = int(len(variances) * alpha)
    return variances[:num_to_keep]

def aggregate_datasets(D, filtered_states, model):
    # Assume the filtered states are (state, variance) tuples
    new_data = []
    for state, _ in filtered_states:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = model(state_tensor).detach().cpu().numpy()[0]  # Use the model's action as expert action
        new_data.append((state, action))
    D.extend(new_data)

def train_model(model, dataset, num_epochs=1000):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    model.train()

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(num_epochs):
        for states, actions in data_loader:
            states = states.float()
            actions = actions.float()

            optimizer.zero_grad()
            predicted_actions = model(states)
            loss = criterion(predicted_actions, actions)
            loss.backward()
            optimizer.step()

# Main loop for DADAgger
def dadagger(env, n_iterations=25, alpha=0.1):
    action_dim = get_action_dim('ant')
    state_dim = get_state_dim('ant')
    D = []  # Initialize dataset
    model = ModelWithDropout(state_dim, action_dim)  # Initial model with dropout layers
    for i in tqdm(range(n_iterations)):
        # print(f"Iteration {i+1}/{n_iterations}")
        # Sample trajectories
        trajectories = sample_trajectories(model, env)
        # Get variance of predictions
        variances = get_variance(trajectories, model)
        # Filter states with highest variance
        filtered_states = filter_states(variances, alpha)
        print(len(D))
        # Aggregate the new dataset
        aggregate_datasets(D, filtered_states, model)
        print(len(D))
        
        # Train the model on the aggregated dataset
        train_model(model, D)
    return model

# if __name__ == '__main__':
# Run DADAgger
final_model = dadagger(env)
# torch.save(final_model.state_dict(), 'model.pth')
# eval_dataset = load_data('')
env.close()
