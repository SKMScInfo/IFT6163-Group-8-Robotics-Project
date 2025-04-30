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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

env = make_env(
    seed=42,
    maze_name="easy7x7a",
    train_visual_styles=True,
    random_start_end=True,
    propagate_seed_to_env=True,
)
print(env)

class ModelWithDropout(nn.Module):
    def __init__(self, state_dim, action_dim, dropout_p=0.5):
        super(ModelWithDropout, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def process_state(state):
    features = np.concatenate([
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
    return features

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
    variances.sort(key=lambda x: np.sum(x[1]), reverse=True)
    num_to_keep = int(len(variances) * alpha)
    return variances[:num_to_keep]

def aggregate_datasets(D, filtered_states, model):
    # Assume the filtered states are (state, variance) tuples
    new_data = []
    for state, _ in filtered_states:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # ADD PPO EXPERT HERE
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

def dadagger(env, n_iterations=25, alpha=0.1):
    action_dim = get_action_dim('ant')
    state_dim = get_state_dim('ant')
    D = []
    models = []
    model = ModelWithDropout(state_dim, action_dim)
    models.append(model)
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

# Run DADAgger
final_model = dadagger(env)
torch.save(final_model.state_dict(), 'model.pth')
# eval_dataset = load_data('')
env.close()
